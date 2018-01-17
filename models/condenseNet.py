import os
import time
import shutil
from datetime import timedelta

import numpy as np
import tensorflow as tf


class CondenseNet:
    globalprogress = 0.0

    def __init__(self, data_provider, depth,
                 total_blocks, keep_prob,
                 weight_decay, nesterov_momentum, model_type, dataset,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 reduction=0.5,
                 bc_mode=False,
                 group=4,
                 condense_factor=4,
                 lasso_decay=0.1,
                 increasing_growth_rate=8,
                 **kwargs):
        """
        Class to implement networks from this paper
        https://arxiv.org/pdf/1611.05552.pdf

        Args:
            data_provider: Class, that have all required data sets
            growth_rate: `int`, variable from paper
            depth: `int`, variable from paper
            total_blocks: `int`, paper value == 3
            keep_prob: `float`, keep probability for dropout. If keep_prob = 1
                dropout will be disables
            weight_decay: `float`, weight decay for L2 loss, paper = 1e-4
            nesterov_momentum: `float`, momentum for Nesterov optimizer
            model_type: `str`, 'DenseNet' or 'DenseNet-BC'. Should model use
                bottle neck connections or not.
            dataset: `str`, dataset name
            should_save_logs: `bool`, should logs be saved or not
            should_save_model: `bool`, should model be saved or not
            renew_logs: `bool`, remove previous logs for current model
            reduction: `float`, reduction Theta at transition layer for
                DenseNets with bottleneck layers. See paragraph 'Compression'
                https://arxiv.org/pdf/1608.06993v3.pdf#4
            bc_mode: `bool`, should we use bottleneck layers and features
                reduction or not.
        """
        self.data_provider = data_provider
        self.data_shape = data_provider.data_shape
        self.n_classes = data_provider.n_classes
        self.depth = depth
        self.increasing_growth_rate = increasing_growth_rate
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.first_output_features = increasing_growth_rate * 2
        self.total_blocks = total_blocks
        self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        self.reduction = reduction
        if not bc_mode:
            print("Build %s model with %d blocks, "
                  "%d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block))
        if bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            print("Build %s model with %d blocks, "
                  "%d bottleneck layers and %d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block,
                      self.layers_per_block))
        print("Reduction at transition layers: %.1f" % self.reduction)
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.lasso_decay = lasso_decay
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        self.dataset_name = dataset
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs
        self.batches_step = 0

        self.group = group
        self.condense_factor = condense_factor
        self.stage = 0
        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())
        logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_writer = logswriter(self.logs_path)

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    @property
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = 'saves/%s' % self.model_identifier
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model.chkpt')
            self._save_path = save_path
        return save_path

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = 'logs/%s' % self.model_identifier
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        return "{}_growth_rate={}_depth={}_dataset_{}".format(
            self.model_type, self.increasing_growth_rate, self.depth, self.dataset_name)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path + 'something')
        except Exception as e:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.save_path)
        self.saver.restore(self.sess, self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    def log_loss_accuracy(self, loss, accuracy, epoch, prefix,
                          should_print=True):
        if should_print:
            print("mean cross_entropy: %f, mean accuracy: %f" % (
                loss, accuracy))
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(
                tag='accuracy_%s' % prefix, simple_value=float(accuracy))
        ])
        self.summary_writer.add_summary(summary, epoch)

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(tf.float32, shape=shape, name='input_images')
        self.labels = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='labels')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[])

    def composite_function(self, _input, out_features, kernel_size=3):
        """
        Function from paper H_l that performs:
            - batch normalization
            - ReLU nonlinearity
            - convolution with required kernel
            - dropout, if required
        """
        # BN
        output = self.batch_norm(_input)
        # Relu
        output = tf.nn.relu(output)
        # Conv
        output = self.groupconv2d(output, out_features=out_features, kernel_size=kernel_size, strides=[1, 1, 1, 1])
        # Dropout
        output = self.dropout(output)
        return output

    def bottleneck_condense(self, _input, out_features):
        output = self.batch_norm(_input)
        output = tf.nn.relu(output)
        with tf.variable_scope("learned_group_conv"):
            output = self.learned_group_conv2d(output, 1, out_features)
        output = self.dropout(output)
        return output

    def add_internal_layers(self, _input, growth_rate):
        """
        Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3*3 Conv layers
        with tf.variable_scope("bottleneck"):
            bottleneck_out = self.bottleneck_condense(_input, out_features=growth_rate * 4)
        shuffle_out = self.shufflelayers(bottleneck_out)
        comp_out = self.composite_function(shuffle_out, out_features=growth_rate)
        # concatenate _input with out from the composite layers
        output = tf.concat(axis=3, values=(_input, comp_out))
        return output

    def add_block(self, _input, growth_rate, layers_per_blocks):
        output = _input
        for layer in range(layers_per_blocks):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layers(output, growth_rate)
        return output

    def transition_layer(self, _input):
        #call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(_input, out_features=out_features, kernel_size=1)
        #run averagepooling
        output = self.avg_pooling(output, k=2)
        return output

    def transition_layer_to_classes(self, _input):
        """
        This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        output = self.batch_norm(_input)
        output = tf.nn.relu(output)
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pooling(output, k=last_pool_kernel)
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W = self.weight_variable_xavier([features_total, self.n_classes], name='w')
        bias = self.bias_variable([self.n_classes])
        logits = tf.matmul(output, W) + bias
        return logits

    def shufflelayers(self, _input):
        height = int(_input.get_shape()[1])
        width = int(_input.get_shape()[2])
        features_num = int(_input.get_shape()[3])
        features_per_group = features_num // self.group
        # transpose and shuffle
        _input = tf.reshape(_input, shape=[-1, height, width, features_per_group, self.group])
        _input = tf.transpose(_input, perm=[0, 1, 2, 4, 3])
        output = tf.reshape(_input, [-1, height, width, features_num])
        return output

    def groupconv2d(self, _input, out_features, kernel_size, strides):
        # group the feature maps
        with tf.variable_scope("groupConv"):
            features_num = int(_input.get_shape()[-1])
            grouped_features = tf.split(_input, num_or_size_splits=self.group, axis=3)
            group_ouputs = []
            for i in range(self.group):
                group_ouputs.append(self.conv2d(grouped_features[i],
                                                out_features=out_features // self.group,
                                                kernel_size=kernel_size,
                                                strides=strides,
                                                name='group_kernel_%d' % i))
            output = tf.concat(group_ouputs, axis=3)
            return output

    def learned_group_conv2d(self, _input, kernel_size, out_channels):
        in_channels = int(_input.get_shape()[-1])
        mask_scale = out_channels // self.group
        # check group and condense_factor
        assert _input.get_shape()[-1] % self.group == 0, "group number cannot be divided by input channels"
        assert _input.get_shape()[-1] % self.condense_factor == 0, "condensation factor can not be divided by input channels"
        weight = self.weight_variable_msra(shape=[kernel_size, kernel_size, in_channels, out_channels],
                                           name="weight")
        print(weight)
        mask = tf.get_variable("mask", [kernel_size, kernel_size, in_channels, out_channels],
                               initializer=tf.constant_initializer(1), trainable=False)
        output = tf.nn.conv2d(_input, tf.multiply(weight, mask), [1, 1, 1, 1], padding='SAME')
        assert output.get_shape()[-1] % self.group == 0, "group number can not be divided by output channels"
        return output

    def conv2d(self, _input, out_features, kernel_size, strides, padding='SAME', name='kernel'):
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(shape=[kernel_size, kernel_size, in_features, out_features],
                                           name=name)
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

    def avg_pooling(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(_input, scale=True, is_training=self.is_training, updates_collections=None)
        return output

    def dropout(self, _input):
        if self.keep_prob < 1 :
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    def _build_graph(self):
        growth_rate = self.increasing_growth_rate
        layers_per_block = self.layers_per_block
        # first initial 3x3 conv to first_output_features
        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                    self.images,
                    out_features=self.first_output_features,
                    kernel_size=3,
                    strides=[1, 1, 1, 1]
            )
        for block in range(self.total_blocks):
            growth_rate = growth_rate * (2**block)
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(output, growth_rate, layers_per_block)
            # last block exist without transition layers
            if block != self.total_blocks - 1 :
                with tf.variable_scope("Transition_layer_after_block_%d"% block):
                    output = self.transition_layer(output)

        with tf.variable_scope("Transition_to_classes"):
            logits = self.transition_layer_to_classes(output)
        prediction = tf.nn.softmax(logits)
        # Losses
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
        self.cross_entropy = cross_entropy
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        lasso_loss = self.lasso_loss()
        # optimizer and training step
        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.train_step = optimizer.minimize(cross_entropy + l2_loss * self.weight_decay + lasso_loss*self.lasso_decay)

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        total_start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            self.globalprogress = epoch / n_epochs
            print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
            start_time = time.time()
            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                learning_rate = learning_rate / 10
                print("Decrease learning rate, new lr = %f" % learning_rate)
            print("Training...")
            loss, acc = self.train_one_epoch(
                self.data_provider.train, batch_size, learning_rate)
            if self.should_save_logs:
                self.log_loss_accuracy(loss, acc, epoch, prefix='train')

            if train_params.get('validation_set', False):
                print("Validation...")
                loss, acc = self.test(
                    self.data_provider.validation, batch_size)
                if self.should_save_logs:
                    self.log_loss_accuracy(loss, acc, epoch, prefix='valid')

            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))
            if self._check_drop():
                self.dropping()
            if self.should_save_model:
                self.save_model()

        total_training_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(
            seconds=total_training_time)))

    def train_one_epoch(self, data, batch_size, learning_rate):
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            images, labels = batch
            feed_dict = {
                self.images: images,
                self.labels: labels,
                self.learning_rate: learning_rate,
                self.is_training: True,
            }
            fetches = [self.train_step, self.cross_entropy, self.accuracy]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, accuracy = result
            total_loss.append(loss)
            total_accuracy.append(accuracy)
            if self.should_save_logs:
                self.batches_step += 1
                self.log_loss_accuracy(
                    loss, accuracy, self.batches_step, prefix='per_batch',
                    should_print=False)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)

        return mean_loss, mean_accuracy

    def test(self, data, batch_size):
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            feed_dict = {
                self.images: batch[0],
                self.labels: batch[1],
                self.is_training: False,
            }
            fetches = [self.cross_entropy, self.accuracy]
            loss, accuracy = self.sess.run(fetches, feed_dict=feed_dict)
            total_loss.append(loss)
            total_accuracy.append(accuracy)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy

    def lasso_loss(self):
        """
        generate the lasso_loss regular item
        """
        lasso_array = []
        for i in range(self.total_blocks):
            for j in range(self.layers_per_block):
                with tf.variable_scope("Block_%d/layer_%d/bottleneck/learned_group_conv" % (i, j), reuse=True):
                    weight = tf.get_variable("weight")
                    mask = tf.get_variable("mask")
                    in_channels = int(weight.get_shape()[-2])
                    out_channels = int(weight.get_shape()[-1])
                    d_out = out_channels // self.group
                    assert weight.get_shape()[0] == 1
                    weight = tf.square(tf.squeeze(tf.multiply(weight, mask)))
                    weight = tf.reshape(weight, [in_channels, self.group, d_out])
                    weight = tf.sqrt(tf.reduce_sum(weight, axis=2))
                    weight = tf.reduce_sum(weight)
                    lasso_array.append(weight)
        lasso_loss = tf.add_n(lasso_array)
        return lasso_loss


    def _check_drop(self):
        progress = self.globalprogress
        old_stage = self.stage
        # get current stage
        for i in range(self.condense_factor - 1):
            if progress * 2 < (i + 1) / (self.condense_factor - 1):
                self.stage = i
                break
        else:
            self.stage = self.condense_factor - 1

        # Need Pruning?
        return old_stage != self.stage

    def dropping(self):
        print("stage_%d" % self.stage)
        for i in range(self.total_blocks):
            for j in range(self.layers_per_block):
                print("pruning the Block_%d/layer_%d/bottleneck/learned_group_conv" % (i, j))
                with tf.variable_scope("Block_%d/layer_%d/bottleneck/learned_group_conv" % (i, j), reuse=True):
                    weight = tf.get_variable("weight")
                    mask = tf.get_variable("mask")
                    in_channels = int(weight.get_shape()[-2])
                    d_in = in_channels // self.condense_factor
                    d_out = int(weight.get_shape()[-1]) // self.group
                    zeros = tf.zeros([d_out])
                    weight_s = tf.abs(tf.squeeze(tf.multiply(weight, mask)))
                    k = in_channels - (d_in * (self.stage-1))
                    print("k=%d" % k)
                    # Sort and Drop
                    for group in range(self.group):
                        wi = weight_s[:, group*d_out:(group + 1)*d_out]
                        # take corresponding delta index
                        _, index = tf.nn.top_k(tf.reduce_sum(wi, axis=1), k=k, sorted=True)
                        print("top_%d, d_in %d" % (k, d_in))
                        d = self.sess.run(index)
                        for _in in range(d_in):
                            # Assume only apply to 1x1 conv to speed up
                            self.sess.run(tf.assign(mask[0, 0, d[-(_in + 1)], group*d_out:(group + 1)*d_out], zeros))


    def weight_variable_msra(self, shape, name=None):
        return tf.get_variable(name=name,
                               shape=shape,
                               initializer=tf.contrib.layers.variance_scaling_initializer())

    def weight_variable_xavier(self, shape, name=None):
        return tf.get_variable(name=name,
                               shape=shape,
                               initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)








