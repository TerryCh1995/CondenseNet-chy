import os
import time
import shutil
from datetime import timedelta

import numpy as np
import tensorflow as tf
import condenseNet


class LearnedGroupConv:
    globalprogress = 0.0
    stage = 0
    def __init__(self, in_channels, out_channels, kernel_size, weight, stride=1, padding='SAME',
                 dilation=1, groups=1, condense_factor=None, Dropout_rate=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.condense_factor = condense_factor
        self.keep_prob = Dropout_rate
        self.weight = weight
        # initialize the mask
        self.mask = tf.Variable(tf.ones(shape=[kernel_size, kernel_size, in_channels, out_channels]))
        # count means the number of remain weights of group
        self.count = self.in_channels
        # check the argument is valid
        assert self.in_channels % self.groups == 0, "group number cannot be divided by input channels"
        assert self.in_channels % self.condense_factor == 0, "condensation factor can not be divided by input channels"
        assert self.out_channels % self.groups == 0, "group number can not be divided by output channels"



    @property
    def _at_stage(self, stage):
        return self.stage == stage

    @property
    def lasso_loss(self):
        """
        generate the lasso_loss regular item
        """
        weight = tf.matmul(self.weight, self.mask)
        d_out = self.out_channels//self.groups
        assert self.weight.get_shape()[0] == 1
        weight = tf.squeeze(weight)
        weight = tf.square(weight)
        tf.reshape(weight, [self.in_channels, d_out, self.groups])
        weight = tf.sqrt(tf.reduce_sum(weight, axis=1))
        weight = tf.reduce_sum(weight)
        return weight

    def _check_drop(self):
        progress = LearnedGroupConv.globalprogress
        delta = 0
        # get current stage
        for i in range(self.condense_factor - 1):
            if progress * 2 < (i + 1) / (self.condense_factor - 1):
                stage = i
                break
        else:
            stage = self.condense_factor - 1

        # Need Pruning?
        if not self._at_stage(stage):
            self.stage = stage
            delta = self.in_channels // self.condense_factor

        if delta > 0:
            # prune
            self._dropping(delta)

    def _dropping(self, delta):
        """
        dropping the weight by using the mask
        shape of the weight : kernel_size, kernel_size, in_channels, out_channels
        """
        weight = tf.matmul(self.weight, self.mask)
        # Assume only apply to 1x1 conv to speed up
        assert weight.get_shape()[0] == 1
        weight = tf.abs(tf.squeeze(weight))
        assert weight.get_shape()[-2] == self.in_channels
        assert weight.get_shape()[-1] == self.out_channels
        # shuffle weight
        d_out = self.out_channels // self.groups
        weight = tf.reshape(weight,[self.in_channels, d_out, self.groups])
        weight = tf.transpose(weight, [0, 2, 1])
        weight = tf.reshape(weight, [self.in_channels, -1])
        # Sort and Drop
        for i in range(self.groups):
            wi = weight[:, i*d_out:(i+1)*d_out]
            # take corresponding delta index
            di, index = tf.nn.top_k(tf.reduce_sum(wi, axis=1), k=self.count, sorted=True)
            for d in index[-delta:]:
                tf.assign(self.mask[:, :, d, i::self.groups], 0)
        self.count = self.count - delta

    def forward(self.):



def ShuffleLayers(_input, groups):
    features_num = _input.get_shape[3]
    batch_size = _input.get_shape[0]
    height = _input.get_shape[1]
    width = _input.get_shape[2]
    features_per_group = features_num // groups
    # tranpose and shuffle
    _input = tf.reshape(_input, shape=[batch_size, height, width, features_per_group,groups])
    _input = tf.transpose(_input,perm=[0, 1, 2, 4, 3])
    output = tf.reshape(_input,[batch_size, height, width, -1])
    return output








