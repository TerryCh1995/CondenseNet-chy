import math


def cos(learning_rate, n_epochs, epoch, n_batches, batch):
    t_total = n_epochs * n_batches
    t_cur = epoch * n_batches + batch
    return 0.5 * learning_rate * (1 + math.cos(math.pi * t_cur / t_total))

print(cos(0.1,10,5,500,0))


