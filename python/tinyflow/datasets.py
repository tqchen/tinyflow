"""auxiliary utility to get the dataset for demo"""
import numpy as np
from collections import namedtuple
from sklearn.datasets import fetch_mldata

class ArrayPacker(object):
    """Dataset packer for iterator"""
    def __init__(self, X, Y):
        self.images = X
        self.labels = Y
        self.ptr = 0

    def next_batch(self, batch_size):
        if self.ptr + batch_size >= self.labels.shape[0]:
            self.ptr = 0
        X = self.images[self.ptr:self.ptr+batch_size]
        Y = self.labels[self.ptr:self.ptr+batch_size]
        self.ptr += batch_size
        return X, Y

MNISTData = namedtuple("MNISTData", ["train", "test"])

def get_mnist(flatten=False, onehot=False):
    mnist = fetch_mldata('MNIST original')
    np.random.seed(1234) # set seed for deterministic ordering
    p = np.random.permutation(mnist.data.shape[0])
    X = mnist.data[p]
    Y = mnist.target[p]
    X = X.astype(np.float32) / 255.0
    if flatten:
        X = X.reshape((70000, 28 * 28))
    else:
        X = X.reshape((70000, 1, 28, 28))
    if onehot:
        onehot = np.zeros((Y.shape[0], 10))
        onehot[np.arange(Y.shape[0]), Y.astype(np.int32)] = 1
        Y = onehot
    X_train = X[:60000]
    Y_train = Y[:60000]
    X_test = X[60000:]
    Y_test = Y[60000:]
    return MNISTData(train=ArrayPacker(X_train, Y_train),
                     test=ArrayPacker(X_test, Y_test))
