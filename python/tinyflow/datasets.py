"""auxiliary utility to get the dataset for demo"""
import numpy as np
from collections import namedtuple
from sklearn.datasets import fetch_mldata
import cPickle
import sys
import os
from subprocess import call


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


CIFAR10Data = namedtuple("CIFAR10Data", ["train", "test"])

def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding="bytes")
        # decode utf8
        for k, v in d.items():
            del(d[k])
        d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32).astype(np.float32)
    labels = np.array(labels, dtype="float32")
    return data, labels


def get_cifar10(swap_axes=False):
    path = "cifar-10-batches-py"
    if not os.path.exists(path):
        tar_file = "cifar-10-python.tar.gz"
        origin = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        if os.path.exists(tar_file):
            need_download = False
        else:
            need_download = True
        if need_download:
            call(["wget", origin])
            call(["tar", "-xvf", "cifar-10-python.tar.gz"])
        else:
            call(["tar", "-xvf", "cifar-10-python.tar.gz"])

    nb_train_samples = 50000

    X_train = np.zeros((nb_train_samples, 3, 32, 32), dtype="float32")
    y_train = np.zeros((nb_train_samples,), dtype="float32")

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        X_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    X_test, y_test = load_batch(fpath)

    if swap_axes:
        X_train = np.swapaxes(X_train, 1, 3)
        X_test  = np.swapaxes(X_test,  1, 3)

    return CIFAR10Data(train=ArrayPacker(X_train, y_train),
            test=ArrayPacker(X_test, y_test))
