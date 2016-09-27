import tinyflow as tf
import numpy as np

def check_ewise(ufunc):
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = ufunc(x, y)
    ax = np.ones((2, 3))
    ay = np.ones((2, 3)) * 4
    sess = tf.Session()
    az = sess.run(z, feed_dict={x:ax, y:ay})
    np.testing.assert_almost_equal(az, ufunc(ax, ay))

def test_ewise():
    check_ewise(lambda x, y: x+y)
    check_ewise(lambda x, y: x*y)

def test_softmax():
    x = tf.placeholder(tf.float32)
    y = tf.nn.softmax(x)
    ax = np.ones((2, 4))
    sess = tf.Session()
    ay = sess.run(y, feed_dict={x:ax})
    np.testing.assert_almost_equal(
        ay, ax / np.sum(ax, axis=1, keepdims=True))

def test_bias_add():
    x = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    y = tf.nn.bias_add(x, b)
    ax = np.random.uniform(size=(2, 3))
    ab = np.random.uniform(size=(3, ))
    sess = tf.Session()
    ay = sess.run(y, feed_dict={x:ax, b:ab})
    np.testing.assert_almost_equal(
        ay, ax + ab)

def test_matmul():
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    ax = np.ones((2, 3))
    ay = np.ones((3, 4)) * 4
    z = tf.matmul(x, y) * 4
    sess = tf.Session()
    az = sess.run(z, feed_dict={x:ax, y:ay})
    np.testing.assert_almost_equal(
        az, np.dot(ax, ay) * 4)

def test_sum():
    axis = [1, 3]
    x = tf.placeholder(tf.float32)
    y = tf.reduce_sum(x, reduction_indices=axis)
    ax = np.random.uniform(size=(2, 4, 8, 7))
    sess = tf.Session()
    ay = sess.run(y, feed_dict={x:ax})
    npy = ax.sum(axis=tuple(axis))
    assert(np.mean(np.abs(ay - npy))) < 1e-6

def test_mean():
    axis = [1, 3]
    x = tf.placeholder(tf.float32)
    y = tf.reduce_mean(x, reduction_indices=axis)
    ax = np.random.uniform(size=(2, 4, 8, 7))
    sess = tf.Session()
    ay = sess.run(y, feed_dict={x:ax})
    npy = ax.mean(axis=tuple(axis))
    assert(np.mean(np.abs(ay - npy))) < 1e-6

def test_argmax():
    x = tf.placeholder(tf.float32)
    y = tf.argmax(x, 1)
    ax = np.random.uniform(size=(700, 10))
    sess = tf.Session()
    ay = sess.run(y, feed_dict={x:ax})
    npy = np.argmax(ax, 1)
    assert(np.mean(np.abs(ay - npy))) < 1e-6

if __name__ == "__main__":
    test_ewise()
    test_sum()
    test_mean()
    test_matmul()
    test_bias_add()
    test_softmax()
    test_argmax()
    pass
