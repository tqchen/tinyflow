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

def check_ewise_scalar(ufunc):
    x = tf.placeholder(tf.float32)
    y = 10;
    z = ufunc(x, y)
    ax = np.ones((2, 3))
    sess = tf.Session()
    az = sess.run(z, feed_dict={x:ax})
    np.testing.assert_almost_equal(az, ufunc(ax, y))

def check_ewise_rscalar(ufunc):
    x = 10;
    y = tf.placeholder(tf.float32)
    z = ufunc(x, y)
    ay = np.ones((2, 3))
    sess = tf.Session()
    az = sess.run(z, feed_dict={y:ay})
    np.testing.assert_almost_equal(az, ufunc(x, ay))

def test_ewise():
    check_ewise(lambda x, y: x+y)
    check_ewise(lambda x, y: x-y)
    check_ewise(lambda x, y: x*y)
    check_ewise(lambda x, y: x/y)
    check_ewise(lambda x, y: x**y)
    check_ewise_scalar(lambda x, y: x+y)
    check_ewise_scalar(lambda x, y: x-y)
    check_ewise_scalar(lambda x, y: x*y)
    check_ewise_scalar(lambda x, y: x/y)
    check_ewise_rscalar(lambda x, y: x-y)
    check_ewise_rscalar(lambda x, y: x**y)

def test_exp():
    x = tf.placeholder(tf.float32)
    y = tf.exp(x)
    ax = np.ones((2, 3)) * 2
    sess = tf.Session()
    ay = sess.run(y, feed_dict={x:ax})
    np.testing.assert_almost_equal(ay, np.exp(ax))

def test_log():
    x = tf.placeholder(tf.float32)
    y = tf.log(x)
    ax = np.ones((2, 3)) * 2
    sess = tf.Session()
    ay = sess.run(y, feed_dict={x:ax})
    np.testing.assert_almost_equal(ay, np.log(ax))

def test_sqrt():
    x = tf.placeholder(tf.float32)
    y = tf.sqrt(x)
    ax = np.ones((2, 3)) * 2
    sess = tf.Session()
    ay = sess.run(y, feed_dict={x:ax})
    np.testing.assert_almost_equal(ay, np.sqrt(ax))

def test_softmax():
    x = tf.placeholder(tf.float32)
    y = tf.nn.softmax(x)
    ax = np.ones((2, 4))
    sess = tf.Session()
    ay = sess.run(y, feed_dict={x:ax})
    np.testing.assert_almost_equal(
        ay, ax / np.sum(ax, axis=1, keepdims=True))

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

def test_pad():
    out_filter = 10
    in_filter  = 4
    pad_width = (out_filter-in_filter)//2
    x = tf.placeholder(tf.float32)
    y = tf.pad(x, dim=1, pad=-pad_width)
    z = tf.pad(y, dim=1, pad=pad_width)
    nx  = np.random.randn(100, 4, 28, 28)
    npy = np.pad(nx, ((0, 0), (pad_width, pad_width), (0, 0), (0, 0)),
            mode='constant', constant_values=0)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    ay = sess.run(z, feed_dict={x : nx})
    assert(np.mean(np.abs(ay - npy))) < 1e-6


if __name__ == "__main__":
    test_ewise()
    test_exp()
    test_log()
    test_sqrt()
    test_sum()
    test_mean()
    test_matmul()
    test_softmax()
    test_argmax()
    test_pad()
    pass
