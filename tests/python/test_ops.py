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

def test_assign():
    x = tf.Variable(tf.zeros(shape=[2,3]))
    sess = tf.Session()
    sess.run(tf.assign(x, tf.zeros(shape=[2,3])))
    print sess.run(x)

def test_softmax():
    x = tf.placeholder(tf.float32)
    y = tf.nn.softmax(x)
    ax = np.ones((2, 3))
    sess = tf.Session()
    ay = sess.run(y, feed_dict={x:ax})
    print ay

if __name__ == "__main__":
    test_softmax()
