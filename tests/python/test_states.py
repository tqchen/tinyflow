import tinyflow as tf
import numpy as np

def test_assign():
    x = tf.Variable(tf.zeros(shape=[2,3]))
    sess = tf.Session()
    sess.run(tf.assign(x, tf.zeros(shape=[2,3])))
    ax = sess.run(x)
    np.testing.assert_almost_equal(ax, np.zeros((2,3)))

def test_group():
    x1 = tf.Variable(tf.zeros(shape=[2,3]))
    x2 = tf.Variable(tf.zeros(shape=[2,3]))
    a1 = tf.assign(x1, tf.zeros(shape=[2,3]))
    a2 = tf.assign(x2, tf.ones(shape=[2,3]))
    sess = tf.Session()
    sess.run(tf.group(a1, a2))
    ax1 = sess.run(x1)
    ax2 = sess.run(x2)
    np.testing.assert_almost_equal(ax1, np.zeros((2,3)))
    np.testing.assert_almost_equal(ax2, np.ones((2,3)))

def test_init():
    x1 = tf.Variable(tf.ones(shape=[2,3]))
    x2 = tf.Variable(tf.zeros(shape=[2,3]))
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    ax1 = sess.run(x1)
    ax2 = sess.run(x2)
    np.testing.assert_almost_equal(ax1, np.ones((2,3)))
    np.testing.assert_almost_equal(ax2, np.zeros((2,3)))

if __name__ == "__main__":

    pass
