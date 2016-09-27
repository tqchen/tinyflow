import tinyflow as tf
import numpy as np

def test_add_grad():
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    ax = np.ones((2, 3))
    ay = np.ones((2, 3)) * 4
    z = x + y
    gx, gy = tf.gradients(z, [x, y])
    sess = tf.Session()
    agx = sess.run(gx, feed_dict={x:ax, y:ay})
    np.testing.assert_almost_equal(agx, np.ones((2,3)))

def test_mul_grad():
    x = tf.placeholder(tf.float32)
    ax = np.ones((2, 3))
    z = x * 14
    gx = tf.gradients(z, [x])[0]
    sess = tf.Session()
    agx = sess.run(gx, feed_dict={x:ax})
    np.testing.assert_almost_equal(agx, np.ones((2,3)) * 14)

def test_sum_grad():
    x = tf.placeholder(tf.float32)
    ax = np.ones((2, 3))
    z = -tf.reduce_sum(x) * 14
    gx = tf.gradients(z, [x])[0]
    sess = tf.Session()
    agx = sess.run(gx, feed_dict={x:ax})
    np.testing.assert_almost_equal(agx, -np.ones((2,3)) * 14)

def test_mean_grad():
    x = tf.placeholder(tf.float32)
    ax = np.ones((2, 3))
    z = -tf.reduce_mean(x) * 14
    gx = tf.gradients(z, [x])[0]
    sess = tf.Session()
    agx = sess.run(gx, feed_dict={x:ax})
    np.testing.assert_almost_equal(agx, -np.ones((2,3)) * 14 / 6.0)

def test_matmul_grad():
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    ax = np.ones((2, 3))
    ay = np.ones((3, 4)) * 4
    z = tf.matmul(x, y) * 4
    gx, gy = tf.gradients(z, [x, y])
    sess = tf.Session()
    agx = sess.run(gx, feed_dict={x:ax, y:ay})
    agy = sess.run(gy, feed_dict={x:ax, y:ay})
    np.testing.assert_almost_equal(
        agx,
        np.dot(np.ones((2,4)), ay.T) * 4)
    np.testing.assert_almost_equal(
        agy,
        np.dot(ax.T, np.ones((2,4))) * 4)


if __name__ == "__main__":
    test_mean_grad()
    pass
