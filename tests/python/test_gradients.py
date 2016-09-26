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


if __name__ == "__main__":
    test_add_grad()
    pass
