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

if __name__ == "__main__":
    test_ewise()
