import tinyflow as tf
import nnvm.graph as graph
import numpy as np

def myactivation(x):
    return (tf.exp(x) - tf.exp(-x)) / (tf.exp(x) + tf.exp(-x))

x = tf.placeholder(tf.float32)
tanh1 = myactivation(x)

ax = np.ones((2, 3))
sess = tf.Session("gpu fusion")
ay = sess.run(tanh1, feed_dict={x:ax})
print(ay)
