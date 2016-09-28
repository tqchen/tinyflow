"""Tinyflow example code.

Same as mnist_softmax.py but use sparse_softmax_cross_entropy_with_logits
"""
import tinyflow as tf
from tinyflow.datasets import get_mnist

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
logit = tf.matmul(x, W)

# Define loss and optimizer
label = tf.placeholder(tf.float32)

# use sparse_softmax_cross_entropy_with_logits
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit, label)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session(device='gpu')
sess.run(tf.initialize_all_variables())

# get the mnist dataset
mnist = get_mnist(flatten=True, onehot=False)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, label:batch_ys})

correct_prediction = tf.equal(tf.argmax(logit, 1), label)
accuracy = tf.reduce_mean(correct_prediction)

print(sess.run(accuracy, feed_dict={x: mnist.test.images, label: mnist.test.labels}))
