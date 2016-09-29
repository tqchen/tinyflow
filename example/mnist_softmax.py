"""Tinyflow example code.

This code is adapted from Tensorflow's MNIST Tutorial with minimum code changes.
"""
import tinyflow as tf
from tinyflow.datasets import get_mnist

stdev = 0.01
# Create the model
x = tf.placeholder(tf.float32, [None, 784])

conv1_filter = tf.Variable(tf.normal([20, 3, 5, 5], stdev=stdev))
conv1_bias = tf.Variable(tf.normal([20], stdev=stdev))
conv1 = tf.nn.conv2d(conv1_filter, bias=conv1_bias)
flatten = tf.nn.flatten(conv1)
y = tf.nn.softmax(tf.matmul(flatten, W))

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session(device='gpu')
sess.run(tf.initialize_all_variables())

# get the mnist dataset
mnist = get_mnist(flatten=True, onehot=True)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(correct_prediction)

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
