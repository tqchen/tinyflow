import tinyflow as tf
import numpy as np
from tinyflow.datasets import get_mnist

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W))

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

# get the mnist dataset
mnist = get_mnist(flatten=True, onehot=True)

nstep = 100
for i in range(nstep):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    loss, _ = sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(correct_prediction)

ay = sess.run(y, feed_dict={x: mnist.train.images})
print np.mean(np.argmax(ay, axis=1) == np.argmax(mnist.train.labels, axis=1))
print(sess.run(correct_prediction, feed_dict={x: batch_xs, y_:batch_ys}))
