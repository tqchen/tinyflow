"""TinyFlow Example: LeNet for Digits classification.

This code uses automatic variable shape inference for shorter code.
"""
import tinyflow as tf
from tinyflow.datasets import get_mnist

# Create the model
x = tf.placeholder(tf.float32)
conv1 = tf.nn.conv2d(x, num_filter=20, ksize=[1, 5, 5, 1], name="conv1", no_bias=False)
tanh1 = tf.tanh(conv1)
pool1 = tf.nn.max_pool(tanh1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
conv2 = tf.nn.conv2d(pool1, num_filter=50, ksize=[1, 5, 5, 1], name="conv2", no_bias=False)
tanh2 = tf.tanh(conv2)
pool2 = tf.nn.max_pool(tanh2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
flatten = tf.nn.flatten_layer(pool2)
fc1 = tf.nn.linear(flatten, num_hidden=500, name="fc1")
tanh3 = tf.tanh(fc1)
fc2 = tf.nn.linear(tanh3, num_hidden=10, name="fc2")

# define loss
label = tf.placeholder(tf.float32)
cross_entropy = tf.nn.mean_sparse_softmax_cross_entropy_with_logits(fc2, label)
train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

sess = tf.Session(config='gpu')

# Auromatic variable shape inference API, infers the shape and initialize the weights.
known_shape = {x: [100, 1, 28, 28], label: [100]}
stdev = 0.01
init_step = []
for v, name, shape in tf.infer_variable_shapes(
        cross_entropy, feed_dict=known_shape):
    init_step.append(tf.assign(v, tf.normal(shape, stdev)))
    print("shape[%s]=%s" % (name, str(shape)))
sess.run(init_step)
sess.run(tf.initialize_all_variables())

# get the mnist dataset
mnist = get_mnist(flatten=False, onehot=False)

print_period = 1000
for epoch in range(10):
    sum_loss = 0.0
    num_batch = 600
    for i in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        loss, _ = sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, label:batch_ys})
        sum_loss += loss
    print("epoch[%d] cross_entropy=%g" % (epoch, sum_loss /num_batch))

correct_prediction = tf.equal(tf.argmax(fc2, 1), label)
accuracy = tf.reduce_mean(correct_prediction)
print(sess.run(accuracy, feed_dict={x: mnist.test.images, label: mnist.test.labels}))
