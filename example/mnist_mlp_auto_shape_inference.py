"""TinyFlow Example code.

Automatic variable creation and shape inductions.
The network structure is directly specified via forward node numbers
The variables are automatically created, and their shape infered by tf.infer_variable_shapes
"""
import tinyflow as tf
from tinyflow.datasets import get_mnist

# Create the model
x = tf.placeholder(tf.float32)
fc1 = tf.nn.linear(x, num_hidden=100, name="fc1", no_bias=False)
relu1 = tf.nn.relu(fc1)
fc2 = tf.nn.linear(relu1, num_hidden=10, name="fc2")

# define loss
label = tf.placeholder(tf.float32)
cross_entropy = tf.nn.mean_sparse_softmax_cross_entropy_with_logits(fc2, label)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session(config='gpu')

# Automatic variable shape inference API, infers the shape and initialize the weights.
known_shape = {x: [100, 28 * 28], label: [100]}
init_step = []
for v, name, shape in tf.infer_variable_shapes(
        cross_entropy, feed_dict=known_shape):
    init_step.append(tf.assign(v, tf.normal(shape)))
    print("shape[%s]=%s" % (name, str(shape)))
sess.run(init_step)

# get the mnist dataset
mnist = get_mnist(flatten=True, onehot=False)

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
