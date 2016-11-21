import tinyflow as tf
from tinyflow.datasets import get_cifar10
import numpy as np

num_epoch  = 10
num_batch  = 600
batch_size = 100


def conv_factory(x, filter_size, in_filters, out_filters):
    x = tf.nn.conv2d(x, num_filter=out_filters,
            ksize=[1, filter_size, filter_size, 1], padding='SAME')
    x = tf.nn.batch_normalization(x)
    x = tf.nn.relu(x)
    return x

def residual_factory(x, in_filters, out_filters):
    if in_filters == out_filters:
        orig_x = x
        conv1 = conv_factory(x,     3, in_filters,  out_filters)
        conv2 = conv_factory(conv1, 3, out_filters, out_filters)
        new = orig_x + conv2
        return tf.nn.relu(new)
    else:
        conv1     = conv_factory(x,     3, in_filters,  out_filters)
        conv2     = conv_factory(conv1, 3, out_filters, out_filters)
        project_x = conv_factory(x,     1, in_filters,  out_filters)
        new = project_x + conv2
        return tf.nn.relu(new)

def resnet(x, n, in_filters, out_filters):
    for i in range(n):
        if i == 0:
            x = residual_factory(x, in_filters, 16)
        else:
            x = residual_factory(x, 16, 16)
    for i in range(n):
        if i == 0:
            x = residual_factory(x, 16, 32)
        else:
            x = residual_factory(x, 32, 32)
    for i in range(n):
        if i == 0:
            x = residual_factory(x, 32, 64)
        else:
            x = residual_factory(x, 64, 64)
    return x


x = tf.placeholder(tf.float32)
conv1 = tf.nn.conv2d(x, num_filter=16, ksize=[1, 5, 5, 1], padding='SAME')
tanh1 = tf.tanh(conv1)
res = resnet(tanh1, 1, 16, 64)
pool1 = tf.nn.avg_pool(res, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME', data_format='NCHW')
conv2 = tf.nn.conv2d(pool1, num_filter=16, ksize=[1, 5, 5, 1])
flatten = tf.nn.flatten_layer(conv2)
fc1 = tf.nn.linear(flatten, num_hidden=10, name="fc1")

# define loss
label = tf.placeholder(tf.float32)
cross_entropy = tf.nn.mean_sparse_softmax_cross_entropy_with_logits(fc1, label)
train_step = tf.train.AdamOptimizer(0.0005).minimize(cross_entropy)

sess = tf.Session(config='gpu')

# Auromatic variable shape inference API, infers the shape and initialize the weights.
known_shape = {x: [batch_size, 3, 32, 32], label: [batch_size]}
stdev = 0.01
init_step = []
for v, name, shape in tf.infer_variable_shapes(
        cross_entropy, feed_dict=known_shape):
    init_step.append(tf.assign(v, tf.normal(shape, stdev)))
    print("shape[%s]=%s" % (name, str(shape)))
sess.run(init_step)
sess.run(tf.initialize_all_variables())

# get the cifar dataset
cifar = get_cifar10()

for epoch in range(num_epoch):
    sum_loss = 0.0
    for i in range(num_batch):
        batch_xs, batch_ys = cifar.train.next_batch(batch_size)
        loss, _ = sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, label:batch_ys})
        sum_loss += loss
    print("epoch[%d] cross_entropy=%g" % (epoch, sum_loss /num_batch))

correct_prediction = tf.equal(tf.argmax(fc1, 1), label)
accuracy = tf.reduce_mean(correct_prediction)
print(sess.run(accuracy, feed_dict={x: cifar.test.images, label: cifar.test.labels}))
