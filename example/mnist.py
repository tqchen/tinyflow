import tinyflow as tf

x = tf.Variable(tf.zeros(shape=[1,2]))
W = tf.Variable(tf.zeros(shape=[1,2]))
b = tf.Variable(tf.zeros(shape=[1,2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

print(cross_entropy.debug_str())

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# example not yet working
