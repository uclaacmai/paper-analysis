import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

n_epochs = 20000
minibatch_size = 50
lr = 1e-4
dropout_prob = 0.5

logs_path = '/tmp/tensorflow_logs/noiselogs'

# define input variables
x = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 10],name='y-labels')

def weight_variable(shape, name = 'Weights'):
    """Initializes weights randomly from a normal distribution
    Params: shape: list of dimensionality of the tensor to be initialized
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = name)


def bias_variable(shape, name = 'bias var'):
    """Initializes the bias term randomly from a normal distribution.
    Params: shape: list of dimensionality for the bias term.
    """
    initial = tf.constant(0.1, shape=shape, name = 'Bias')
    return tf.Variable(initial, name = name)

def conv2d(x, W):
    """Performs a convolution over a given patch x with some filter W.
    Uses a stride of length 1 and SAME padding (padded with zeros at the edges)
    Params:
    x: tensor: the image to be convolved over
    W: the kernel (tensor) with which to convolve.
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name = 'max pooling'):
    """Performs a max pooling operation over a 2 x 2 region"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name = name)

x_image = tf.reshape(x, [-1,28,28,1], name = 'x-image-reshaped') # covert x to a 4-d tensor

with tf.name_scope('Model'):

	W_conv1 = weight_variable([5, 5, 1, 32], name = 'conv1-weights')
	b_conv1 = bias_variable([32], name='conv1-bias')
	# LAYER 1: convolution->ReLu -> max pooling -> local response norm
	with tf.name_scope('conv-layer-1'):
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name='conv-relu-1')
		h_pool1 = max_pool_2x2(h_conv1, name='max-pool-1')
		h_pool1 = tf.nn.local_response_normalization(h_pool1, name='local-response-norm-1')


	# define weights and biases for next convolution operation

	W_conv2 = weight_variable([5, 5, 32, 64], name='conv2-weights')
	b_conv2 = bias_variable([64], name = 'conv2-bias')

	# LAYER 2: convolution -> ReLu -> max pooling -> local response normalization
	with tf.name_scope('conv-layer-2'):
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name = 'conv-relu-2')
		h_pool2 = max_pool_2x2(h_conv2, name = 'max-pool-2')
		h_pool2 = tf.nn.local_response_normalization(h_pool2, name='local-response-norm-2')

	# add 1 x 1 convolution step to 128 feature maps
	W_conv3 = weight_variable([1,1,64,128], name = 'conv3-weights-1x1')
	b_conv3 = bias_variable([128], name = 'conv3-bias-1x1')
	with tf.name_scope('conv-layer-3-1x1'):
		h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3, name = 'conv-relu-3-1x1')

	# define weights and biases for the fully-connected layer
	W_fc1 = weight_variable([7 * 7 * 128, 1024], name = 'FC-layer-3-weights') # with 1024 neurons
	b_fc1 = bias_variable([1024], name = 'bias-FC-layer-3')

	# Layer 3: fully connected w/ReLu neurons
	with tf.name_scope('FC-layer-3'):
		h_pool2_flat = tf.reshape(h_conv3, [-1, 7*7*128])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name = 'FC-layer-3')

	# define drop out variables
	with tf.name_scope('dropout-layer-3'):
		keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name = 'dropout-layer-3')

	# define weights and biases for the next FC layer
	W_fc2 = weight_variable([1024, 256], name = 'weights-FC-layer-4') # 256 neurons
	b_fc2 = bias_variable([256], name = 'bias-FC-layer-4')

	# second fully connected layer
	with tf.name_scope('FC-layer-4'):
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='FC-layer-4')

	# dropout for second fc layer
	with tf.name_scope('dropout-layer-4'):
		h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob, name = 'dropout-layer-4')


	# define weights and biases for the softmax layer
	W_fc3 = weight_variable([256, 10], name = 'weights-softmax-layer') # 10 output units
	b_fc3 = bias_variable([10], name = 'bias-softmax-layer')

	# Output layer: 10-way softmax
	with tf.name_scope('softmax-output'):
		y_out = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3, name = 'softmax-output')

with tf.name_scope('Loss'):
	cross_entropy = -tf.reduce_sum(y_*tf.log(y_out))

with tf.name_scope('SGD'):
	lr = 1e-4
	train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
	correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cross_entropy)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
	sess.run(init)
	summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
	for i in range(n_epochs):
  		batch = mnist.train.next_batch(minibatch_size)
  		if i%100 == 0:
  			print("epoch: {}".format(i))
  			train_acc = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
  			print("train acc: {}".format(train_acc))
  		p_corrupt = 0.5
  		for i in range(batch[1].shape[0]):
  			if np.random.random() < p_corrupt:
  				label = np.argmax(batch[1][i], axis = 0)
  				opts = list(np.arange(10))
  				opts.remove(label)
  				new_label = np.random.choice(opts)
  				batch[1][i][label], batch[1][i][new_label] = 0, 1
  		_, c, summary = sess.run([train_step, cross_entropy, merged_summary_op], 
  			feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1 - dropout_prob})
  		summary_writer.add_summary(summary, i)
	test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
	print("test accuracy {}".format(test_acc))
	print("run tensorboard --logdir=/tmp/tensorflow_logs  and then open up localhost:6006")
	sess.close()




