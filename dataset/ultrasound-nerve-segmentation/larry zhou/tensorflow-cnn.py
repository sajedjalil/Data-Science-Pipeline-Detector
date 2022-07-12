import os
import tensorflow as tf
import cv2
IMAGE_DIR = '../input/train/'
IMAGES = [i for i in os.listdir(IMAGE_DIR) if 'mask' not in i]
IMAGE_ROWS = 420
IMAGE_COLS = 580
def read_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[-1]])
    h = tf.nn.relu(conv2d(input, W) + b)
    h_pool = max_pool_2x2(h)
    # dropout
    # keep_prob = tf.placeholder(tf.float32)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    return h_pool

def conn_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[-1]])
    return tf.nn.relu(tf.matmul(input, W) + b)
    
def training(loss, learning_rate):
  tf.scalar_summary(loss.op.name, loss)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  return optimizer.minimize(loss, global_step=global_step)
sess = tf.InteractiveSession()
# input
# 420x580
x = tf.placeholder(tf.float32, shape=[IMAGE_ROWS, IMAGE_COLS])
# 420x580
y = tf.placeholder(tf.float32, shape=[IMAGE_ROWS, IMAGE_COLS])

# 1x420x580x1
x_image = tf.reshape(x, [-1, IMAGE_ROWS, IMAGE_COLS, 1])

# network
layer1 = conv_layer(x_image, [5, 5, 1, 32])
layer2 = conv_layer(layer1, [5, 5, 32, 64])
layer2_flat = tf.reshape(layer2, [-1, 105 * 145 * 64])
fc_layer1 = conn_layer(layer2_flat, [105 * 145 * 64, 1024])

y_ = tf.nn.sigmoid(fc_layer1)

#fc_layer2 = conn_layer(fc_layer1, [1024, IMAGE_ROWS * IMAGE_COLS])

i = read_image(IMAGE_DIR + '6_33.tif')

sess.run(tf.initialize_all_variables())

tf.shape(fc_layer1).eval(feed_dict={x: i})
#with tf.Session() as sess:
#    image = tf.placeholder(tf.int32)
#    mask = tf.placeholder(tf.int32)
#    pred = tf.placeholder(tf.int32)
#    
#    intersection = tf.reduce_sum(tf.mul(mask, pred))
#    loss = tf.constant(1) - (tf.constant(2) * intersection) / (tf.reduce_sum(mask) + tf.reduce_sum(pred))