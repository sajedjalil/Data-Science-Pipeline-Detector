# Code based on https://github.com/tensorflow/cleverhans/blob/master/examples/nips17_adversarial_competition/sample_attacks/fgsm/attack_fgsm.py

import os
from cleverhans.attacks import FastGradientMethod
import numpy as np
from PIL import Image
from scipy.misc import imread
from scipy.misc import imsave
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim
tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
tf.flags.DEFINE_string(
    'checkpoint_path', '../input/inception-v3/inception_v3.ckpt', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')
tf.flags.DEFINE_float(
    'max_epsilon', 4.0, 'Maximum size of adversarial perturbation.')
tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')
FLAGS = tf.flags.FLAGS

def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    # Limit to first 20 images for this example
    for filepath in sorted(tf.gfile.Glob(os.path.join(input_dir, '*.png')))[:20]:
        with tf.gfile.Open(filepath, "rb") as f:
            images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float)*2.0/255.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')

class InceptionModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs

eps = 2.0 * 16.0 / 255.0
batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
num_classes = 1001

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default():
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    model = InceptionModel(num_classes)

    fgsm = FastGradientMethod(model)
    x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        for filenames, images in load_images("../input/nips-2017-adversarial-learning-development-set/images/", batch_shape):
            adv_images = sess.run(x_adv, feed_dict={x_input: images})
            save_images(adv_images, filenames, "")