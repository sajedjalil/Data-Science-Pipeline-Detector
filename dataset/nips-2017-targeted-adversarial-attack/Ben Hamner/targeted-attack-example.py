# Code modified from https://github.com/tensorflow/cleverhans/blob/master/examples/nips17_adversarial_competition/sample_targeted_attacks/step_target_class/attack_step_target_class.py

import csv
import os
import numpy as np
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
    'input_dir', '../input/nips-2017-adversarial-learning-development-set/images', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS

def load_target_class(input_dir):
    with tf.gfile.Open(os.path.join(input_dir, '../images.csv')) as f:
        next(f) # skip header
        return {row[0]+".png": int(row[6]) for row in csv.reader(f) if len(row) >= 7}

def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, filename), 'wb') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')

def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    # Limit to first 20 images for this example
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png'))[:20]:
        with tf.gfile.Open(filepath, "rb") as f:
            images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float) / 255.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

eps = 2.0 * FLAGS.max_epsilon / 255.0
batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
num_classes = 1001

tf.logging.set_verbosity(tf.logging.INFO)

all_images_taget_class = load_target_class(FLAGS.input_dir)

with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(
            x_input, num_classes=num_classes, is_training=False)

    target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    one_hot_target_class = tf.one_hot(target_class_input, num_classes)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                     end_points['AuxLogits'],
                                                     label_smoothing=0.1,
                                                     weights=0.4)
    x_adv = x_input - eps * tf.sign(tf.gradients(cross_entropy, x_input)[0])
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        for filenames, images in load_images(FLAGS.input_dir, batch_shape):
            target_class_for_batch = (
                [all_images_taget_class[n] for n in filenames]
                 + [0] * (FLAGS.batch_size - len(filenames)))
            adv_images = sess.run(x_adv,
                                  feed_dict={
                                      x_input: images,
                                      target_class_input: target_class_for_batch
                                  })
            save_images(adv_images, filenames, FLAGS.output_dir)
