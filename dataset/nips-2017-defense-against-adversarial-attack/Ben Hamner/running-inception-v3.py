# Code modified from https://github.com/tensorflow/cleverhans/blob/master/examples/nips17_adversarial_competition/sample_defenses/base_inception_model/defense.py

import os
from cleverhans.attacks import FastGradientMethod
import numpy as np
import pandas as pd
from PIL import Image
from scipy.misc import imread
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
    'output_file', '', 'Output file to save labels.')

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
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png'))[:20]:
        with tf.gfile.Open(filepath, "rb") as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
num_classes = 1001
image_labels = pd.read_csv("../input/nips-2017-adversarial-learning-development-set/images.csv")
predictions = []

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    with slim.arg_scope(inception.inception_v3_arg_scope()):
        _, end_points = inception.inception_v3(
          x_input, num_classes=num_classes, is_training=False)

    predicted_labels = tf.argmax(end_points['Predictions'], 1)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        for filenames, images in load_images(FLAGS.input_dir, batch_shape):
            labels = sess.run(predicted_labels, feed_dict={x_input: images})
            for filename, label in zip(filenames, labels):
                true_label = image_labels.merge(pd.DataFrame({"ImageId":[filename[:-4]]}), on="ImageId")["TrueLabel"][0]
                predictions.append([filename[:-4], true_label, label])

pd.DataFrame(predictions, columns=["ImageId", "TrueLabel", "PredictedLabel"]).to_csv("predictions.csv")