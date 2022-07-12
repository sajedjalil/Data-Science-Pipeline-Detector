# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import csv

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/new_disk/myfolder/data/statefarm/retrain',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 10,
                            """Display this many predictions.""")
tf.app.flags.DEFINE_string(
  'test_dir','/new_disk/myfolder/data/statefarm/test/',
  """ The path for the test images directory """)
tf.app.flags.DEFINE_string(
  'output_file','/new_disk/myfolder/data/statefarm/statefarm_out',
  """ The path for the output predictions csv file """)

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'output_graph.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(split_idx, num_splits):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  # Creates graph from saved GraphDef.
  create_graph()

  image_list = glob.glob(os.path.join(FLAGS.test_dir, '*.jpg'));
  total_files = len(image_list)
  split_idx = int(split_idx)
  num_splits = int(num_splits)
  start_idx = split_idx * int(total_files/num_splits)
  end_idx = (split_idx+1) * int(total_files/num_splits)
  if(split_idx == num_splits-1):
    end_idx = total_files
  print ("Start and End Idx", start_idx, end_idx)

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    readable_idx = ["c9", "c8", "c3", "c2", "c1", "c0", "c7", "c6", "c5", "c4"]

    final_result = []
    num_images = 0

    for image in image_list[start_idx:end_idx]:
      image_data = tf.gfile.FastGFile(image, 'rb').read()
      predictions = sess.run(softmax_tensor,
                             {'DecodeJpeg/contents:0': image_data})
      predictions = np.squeeze(predictions)

      # Creates node ID --> English string lookup.
      # node_lookup = NodeLookup()
      filename = image.split("/")[-1]
      modified_predictions = [filename, predictions[5], predictions[4], predictions[3], predictions[2], predictions[9], predictions[8], predictions[7], predictions[6], predictions[1], predictions[0]]
      # print (modified_predictions)
      final_result.append(modified_predictions)
      if(num_images%100 == 0):
        print("Num images", num_images)
      num_images += 1

      # top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]

      # print (image)
      # for node_id in top_k:
      #   # human_string = node_lookup.id_to_string(node_id)
      #   score = predictions[node_id]
      #   print('%s (score = %.5f)' % (readable_idx[node_id], score))
    with open(FLAGS.output_file+str(start_idx)+"_"+str(end_idx)+".csv", "wb") as f:
      f.write("img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n")
      writer = csv.writer(f)
      writer.writerows(final_result)
    # print (final_result)
    # np_final_result = np.array(final_result)
    # print (np_final_result)
    # np.savetxt(FLAGS.output_file, np_final_result, delimiter=',', header="img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9", comments="", fmt=['%s','%1.5f','%1.5f','%1.5f','%1.5f','%1.5f','%1.5f','%1.5f','%1.5f','%1.5f','%1.5f'])

def main(argv=None):
  print (argv)
  run_inference_on_image(argv[1], argv[2])


if __name__ == '__main__':
  tf.app.run()
