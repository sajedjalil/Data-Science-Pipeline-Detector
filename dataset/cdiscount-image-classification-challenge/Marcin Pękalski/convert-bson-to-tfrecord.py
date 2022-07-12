# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import io
import bson 
import tensorflow as tf
from skimage.data import imread   # or, whatever image library you prefer
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# helper functions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecords_filename = 'output_file.tfrecords'
opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)


# on my workstation it takes about 5 min per 100k entries, so should finish in about 6h
z = 0 
data = bson.decode_file_iter(open('../input/train.bson', 'rb'))
with tf.python_io.TFRecordWriter(tfrecords_filename, options=opts) as writer:
    for c, d in enumerate(data):       
        n_img = len(d['imgs'])
        for index in range(n_img):
            img_raw = d['imgs'][index]['picture']
            img = imread(io.BytesIO(img_raw))
            height = img.shape[0]
            width = img.shape[1]
            product_id = d['_id']
            category_id = d['category_id'] 
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'category_id': _int64_feature(category_id),
                'product_id': _int64_feature(product_id),
                'img_raw':_bytes_feature(img_raw)
            }))
            writer.write(example.SerializeToString())
        z = z + 1
        if z % 10000 == 0:
            print(z)
            break
        
