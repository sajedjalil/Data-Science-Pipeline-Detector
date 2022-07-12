import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import seaborn as sns
from IPython.display import YouTubeVideo
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["ls", "../input/video_level"]).decode("utf8"))

labels_df = pd.read_csv('../input/label_names.csv')
print("we have {} unique labels in the dataset".format(len(labels_df['label_name'].unique())))

labels_df = pd.read_csv('../input/label_names.csv')
labels = []
textual_labels = []
textual_labels_nested = []
filenames = ["../input/video_level/train-{}.tfrecord".format(i) for i in range(10)]
total_sample_counter = 0

label_counts = []

for filename in filenames:
    for example in tf.python_io.tf_record_iterator(filename):
        total_sample_counter += 1
        tf_example = tf.train.Example.FromString(example)

        label_example = list(tf_example.features.feature['labels'].int64_list.value)
        label_counts.append(len(label_example))
        labels = labels + label_example
        label_example_textual = list(labels_df[labels_df['label_id'].isin(label_example)]['label_name'])
        textual_labels_nested.append(set(label_example_textual))
        textual_labels = textual_labels + label_example_textual
        if len(label_example_textual) != len(label_example):
            print('label names lookup failed: {} vs {}'.format(label_example, label_example_textual))

print('label ids missing from label_names.csv: {}'.format(sorted(set(labels) - set(labels_df['label_id']))))
print('Found {} samples in all of the 10 available tfrecords'.format(total_sample_counter))

