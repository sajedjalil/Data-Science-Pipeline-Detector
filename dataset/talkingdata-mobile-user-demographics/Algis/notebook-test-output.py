# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import requests
from subprocess import check_output
print(check_output(["ls", "-la", "/"]).decode("utf8"))

exit(0)

def load(name):
    print(name)
    df = pd.read_csv("../input/"+name)
    print(df.info())
    print(df.head())
    return df
    
files = ["app_events.csv", "app_labels.csv", "events.csv"]

dfs = {}

for f in files:
    dfs[f] = load(f)

exit(0)

# Any results you write to the current directory are saved as output.

import tensorflow as tf

graph = tf.get_default_graph()

print(graph.get_operations())

input_value = tf.constant(1.0)

weight = tf.Variable(0.8)

output_value = weight * input_value

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

print(sess.run(output_value))