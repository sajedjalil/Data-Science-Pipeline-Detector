# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


data=tf.placeholder(tf.float32, [None,1])

correctOutput=tf.placeholder(tf.float32, [None,1])

var=tf.Variable(.1)

output=data+var

error=tf.reduce_sum((output-correctOutput)**2)

train_step=tf.train.GradientDescentOptimizer(0.02).minimize(error)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
for i in range(10):

    _,v,o = sess.run([train_step,var,output], feed_dict={data: [[1],[2],[3],[4],[5],[6],[7],[8]], correctOutput: [[3],[4],[5],[6],[7],[8],[9],[10]]})


print(sess.run(output, feed_dict={data: [[15],[16],[17],[18],[19],[20]]}))
