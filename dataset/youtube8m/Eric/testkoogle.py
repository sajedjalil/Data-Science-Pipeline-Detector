# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#--------------------------------------------------------------------
#print(check_output(["ls", "../input/video_level"]).decode("utf8"))
#--------------------------------------------------------------------
#print(check_output(["ls", "../input/frame_level"]).decode("utf8"))
# Any results you write to the current directory are saved as output

#content = tf.train-0.tfrecord
detail_label_names = pd.read_csv('../input/label_names.csv') 
print(detail_label_names)
#print(len(detail_label_names)) # There are 4716 labels

