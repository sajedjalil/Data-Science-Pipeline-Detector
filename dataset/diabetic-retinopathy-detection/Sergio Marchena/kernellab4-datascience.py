# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 




####  SERGIO MARCHENA, PABLO VIANA, JOSE MARTINEZ ####
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# YA CON LA DATA CARGADA COMENZAMOS A ANALIZAR Y A PREPARALA.

import tensorflow as tf
tf.enable_eager_execution()
tf.__version__

img_path = "/kaggle/input/diabetic-retinopathy-detection/1043_left.jpeg"
img_raw1 = tf.io.read_file(img_path)
print(repr(img_raw1)[:100])

img_tensor = tf.image.decode_image(img_raw1)

print(img_tensor.shape)
print(img_tensor.dtype)