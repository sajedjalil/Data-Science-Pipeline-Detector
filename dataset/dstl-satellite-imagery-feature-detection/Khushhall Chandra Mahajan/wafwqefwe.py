# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import os
image_id = '6120_2_2'
os.system('ls')
print ("wdw")
os.system("ls ../input/sixteen_band/6120_2_2*")
print("done")
import glob
glob.glob('./*6120_2_2*')

filename = os.path.join('..', 'input', 'sixteen_band', '{}_M.tif'.format(image_id))
print (filename)