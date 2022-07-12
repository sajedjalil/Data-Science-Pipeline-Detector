# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# # Any results you write to the current directory are saved as output.

# print(check_output(["ls", "../input/train"]).decode("utf8"))

data_path='../input/train'

import glob
data_file_list=list(glob.glob(data_path+'/*'))

cats_path_list=[]
dogs_path_list=[]
for path in data_file_list:
    label=path.split('.')[0]
    if label == 'cat':
        cats_path_list.append(path)
    else:
        dogs_path_list.append(path)

print(len(cats_path_list))
print(len(dogs_path_list))
    


