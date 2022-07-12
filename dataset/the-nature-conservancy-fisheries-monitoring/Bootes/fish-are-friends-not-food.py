# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from scipy import misc
#######################################
def get_file_names(index):
    path = "../input/train/"
    sub_folders=['ALB/','BET/','DOL/','LAG/','NoF/','OTHER/','SHARK/','YFT/']
    tmp_files = [path+sub_folders[index]+f for f in os.listdir(path+sub_folders[index]) if f.endswith('.jpg')]
    return tmp_files
#######################################    
def turn_img_2matrix(file_list):
    nQual=4
    print('One issue is that they are differently sized images!')
    for i in file_list:#Determine shape of the image file in an array
        tmp_img=misc.imread(i)
        tmp_img=tmp_img[::nQual,::nQual]#Cuts quality in half if n=2							
        tmp_img=tmp_img.reshape(-1,3)#Turns the image into long rows of 3 different colors
        break
    arr1=np.zeros((len(file_list),len(tmp_img)))
    arr2,arr3=np.copy(arr1),np.copy(arr1)#Same size and shape
    n=0
    for i in file_list:
        tmp_img=misc.imread(i)
        tmp_img=tmp_img[::nQual,::nQual]#Cuts quality in half if n=2
        max_val,min_val=(np.max(tmp_img),np.min(tmp_img))
        tmp_img=tmp_img/max_val
        tmp_img=tmp_img.reshape(-1,3)#Turns the image into long rows of 3 different colors
        arr1[n][:],arr2[n][:],arr3[n][:]=tmp_img[:,0],tmp_img[:,1],tmp_img[:,2]
        print(arr1[n][:])
        n+=1
        break
    return arr1,arr2,arr3#3 Colors
#######################################
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


print('Related guide: https://felixlaumon.github.io/2015/01/08/kaggle-right-whale.html')


#Which files are for which fish?
ALB_files = get_file_names(0)
BET_files =  get_file_names(1)
DOL_files =  get_file_names(2)
LAG_files =  get_file_names(3)
NoF_files =  get_file_names(4)
OTHER_files = get_file_names(5)
SHARK_files =  get_file_names(6)
YFT_files = get_file_names(7)

turn_img_2matrix(ALB_files)

path = "../input/test_stg1/"
test_files = [f for f in os.listdir(path) if f.endswith('.jpg')]

