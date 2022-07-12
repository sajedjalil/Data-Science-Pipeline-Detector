# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/train.csv')

def plot_accuracy_for_place(df_all, placeid):

    df = df_all[df_all.place_id == placeid]
    plt.suptitle("placeid-%d" % placeid)
    
    plt.subplot(3,1,1)
    plt.ylabel('accuracy')
    plt.xlabel('x')
    plt.plot(df.x, df.accuracy, 'r.')

    plt.subplot(3,1,2)
    plt.ylabel('accuracy')
    plt.xlabel('y')
    plt.plot(df.y, df.accuracy, 'r.')

    plt.subplot(3,1,3)
    plt.ylabel('accuracy')
    plt.xlabel('time')
    plt.plot(df.time, df.accuracy, 'r.')
    
    plt.savefig("placeid-%d" % placeid)
    

    
places = np.unique(df.place_id)
for i in range(0,50):
    plot_accuracy_for_place(df, places[i])