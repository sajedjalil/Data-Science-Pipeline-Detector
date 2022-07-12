# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

x = 1

f2 = open("Paper_titles.txt", "w");

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if(x>0):
          ##print(os.path.join(dirname, filename))
          x = x+1
          
          if filename[-5:len(filename)] == ".json":
            f = open(os.path.join(dirname, filename), "r")
            f.readline()
            f2.write(f.readline()) ##paper id
            f.readline()
            f2.write(f.readline()) ## title
            f.close()
            
f2.close();

print("Complete!")
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session