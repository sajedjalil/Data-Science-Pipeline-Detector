# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import os
os.chdir(os.getcwd())
'''
for x in ['sample_submission.csv','test.csv','train.csv']:
    print(x)
    print(os.system("head ../input/" + x))
    print(os.system("wc -l ../input/" + x))

#wc -l train.csv - 29.11M
#wc -l test.csv  -  8.6M
'''
#print(os.system("awk -F ',' '{print $6}' ../input/train.csv | sort | uniq | wc -l ")) - 108391
#print(os.system("awk -F ',' '{print $4}' ../input/train.csv | sort | uniq | wc -l ")) - 1026
print(os.system("sort -t',' -nk4 train.csv  | awk 'NR==1;END{print}'"))



#Lat - 0.0000, 10.0000
#Lon - 0.0000, 10.0000
#acc -       , 1033


