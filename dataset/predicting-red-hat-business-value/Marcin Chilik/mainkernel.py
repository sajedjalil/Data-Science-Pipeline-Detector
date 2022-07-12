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
#Red Hat Business Value data files:
#people.csv	.zip (3.22 mb)
#sample_submission.csv	.zip (1.18 mb)
#act_test.csv	.zip (4.03 mb)
#act_train.csv




class MainKernel():
    def __init__(self,inDataPath,dataNames):
        self.dataPath = inDataPath
        self.dataNames = dataNames
    def readData(self):
        self.Data = {}
        for iPth in self.dataPath:
            for iName in self.dataNames:
                if iName in iPth:
                    self.Data[iName] = pd.read_csv(iPth, parse_dates=['date'])

    def getDates(self):
        df_train = pd.merge(self.Data[self.dataNames[2]], self.Data[self.dataNames[0]], on='people_id')
        df_test = pd.merge(self.Data[self.dataNames[1]], self.Data[self.dataNames[0]], on='people_id')
        
        for d in ['date_x', 'date_y']:
            print('----------------------------------------------')
            print('Start of ' + d + ': ' + str(df_train[d].min().date()))
            print('  End of ' + d + ': ' + str(df_train[d].max().date()))
            print('Range of ' + d + ': ' + str(df_train[d].max() - df_train[d].min()) + '\n')
            print('----------------------------------------------')
        print("Train Size :")
        print(df_train.shape)
        print("Test Size :")
        print(df_test.shape)
        print(df_test.head(5))
        print(df_train.head(5))
        
        
 #The challenge is to create a classification algorithm that accurately 
 #identifies which customers have the most potential business 
 #value for Red Hat based on their characteristics and activities.
 
#Reading input paths
inputDataPath = ["../input/people.csv","../input/act_test.csv","../input/act_train.csv"]
dataNames = []
for path in inputDataPath:
    sp = path.split("/")
    sp = sp[-1].split(".")
    dataNames.append(sp[0])
i=0
for name in dataNames:
    print ("Data Name no %d :" % i  + name)
    i+=1

mainRun = MainKernel(inputDataPath,dataNames)

mainRun.readData()
mainRun.getDates()




