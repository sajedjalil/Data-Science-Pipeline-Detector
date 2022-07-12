# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv("../input/train_numeric.csv", nrows = 5000)#, usecols = ['Id', 'L0_S0_F0', 'L0_S0_F2', 'L0_S0_F4', 'L0_S0_F6', 'L0_S0_F8', 'L0_S0_F10', 'Response'])
#print(df[df['Response'] == 1], '\n')
#print("Failed mean ", df[df['Response'] == 1].mean(axis = 0), '\n')
print("Total rows", df.count().get_value(label = "Response"), '\n')
failedmean = df[df['Response'] == 1].mean(axis = 0)
failedstd = df[df['Response'] == 1].std(axis = 0)
print("Failed std ", failedstd, '\n')
print("Failed std non NAN", df[df['Response'] == 1].count().sort_values(), '\n')
s = df[df['Response'] == 1].count().sort_values()
s2 = s[s > 29]
print("Failed std non NAN gt 40", s2, '\n')
indexeswithoccurencegt40 = s2.head(s2.size-2)
#s3 = s2.keys()
#print(indexeswithoccurencegt40.axes)
df2 = pd.DataFrame(df, columns = indexeswithoccurencegt40.axes)
#print(df2)
#print(pd.Series(failedmean, index = indexeswithoccurencegt40.axes))
#print("Failed std withoccurencegt40", pd.Series(failedstd, index = indexeswithoccurencegt40.axes), '\n') #
failedmeanwithoccurencegt40 = pd.Series(failedmean, index = indexeswithoccurencegt40.axes)
failedstdwithoccurencegt40 = pd.Series(failedstd, index = indexeswithoccurencegt40.axes)
minthreshold1 = np.inf;
maxthreshold1 = -np.inf;
minthreshold2 = np.inf;
maxthreshold2 = -np.inf;
for index, row in df2.iterrows():
   rowminusmean = row - failedmeanwithoccurencegt40
   prob = np.exp(0.5*-np.power(rowminusmean/failedstdwithoccurencegt40, 2.))/(failedstdwithoccurencegt40*np.sqrt(2*math.pi))
#   prob = np.exp(-np.power(row - pd.Series(failedmean, index = indexeswithoccurencegt40.axes), 2.) / 
#   (2 * np.power(pd.Series(failedstd, index = indexeswithoccurencegt40.axes), 2.)))
   CumprodProb = prob.prod()   
   if df.iloc[index]['Response'] > 0.5:
      if CumprodProb < minthreshold1:
         minthreshold1 = CumprodProb
      if CumprodProb > maxthreshold1:
         maxthreshold1 = CumprodProb
   else:
      if CumprodProb < minthreshold2:
         minthreshold2 = CumprodProb
      if CumprodProb > maxthreshold2:
         maxthreshold2 = CumprodProb
print("minthreshold1 = ", minthreshold1)
print("maxthreshold1 = ", maxthreshold1)
print("minthreshold2 = ", minthreshold2)
print("maxthreshold2 = ", maxthreshold2)
plt.plot(prob.values)
#df3 = pd.DataFrame(df, columns = ['L3_S29_F3485'])
#print("df3 = ", df3)
#print("Failed std sorted", df[df['Response'] == 1].std(axis = 0).sort_values(), '\n')

#print("All mean ", df.mean(axis = 0), '\n')
#print("All std ", df.std(axis = 0), '\n')
#print("end")



# Any results you write to the current directory are saved as output.