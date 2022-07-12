# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
import math
def mean_confidence_interval(data, confidence=0.95,theta=2):
	data=[i for i in data if not np.isnan(i)]
	if len(data)<1:
		return np.nan,np.nan
	a = 1.0*np.array(data)
	n = len(a)
	t=0
	while t!=n and n>1:
		t=len(a)
		m, se = np.mean(a), scipy.stats.sem(a)
		st=se*math.sqrt(n)
		a=a[np.where((a-m<=theta*st)&(a-m>=-theta*st))]
		n=len(a)
	return np.mean(a), scipy.stats.sem(a)	
dataset=pd.read_csv('../input/train.csv',header=0)
separator=len(dataset)
dataset=pd.concat((dataset,pd.read_csv('../input/test.csv',header=0))).reset_index(drop=True)

cols=[i for i in dataset.columns.tolist() if i[0]=='X']
ci_cols=[]
for col in cols:
    dataset['tmp_%s'%col]=dataset[[col,'y']].groupby(col).transform(lambda x:mean_confidence_interval(x.tolist()))['y']
    dataset['x_%s_ci_hi'%col]=dataset['tmp_%s'%col].apply(lambda x:x[0]+x[1])
    dataset['x_%s_ci_lo'%col]=dataset['tmp_%s'%col].apply(lambda x:x[0]-x[1])
    ci_cols+=['x_%s_ci_lo'%col,'x_%s_ci_hi'%col]
    dataset.drop('tmp_%s'%col,axis=1,inplace=True)
dataset['x_all_ci_lo']=dataset[ci_cols].min(axis=1)
dataset['x_all_ci_hi']=dataset[ci_cols].max(axis=1)
print(dataset[['ID','x_all_ci_lo','y','x_all_ci_hi']])