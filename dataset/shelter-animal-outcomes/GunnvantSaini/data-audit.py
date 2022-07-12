# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame,Series


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data=pd.read_csv("../input/train.csv",na_values=["","Missing"])
data.isnull().sum()
##Data Audit##
d=data.dtypes[data.dtypes!='object'].index.values
data[d]=data[d].astype('float64')
mean=DataFrame({'mean':data[d].mean()})
std_dev=DataFrame({'std_dev':data[d].std()})
missing= DataFrame({'missing':data[d].isnull().sum()})
obs=DataFrame({'obs':np.repeat(data[d].shape[0],len(d))},index=d)
missing_perc=DataFrame({'missing_perc':data[d].isnull().sum()/data[d].shape[0]})
minimum=DataFrame({'min':data[d].min()})
maximum=DataFrame({'max':data[d].max()})
unique=DataFrame({'unique':data[d].apply(lambda x:len(x.unique()),axis=0)})
q5=DataFrame({'q5':data[d].apply(lambda x:x.dropna().quantile(0.05))})
q10=DataFrame({'q10':data[d].apply(lambda x:x.dropna().quantile(0.10))})
q25=DataFrame({'q25':data[d].apply(lambda x:x.dropna().quantile(0.25))})
q50=DataFrame({'q50':data[d].apply(lambda x:x.dropna().quantile(0.50))})
q75=DataFrame({'q75':data[d].apply(lambda x:x.dropna().quantile(0.75))})
q85=DataFrame({'q85':data[d].apply(lambda x:x.dropna().quantile(0.85))})
q95=DataFrame({'q95':data[d].apply(lambda x:x.dropna().quantile(0.95))})
q99=DataFrame({'q99':data[d].apply(lambda x:x.dropna().quantile(0.99))})
DQ=pd.concat([mean,std_dev,obs,missing,missing_perc,minimum,maximum,unique,q5,q10,q25,q50,q75,q85,q95,q99],axis=1)

c=data.dtypes[data.dtypes=='object'].index.values
Mean=DataFrame({'mean':np.repeat('Not Applicable',len(c))},index=c)
Std_Dev=DataFrame({'std_dev':np.repeat('Not Applicable',len(c))},index=c)
Missing=DataFrame({'missing':data[c].isnull().sum()})
Obs=DataFrame({'obs':np.repeat(data[d].shape[0],len(c))},index=c)
Missing_perc=DataFrame({'missing_perc':data[c].isnull().sum()/data[c].shape[0]})
Minimum=DataFrame({'min':np.repeat('Not Applicable',len(c))},index=c)
Maximum=DataFrame({'max':np.repeat('Not Applicable',len(c))},index=c)
Unique=DataFrame({'unique':data[c].apply(lambda x:len(x.unique()),axis=0)})
Q5=DataFrame({'q5':np.repeat('Not Applicable',len(c))},index=c)
Q10=DataFrame({'q10':np.repeat('Not Applicable',len(c))},index=c)
Q25=DataFrame({'q25':np.repeat('Not Applicable',len(c))},index=c)
Q50=DataFrame({'q50':np.repeat('Not Applicable',len(c))},index=c)
Q75=DataFrame({'q75':np.repeat('Not Applicable',len(c))},index=c)
Q85=DataFrame({'q85':np.repeat('Not Applicable',len(c))},index=c)
Q95=DataFrame({'q95':np.repeat('Not Applicable',len(c))},index=c)
Q99=DataFrame({'q99':np.repeat('Not Applicable',len(c))},index=c)
dq=pd.concat([Mean,Std_Dev,Obs,Missing,Missing_perc,Minimum,Maximum,Unique,Q5,Q10,Q25,Q50,Q75,Q85,Q95,Q99],axis=1)

DQ=pd.concat([DQ,dq])
DQ.to_csv('dataquality.csv')


# Any results you write to the current directory are saved as output.