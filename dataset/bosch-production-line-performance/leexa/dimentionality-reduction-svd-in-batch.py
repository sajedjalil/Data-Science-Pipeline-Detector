import pandas as pd
import math
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
#### Getting the column headers  from first 100 rows####
#### Reading as float32 to save Memory ####
data = pd.read_csv('../input/train_numeric.csv',nrows=100)
float_cols = [c for c in data]
float32_cols = {c: np.float32 for c in float_cols}

#### Pick number of Principal components you want, i like 5 ####
n_components =5


#### Read Panda dataframe by chunks and fit with PCA with n_components ####
chunksize,counter  = 50000,0
ipca = IncrementalPCA(n_components=n_components)
predictors = [x for x in data.keys() if (x != 'Response' and x != 'Id')]
for chunk in pd.read_csv('../input/train_numeric.csv', chunksize=chunksize,dtype=float32_cols):
    counter += chunksize
    print ('processed',counter,'samples')
    #### I decided to fill Nan with 999 ####
    chunk  = chunk.fillna(999)
    ipca.partial_fit(chunk[predictors])
print ('Number of Samples Seen:',ipca.n_samples_seen_ )
print ('Explained variance by %d PCs:' %n_components, np.sum(ipca.explained_variance_ratio_))

#### Make Train DataFrame only with n PC ####
PC_n = ['f'+str(x) for x in range(0,n_components)]
date_final = pd.DataFrame(columns=PC_n)
for cat in pd.read_csv('../input/train_numeric.csv', chunksize=chunksize,dtype=float32_cols):
        cat  = cat.fillna(999)
        y=ipca.transform(cat[predictors])
        temp = cat['Id'].to_frame()
        for i in PC_n:
            temp[i]=0
        temp[PC_n]=y
        date_final = date_final.merge(temp, how='outer')
print (date_final)
date_final['Id'] = date_final['Id'].astype(np.int32)

### Uncomment next line to save to CSV file ####
#date_final.to_csv('train_numeric_SVD35.csv',index=0)

#### Make Test DataFrame only with n PC ####
PC_n = ['f'+str(x) for x in range(0,n_components)]
date_final = pd.DataFrame(columns=PC_n)
for cat in pd.read_csv('../input/test_numeric.csv', chunksize=chunksize,dtype=float32_cols):
        cat  = cat.fillna(999)
        y=ipca.transform(cat[predictors])
        temp = cat['Id'].to_frame()
        for i in PC_n:
            temp[i]=0
        temp[PC_n]=y
        date_final = date_final.merge(temp, how='outer')
date_final['Id'] = date_final['Id'].astype(np.int32)

### Uncomment next line to save to CSV file ####
#date_final.to_csv('test_numeric_SVD35.csv',index=0)
