# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
from pandas.plotting import scatter_matrix 
import seaborn as sns
sns.set()
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
seed=42


# %% [code]
## Import data
train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')


# %% [code]
##See data
train_targets_scored.head()

# %% [code]
train_targets_nonscored.head()


# %% [code]
train_features.head()

# %% [code]
test_features.head()

# %% [code]
train_features.describe()


# %% [code]
test_features.describe()

# %% [code]
train_targets_scored.describe()

# %% [code]
# No Missing values. Also not for the other tables
missing_vals = train_features.isnull().sum() / train_features.shape[0]
missing_vals[missing_vals > 0].sort_values(ascending=False)

# %% [code]
# No repeated compounds
train_features.sig_id.value_counts().max()


# %% [code]
# Checking how balanced are the predictors
plt.figure(figsize=(12, 12))
plt.subplot(311)
train_features['cp_type'].value_counts(normalize=True).plot(kind='bar', stacked=True, title="Treatment vs Control")
plt.subplot(312)
train_features['cp_dose'].value_counts(normalize=True).plot(kind='bar', stacked=True, title="How many of the two dosages")
plt.subplot(313)
train_features['cp_time'].value_counts(normalize=True).plot(kind='bar', stacked=True, title="How long were they exposed to the drugs")

# %% [code]
# Checking how balanced are the predictors

train_targets_scored.value_counts(normalize=True).plot(kind='bar', stacked=True, title="Treatment vs Control")


# %% [code]
# Check how variable distributions are for genes. 
train_features['g-0'].hist()
train_features['g-10'].hist()
train_features['g-20'].hist()
train_features['g-30'].hist()
train_features['g-40'].hist()

# %% [code]
# Check how variable distributions are for cell lines. 
train_features['c-0'].hist()
train_features['c-10'].hist()
train_features['c-20'].hist()
train_features['c-30'].hist()
train_features['c-40'].hist()

# %% [code]
# determine categorical and numerical features
train_features.dtypes
numerical_ix = train_features.select_dtypes(include=['int32','int64', 'float64']).columns
categorical_ix = train_features.select_dtypes(include=['object', 'bool']).columns

#train_features[categorical_ix].head()
train_features[categorical_ix].nunique()

##cp_type contains trt_cp and ctl_vehicle
#train_features['cp_type'].unique()

##cp_dose contains D1 and D2
#train_features['cp_dose'].unique()

common  = ['sig_id',
 'cp_type',
 'cp_time',
 'cp_dose']

features = [col for col in train_features.columns if col not in ['sig_id', 'cp_time', 'cp_type', 'cp_dose']]

genes = list(filter(lambda x : "g-" in x  , list(train_features)))

cells = list(filter(lambda x : "c-" in x  , list(train_features)))

# %% [code]
# Only Partner correlate slightly with Dependents and Tenure. PaperlessBilling with MonthlyCharges and Churn negatively with Tenure
#correlations = train_features[numerical_ix].corr()
corr_genes = train_features[genes].corr()
corr_genes


# %% [code]
##Chech if there are any correlation patterns
f = plt.figure(figsize=(19, 15))
plt.matshow(corr_genes, fignum=f.number)
cb = plt.colorbar()
plt.title('Correlation Matrix for MoA genes', fontsize=16);
plt.show()

# %% [code]
# Only Partner correlate slightly with Dependents and Tenure. PaperlessBilling with MonthlyCharges and Churn negatively with Tenure
#correlations = train_features[numerical_ix].corr()
corr_cells = train_features[cells].corr()
##Chech if there are any correlation patterns
f = plt.figure(figsize=(19, 15))
plt.matshow(corr_cells, fignum=f.number)
cb = plt.colorbar()
plt.title('Correlation Matrix for MoA cells', fontsize=16);
plt.show()


# %% [code]
#### PCA
pca = PCA(n_components=2)
#pca_gene.fit(train_features[genes])
#explained_variance = pca_gene.explained_variance_ratio_.sum()
#print("Number of components = %r and explained variance = %r"%(20,pca_gene.explained_variance_ratio_.sum()))
    
principalComponents_gene = pca.fit_transform(train_features[genes])
principalDf = pd.DataFrame(data = principalComponents_gene
             , columns = ['principal component 1', 'principal component 2'])
principalDf


# %% [code]
##By eye I could distinguish three groups of how drugs affect genes
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

ax.scatter(principalDf['principal component 1']
               , principalDf['principal component 2']
               #, c = color
               , s = 50)

ax.grid()

# %% [code]
pca.explained_variance_ratio_

# %% [code]

principalComponents_cell = pca.fit_transform(train_features[cells])
principalDf = pd.DataFrame(data = principalComponents_cell
             , columns = ['principal component 1', 'principal component 2'])
principalDf

##By eye I could distinguish 4/5 groups of how drugs affect cells
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

ax.scatter(principalDf['principal component 1']
               , principalDf['principal component 2']
               #, c = color
               , s = 50)

ax.grid()


#def plot_2d_space(X, y, label='Classes'):   
#    colors = ['#1F77B4', '#FF7F0E']
#    markers = ['o', 's']
#    plt.figure(figsize=(12, 12))
#    for l, c, m in zip(np.unique(y), colors, markers):
#        plt.scatter(
#            X[y==l, 0],
#            X[y==l, 1],
#            c=c, label=l, marker=m
#        )
#    plt.title(label)
#    plt.legend(loc='upper right')
#    plt.show()

##Both classes seem to be pretty similar distributions although there seem to be a few blue blobs isolated    
#plot_2d_space(X_pca, y, 'Imbalanced dataset (2 PCA components)')

