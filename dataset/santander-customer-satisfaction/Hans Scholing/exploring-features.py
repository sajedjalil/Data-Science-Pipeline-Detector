# Based on https://www.kaggle.com/benhamner/d/uciml/iris/python-data-visualizations/notebook
# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd
import numpy as np

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Next, we'll load the train and test dataset, which is in the "../input/" directory
train = pd.read_csv("../input/train.csv") # the train dataset is now a Pandas DataFrame
test = pd.read_csv("../input/test.csv") # the train dataset is now a Pandas DataFrame

# Let's see what's in the trainings data - Jupyter notebooks print the result of the last thing you do
train.head()

# Press shift+enter to execute this cell
# 116 values in column var3 are -999999
# var3 is suspected to be the nationality of the customer
# -999999 would mean that the nationality of the customer is unknown
train.loc[train.var3==-999999].shape
# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
train = train.replace(-999999,2)
train.loc[train.var3==-999999].shape
# # Add feature that counts the number of zeroes in a row
X = train.iloc[:,:-1]
y = train.TARGET

X['n0'] = (X==0).sum(axis=1)
train['n0'] = X['n0']
# # Var38

# var38 is important according to XGBOOST
# see https://www.kaggle.com/cast42/santander-customer-satisfaction/xgboost-with-early-stopping/files
# Also RFC thinks var38 is important
# see https://www.kaggle.com/tks0123456789/santander-customer-satisfaction/data-exploration/notebook
# Var38 is suspected to be the mortage value with the bank. If the mortage is with another bank the national
# average is used. 
# See https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/19895/var38-is-mortgage-value
train.var38.describe()
# How is var38 looking when customer is unhappy ?
train.loc[train['TARGET']==1, 'var38'].describe()
# Histogram for var 38 is not normal distributed
train.var38.hist(bins=1000)
train.var38.map(np.log).hist(bins=1000)
# where is the spike between 11 and 12  in the log plot ?
train.var38.map(np.log).mode()
# What are the most common values for var38 ?
train.var38.value_counts()
# the value 117310.979016 appears 14868 times in colum var38
# the most common value is very close to the mean of the other values
train.var38[train['var38'] != 117310.979016494].mean()
# what if we exclude the most common value
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].value_counts()
# Look at the distribution
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].map(np.log).hist(bins=100)
# Above plot suggest we split up var38 into two variables
# var38mc == 1 when var38 has the most common value and 0 otherwise
# logvar38 is log transformed feature when var38mc is 0, zero otherwise
train['var38mc'] = np.isclose(train.var38, 117310.979016)
train['logvar38'] = train.loc[~train['var38mc'], 'var38'].map(np.log)
train.loc[train['var38mc'], 'logvar38'] = 0
#Check for nan's
print('Number of nan in var38mc', train['var38mc'].isnull().sum())
print('Number of nan in logvar38',train['logvar38'].isnull().sum())
# # var15
# The most important feature for XGBoost is var15. According to [a Kaggle form post](https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/19291/data-dictionary/110414#post110414)
#     var15 is the age of the customer. Let's explore var15
train['var15'].describe()
#Looks more normal, plot the histogram
train['var15'].hist(bins=100)
# Let's look at the density of the age of happy/unhappy customers
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "var15") \
   .add_legend()
plt.title('Unhappy customers are slightly older')
plt.show()
# # saldo_var30
# Let's look at the density of the age of happy/unhappy customers for saldo_var30
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "saldo_var30") \
   .add_legend()
plt.show()
# # Explore the interaction between var15 (age) and var38
sns.FacetGrid(train, hue="TARGET", size=10) \
   .map(plt.scatter, "var38", "var15") \
   .add_legend()
sns.FacetGrid(train, hue="TARGET", size=10) \
   .map(plt.scatter, "logvar38", "var15") \
   .add_legend()
plt.ylim([0,120]) # Age must be positive ;-)
# Exclude most common value for var38 
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10) \
   .map(plt.scatter, "logvar38", "var15") \
   .add_legend()
plt.ylim([0,120])
# What is distribution of the age when var38 has it's most common value ?
sns.FacetGrid(train[train.var38mc], hue="TARGET", size=6) \
   .map(sns.kdeplot, "var15") \
   .add_legend()
# What is density of n0 ?
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "n0") \
   .add_legend()
# # Select the most important features
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

# First select features based on chi2 and f_classif
p = 3

X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s]
print (features)
# Make a dataframe with the selected features and the target variable
X_sel = train[features+['TARGET']]
# # var36
# var36 (important for XGB and RFC is not selected but var36 is. Let's explore
X_sel['var36'].value_counts()
# var36 is most of the times 99 or [0,1,2,3]
# Let's plot the density in function of the target variabele
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "var36") \
   .add_legend()
# In above plot we see that the density of unhappy custormers is lower when var36 is not 99
# var36 in function of var38 (most common value excluded) 
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10) \
   .map(plt.scatter, "var36", "logvar38") \
   .add_legend()

# Let's seperate that in two plots
sns.FacetGrid(train[(~train.var38mc) & (train.var36 < 4)], hue="TARGET", size=10) \
   .map(plt.scatter, "var36", "logvar38") \
   .add_legend()
# Let's plot the density in function of the target variabele, when var36 = 99
sns.FacetGrid(train[(~train.var38mc) & (train.var36 ==99)], hue="TARGET", size=6) \
   .map(sns.kdeplot, "logvar38") \
   .add_legend()
sns.pairplot(train[['var15','var36','logvar38','TARGET']], hue="TARGET", size=2, diag_kind="kde")
train[['var15','var36','logvar38','TARGET']].boxplot(by="TARGET", figsize=(12, 6))
# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature
from pandas.tools.plotting import radviz
radviz(train[['var15','var36','logvar38','TARGET']], "TARGET")
# # now look at all 8 features together
features
radviz(train[features+['TARGET']], "TARGET")
sns.pairplot(train[features+['TARGET']], hue="TARGET", size=2, diag_kind="kde")
# # Correlations
cor_mat = X.corr()
f, ax = plt.subplots(figsize=(15, 12))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cor_mat,linewidths=.5, ax=ax)
