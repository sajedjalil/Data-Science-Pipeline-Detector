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
train.describe()
# happy customers have TARGET==0, unhappy custormers have TARGET==1
# A little less then 4% are unhappy => unbalanced dataset
df = pd.DataFrame(train.TARGET.value_counts())
df['Percentage'] = 100*df['TARGET']/train.shape[0]
df
# # var3: nationality of the customer

# Top-10 most common values
train.var3.value_counts()[:10]
# 116 values in column var3 are -999999
# var3 is suspected to be the nationality of the customer
# -999999 would mean that the nationality of the customer is unknown
train.loc[train.var3==-999999].shape
# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
train = train.replace(-999999,2)
train.loc[train.var3==-999999].shape
# # Add feature that counts the number of zeros in a row

X = train.iloc[:,:-1]
y = train.TARGET

X['n0'] = (X==0).sum(axis=1)
train['n0'] = X['n0']
train.n0.describe()
# # num_var4 : number of bank products

# According to dmi3kno (see https://www.kaggle.com/cast42/santander-customer-satisfaction/exploring-features/comments#115223)
# num_var4 is the number of products. Let's plot the distribution:
train.num_var4.hist(bins=100)
plt.xlabel('Number of bank products')
plt.ylabel('Number of customers in train')
plt.title('Most customers have 1 product with the bank')
plt.show()
# Let's look at the density of the of happy/unhappy customers in function of the number of bank products
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(plt.hist, "num_var4") \
   .add_legend()
plt.title('Unhappy cosutomers have less products')
plt.show()
train[train.TARGET==1].num_var4.hist(bins=6)
plt.title('Amount of unhappy customers in function of the number of products');
# # Var38
# var38 is important according to XGBOOST
# see https://www.kaggle.com/cast42/santander-customer-satisfaction/xgboost-with-early-stopping/files
# 
# Also RFC thinks var38 is important
# see https://www.kaggle.com/tks0123456789/santander-customer-satisfaction/data-exploration/notebook
# 
# Var38 is suspected to be the mortage value with the bank. If the mortage is with another bank the national
# average is used. 
# See https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/19895/var38-is-mortgage-value
# 
# [dmi3kno](https://www.kaggle.com/dmi3kno) says that var38 is value of the customer: [https://www.kaggle.com/cast42/santander-customer-satisfaction/exploring-features/comments#115223](https://www.kaggle.com/cast42/santander-customer-satisfaction/exploring-features/comments#115223)

train.var38.describe()
# How is var38 looking when customer is unhappy ?
train.loc[train['TARGET']==1, 'var38'].describe()
# Histogram for var 38 is not normal distributed
train.var38.hist(bins=1000);
train.var38.map(np.log).hist(bins=1000);
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
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].map(np.log).hist(bins=100);
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
train['var15'].hist(bins=100);
# Let's look at the density of the age of happy/unhappy customers
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "var15") \
   .add_legend()
plt.title('Unhappy customers are slightly older');
# # saldo_var30

train.saldo_var30.hist(bins=100)
plt.xlim(0, train.saldo_var30.max());
# improve the plot by making the x axis logarithmic
train['log_saldo_var30'] = train.saldo_var30.map(np.log)
# Let's look at the density of the age of happy/unhappy customers for saldo_var30
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "log_saldo_var30") \
   .add_legend();
# # Explore the interaction between var15 (age) and var38

sns.FacetGrid(train, hue="TARGET", size=10) \
   .map(plt.scatter, "var38", "var15") \
   .add_legend();
sns.FacetGrid(train, hue="TARGET", size=10) \
   .map(plt.scatter, "logvar38", "var15") \
   .add_legend()
plt.ylim([0,120]); # Age must be positive ;-)
# Exclude most common value for var38 
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10) \
   .map(plt.scatter, "logvar38", "var15") \
   .add_legend()
plt.ylim([0,120]);
# What is distribution of the age when var38 has it's most common value ?
sns.FacetGrid(train[train.var38mc], hue="TARGET", size=6) \
   .map(sns.kdeplot, "var15") \
   .add_legend();
# What is density of n0 ?
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "n0") \
   .add_legend()
plt.title('Unhappy customers have a lot of features that are zero');
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
X_sel['var36'].value_counts()
# var36 is most of the times 99 or [0,1,2,3]
# Let's plot the density in function of the target variabele
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "var36") \
   .add_legend()
plt.title('If var36 is 0,1,2 or 3 => less unhappy customers');
# In above plot we see that the density of unhappy custormers is lower when var36 is not 99
# var36 in function of var38 (most common value excluded) 
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10) \
   .map(plt.scatter, "var36", "logvar38") \
   .add_legend();
# Let's seperate that in two plots
sns.FacetGrid(train[(~train.var38mc) & (train.var36 < 4)], hue="TARGET", size=10) \
   .map(plt.scatter, "var36", "logvar38") \
   .add_legend()
plt.title('If var36==0, only happy customers');
# Let's plot the density in function of the target variabele, when var36 = 99
sns.FacetGrid(train[(~train.var38mc) & (train.var36 ==99)], hue="TARGET", size=6) \
   .map(sns.kdeplot, "logvar38") \
   .add_legend();
# # num_var5
train.num_var5.value_counts()
train[train.TARGET==1].num_var5.value_counts()
train[train.TARGET==0].num_var5.value_counts()
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(plt.hist, "num_var5") \
   .add_legend();
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "num_var5") \
   .add_legend();
sns.pairplot(train[['var15','var36','logvar38','TARGET']], hue="TARGET", size=2, diag_kind="kde");
from pandas.tools.plotting import scatter_matrix
scatter_matrix(train[['var15','var36','logvar38','TARGET']], alpha=0.2, figsize=(6, 6), diagonal='kde')
from pandas.tools.plotting import andrews_curves
plt.figure()
andrews_curves(train[['var15','var36','logvar38','TARGET']], 'TARGET')
from pandas.tools.plotting import parallel_coordinates
plt.figure()
parallel_coordinates(train[['var15','var36','logvar38','TARGET']], 'TARGET')
def parallel_coordinates(frame, class_column, cols=None, ax=None, color=None,
                     use_columns=False, xticks=None, colormap=None,
                     **kwds):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    n = len(frame)
    class_col = frame[class_column]
    class_min = np.amin(class_col)
    class_max = np.amax(class_col)

    if cols is None:
        df = frame.drop(class_column, axis=1)
    else:
        df = frame[cols]

    used_legends = set([])

    ncols = len(df.columns)

    # determine values to use for xticks
    if use_columns is True:
        if not np.all(np.isreal(list(df.columns))):
            raise ValueError('Columns must be numeric to be used as xticks')
        x = df.columns
    elif xticks is not None:
        if not np.all(np.isreal(xticks)):
            raise ValueError('xticks specified must be numeric')
        elif len(xticks) != ncols:
            raise ValueError('Length of xticks must match number of columns')
        x = xticks
    else:
        x = range(ncols)

    fig = plt.figure()
    ax = plt.gca()

    Colorm = plt.get_cmap(colormap)

    for i in range(n):
        y = df.iloc[i].values
        kls = class_col.iat[i]
        #ax.plot(x, y, color=Colorm((kls - class_min)/(class_max-class_min)), **kwds)
        ax.plot(x, y, color=Colorm((kls - class_min)/(class_max-class_min)), alpha=(kls - class_min)/(class_max-class_min), **kwds)

    for i in x:
        ax.axvline(i, linewidth=1, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(df.columns)
    ax.set_xlim(x[0], x[-1])
    ax.legend(loc='upper right')
    ax.grid()

    bounds = np.linspace(class_min,class_max,10)
    cax,_ = mpl.colorbar.make_axes(ax)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=Colorm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%.2f')

    return fig
parallel_coordinates(train[['var15','var36','logvar38','TARGET']], 'TARGET', colormap="binary")
train[['var15','var36','logvar38','TARGET']].boxplot(by="TARGET", figsize=(12, 6));
# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature
from pandas.tools.plotting import radviz
radviz(train[['var15','var36','logvar38','TARGET']], "TARGET");
# # now look at all 8 features together
features
radviz(train[features+['TARGET']], "TARGET");
sns.pairplot(train[features+['TARGET']], hue="TARGET", size=2, diag_kind="kde");
# # Correlations
cor_mat = X.corr()
f, ax = plt.subplots(figsize=(15, 12))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cor_mat,linewidths=.5, ax=ax);
cor_mat = X_sel.corr()
f, ax = plt.subplots(figsize=(15, 12))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cor_mat,linewidths=.5, ax=ax);