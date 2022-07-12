import pandas as pd
import numpy

#import plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

train=train.drop('Id',1)

#print("Description of training set")
#print(train.dtypes)

# Plotting data
cols=train.columns
size=len(cols)
x=cols[size-1]
y=cols[:-1]
#for i in range(0,size-1):
#   sns.violinplot(data=train, x=x, y=y[i])
#   plt.show()


# Data cleaning  
rem=[]
for i in cols:
    if train[i].std()==0:
        rem.append(i)

train_clean=train.drop(rem,1)
#print(train_clean.dtypes)


#Preparing the data
r,c=train.shape
values=train.values #forming just the matrix of values (no names, dataframes)
X=values[:,0:c-1]
Y=values[:,c-1]

from sklearn.preprocessing import StandardScaler

size=10 #size where categorical data (wilderness and soil type) begins
X_temp=StandardScaler().fit_transform(X[:,0:size])
print(X_temp.mean)
X_con = numpy.concatenate((X_temp,X[:,size:]),axis=1)