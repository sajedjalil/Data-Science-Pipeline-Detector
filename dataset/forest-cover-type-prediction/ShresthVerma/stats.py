
import pandas as pd
# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
#test  = pd.read_csv("../input/test.csv")

# Write 
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import validation_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation 
import seaborn as sns
import matplotlib.pyplot as plt

#correlation
size=10
numcols=train.columns[:10]
numdata=train[numcols]
data_corr=numdata.corr()
threshold=.5
corr_list = []
cols=numdata.columns 
#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))
for v,i,j in s_corr_list:
    sns.pairplot(train, hue="Cover_Type", size=6, x_vars=cols[i],y_vars=cols[j] )
    plt.show()
    plt.savefig("output"+str(i)+".png")

cols=train.columns
x=cols[-1]
y=cols[0:-1]
for i in range(3,len(y)-5):
    sns.violinplot(data=train,x=x,y=y[i])
    plt.show()
    plt.savefig("output"+str(i)+".png")
    
# Any files you write to the current directory get shown as outputs