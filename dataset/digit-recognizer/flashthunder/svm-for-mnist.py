print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
train=pd.read_csv('../input/train.csv')
data_test=pd.read_csv('../input/test.csv')
#trainnum=20000
#testnum=20000

#data_train= train.iloc[:trainnum,1:].values
#label_train=train.iloc[:trainnum,0].values

#data_test= train.iloc[trainnum:trainnum+testnum,1:].values
#label_test= train.iloc[trainnum:trainnum+testnum,0].values

data_train=train.iloc[:,1:].values
label_train=train.iloc[:,0].values

data_test=data_test.values
	
data_train[data_train>0]=1
data_test[data_test>0]=1

print(data_train)
print(label_train)
# Create a classifier: a support vector classifier
classifier = svm.SVC(C=200,kernel='rbf',gamma=0.01,cache_size=8000,probability=False)
#classifier = svm.SVC()


# We learn the digits on the first half of the digits
classifier.fit(data_train, label_train)

# Now predict the value of the digit on the second half:
#expected = label_test
predicted = classifier.predict(data_test)

'''print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))'''

#print(expected)
df=pd.DataFrame(predicted)
df.index+=1
df.index.name='ImageId'
df.columns=['Label']
df.to_csv('results.csv',header=True)
print(predicted)
