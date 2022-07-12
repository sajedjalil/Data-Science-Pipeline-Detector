import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import os
print(os.listdir("../input"))

#************************************
#*********Import Module**************
#************************************
import time
import tensorflow as tf
from numpy import*
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
#************************************
#*********Import CsvData*************
#************************************
print('Start Iputdata',time.asctime( time.localtime(time.time())))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test_y = pd.read_csv("../input/sample_submission.csv")
print('Start Extractdata',time.asctime( time.localtime(time.time())))
train_x = pd.concat((train.loc[:,'v2a1':'agesq'],))
train_y = (array(train['Target']) - 1)
test_y = (array(test_y['Target']) - 1)

train_y = train_y.transpose()
m1,n1 = shape(train)
Test_x = pd.concat((test.loc[:,'v2a1':'agesq'],))
#************************************
#*********Statr Preprocessing********
#************************************
print('Start PreProcessing',time.asctime( time.localtime(time.time())))
#train_x,test_x,train_y,test_y = train_test_split(dataMat,labelMat,test_size = 0.3,random_state = 0)##Users can choose train/test split to validation

numeric_feats = train_x.dtypes[train_x.dtypes != 'object'].index
train_x[numeric_feats] = train_x[numeric_feats].apply(lambda x:(x - x.mean()) / (x.std()))
Test_x[numeric_feats] = Test_x[numeric_feats].apply(lambda x:(x - x.mean()) / (x.std()))
#test_x[numeric_feats] = test_x[numeric_feats].apply(lambda x:(x - x.mean()) / (x.std()))

train_x = pd.get_dummies(train_x,dummy_na = True)
#test_x = pd.get_dummies(test_x,dummy_na = True)
Test_x = pd.get_dummies(Test_x,dummy_na = True)
train_x = train_x.fillna(train_x.mean())
#test_x = test_x.fillna(test_x.mean())
Test_x = Test_x.fillna(Test_x.mean())
train_x.drop('elimbasu5',axis = 1,inplace = True)
#test_x.drop('elimbasu5',axis = 1,inplace = True)
Test_x.drop('elimbasu5',axis = 1,inplace = True)
#************************************
#*********Statr PCA******************
#************************************
print('Start PCA',time.asctime( time.localtime(time.time())))
features_num = 30  #Users can choose numbers of features
pca = PCA(n_components = features_num)
trainx = pca.fit_transform(train_x)
#testx = pca.fit_transform(test_x)
Testx = pca.fit_transform(Test_x)
#************************************
#*********Train Data*****************
#************************************
print('Start DNN_train',time.asctime( time.localtime(time.time())))
def get_train_inputs():
    x = tf.constant(trainx)
    y = tf.constant(train_y)
    return x, y
feature_columns = [tf.contrib.layers.real_valued_column("", dimension = features_num)]
classifier = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,hidden_units=[10, 20, 20, 20, 10],n_classes = 4)
classifier.fit(input_fn = get_train_inputs,steps = 800)
def get_test_inputs():
    x = tf.constant(Testx)
    y = tf.constant(test_y)
    return x, y
accuracy_score = classifier.evaluate(input_fn = get_test_inputs,steps=1)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
print('End DNN_train',time.asctime( time.localtime(time.time())))
#************************************
#*********Test Prediction************
#************************************
def new_samples():
    return Testx
predictions = array(list(classifier.predict_classes(input_fn = new_samples)))
#************************************
#*********Outout to CSV**************
#************************************
print(check_output(["ls","../input"]).decode("utf8"))

print('Write_to_CSV',time.asctime( time.localtime(time.time())))
submission = pd.DataFrame({'Id':test['Id'],'Target':(predictions + 1)})
submission.to_csv('submission2.csv',index = False)
#print(submission)