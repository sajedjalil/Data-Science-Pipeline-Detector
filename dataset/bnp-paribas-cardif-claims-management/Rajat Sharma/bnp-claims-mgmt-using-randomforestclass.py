# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.cross_validation import KFold


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

start_time=time.time()

print('Loading data...')
trainData = pd.read_csv("../input/train.csv")
targetData = trainData['target'].values


trainData = trainData.drop(['ID','target','v3', 'v74','v38','v75','v23','v71','v72'],axis=1)
testData = pd.read_csv("../input/test.csv")

id_test = testData['ID'].values
testData = testData.drop(['ID','v3', 'v74','v38','v75','v23','v71','v72'],axis=1)

print('Clearing the data...')

#We cannot apply Machine learning models on empty or String sets.
#So, first we have to convert String values and empty entries into some numeric values.
for (trainData_name, trainData_series), (testData_name, testData_series) in zip(trainData.iteritems(),testData.iteritems()):
    if trainData_series.dtype == 'O':
        #for String values
        trainData[trainData_name], temporary_ind = pd.factorize(trainData[trainData_name])
        testData[testData_name] = temporary_ind.get_indexer(testData[testData_name])
        #but now we have -1 values (NaN)
    else:
        #for NaN values
        tmp_len = len(trainData[trainData_series.isnull()])
        if tmp_len>0:
          
            trainData.loc[trainData_series.isnull(), trainData_name] = -999 
       
        tmp_len = len(testData[testData_series.isnull()])
        if tmp_len>0:
            testData.loc[testData_series.isnull(), testData_name] = -999 

#Code for Learning Curve
'''cv = KFold(n_folds=10, n=len(targetData), shuffle=True)

train_sizes, train_scores, validation_scores = learning_curve(Ridge(alpha=1), trainData, targetData, cv=cv)


def plot_learning_curve(train_sizes, train_scores, validation_scores):
  
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.ylim(max(-3, validation_scores_mean.min() - .1), train_scores_mean.max() + .1)
    
plot_learning_curve(train_sizes, train_scores, validation_scores)'''




TrainingSet = trainData
TestSet = testData

'''#Logistic regression
logreg = LogisticRegression()
logreg.fit(TrainingSet, targetData)
print("LogisticRegression mean accuracy is:")
print(logreg.score(TrainingSet, targetData))'''


print ("Fitting the data...")
  
                 
model = RandomForestClassifier(n_estimators=1000,random_state=42,max_features= "auto",n_jobs=-1,min_samples_split= 5,
                            max_depth= 50, min_samples_leaf= 5)

model.fit(TrainingSet, targetData)

feature_importances = pd.Series(model.feature_importances_, index=TrainingSet.columns)
feature_importances.sort_values(inplace=True)
feature_importances.plot(kind="barh",figsize=(100,100));

print('Predicting and Saving in csv file...')
PredictedProbability = model.predict_proba(TestSet)
print("RandomForestClassifier mean accuracy is:")
print(model.score(TrainingSet, targetData))


pd.DataFrame({"ID": id_test, "PredictedProb": PredictedProbability[:,1]}).to_csv('Results.csv',index=False)
print ("File created! Process completed!")
print("---%.4s seconds taken to complete the process." %(time.time() - start_time))