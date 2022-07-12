__author__ = 'stevenydc'
import csv as csv
import numpy as np
import pandas as pd
import pylab as P
from sklearn.ensemble import RandomForestClassifier
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn import decomposition
import re
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation

StandardDataCleaning = False
df = pd.read_csv("train.csv",header=0)

# Creating Gender variable that encodes sex with 0 for female and 1 for male
df['Gender'] = df['Sex'].map( {'female':0, 'male': 1} ).astype(int)


# Calculating median age for every gender/class group... will be used to fill missing age data
median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

# Creating new variable AgeFill that fills null data points in the original Age variable
df['AgeFill'] = df.apply(lambda x: median_ages[x.Gender,x.Pclass-1] if x.Age != x.Age else x.Age, axis = 1)

# Alternative way of doing the same thing as above (one liner):
# def f(x):
# 	return median_ages[x.Gender,x.Pclass-1] if x.Age != x.Age else x.Age
# for i in range(0,2):
#     for j in range(0,3):
#         df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

# Filling the missing Embarked values with most popular port (the mode of the port variable)
df.loc[df['Embarked'].isnull(), 'Embarked'] = df['Embarked'].dropna().mode().values

# # Changing categorical variable to numerical... which is required for most ML algorithms
# Ports = list(enumerate(df['Embarked'].unique()))     #interesting way to create a list of enumerates
# Ports_dict = {name:i for i,name in Ports}           # creating a dict so that we can map letters to values
# df['Embarked'] = df['Embarked'].map(Ports_dict)

df = pd.concat([df, pd.get_dummies(df.Embarked).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)

df['Names'] = df['Name'].map(lambda x: len(re.split(' ', x)))
# All titles in data is preceded by a ',' and is followed by a '.'
# The .*? makes the '*' operation non greedy! 
df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
# Group low-occuring, related titles together
df['Title'][df.Title == 'Jonkheer'] = 'Master'
df['Title'][df.Title.isin(['Ms','Mlle'])] = 'Miss'
df['Title'][df.Title == 'Mme'] = 'Mrs'
df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
df['Title'][df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
# Changing Title variable to several binary variables and merge it back to df
df = pd.concat([df,pd.get_dummies(df['Title']).rename(columns=lambda x: 'Title_' + str(x))],axis=1)

# Replace missing values with "U0"
df['Cabin'][df.Cabin.isnull()] = 'U0'
# create feature for the alphabetical part of the cabin number
df['CabinLetter'] = df['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").findall(x)[0])
# convert the distinct cabin letters with incremental integer values
df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]

# # Divide all fares into quartiles
# df['Fare_bin'] = pd.qcut(df['Fare'], 4)
# # qcut() creates a new variable that identifies the quartile range, but we can't use the string so either
# # factorize or create dummies from the result
# df['Fare_bin_id'] = pd.factorize(df['Fare_bin'])[0]


# chopping off data that will not be used
df.drop(['Name','Sex','Ticket','Cabin','PassengerId', 'Age', 'Embarked', 'Title'],inplace=True, axis = 1)

'''
======= END of Training data cleaning ====
'''

'''
======= Start of Testing data cleaning ====
'''
# Creating Gender feature like before
test_df = pd.read_csv("test.csv", header = 0)
test_df['Gender'] = test_df['Sex'].map({'female':0, 'male':1})

# Creating AgeFill feature like before
test_df["AgeFill"] = test_df.apply(lambda x: median_ages[x.Gender, x.Pclass-1] if x.Age != x.Age else x.Age, axis = 1)

# this table contains the median_Fare for each class. Will be used to fill in empty Fare values for some data
median_Fare = np.zeros(3)
for i in range(3):
    median_Fare[i] = test_df.Fare[test_df.Pclass == i+1].median()
# The following code is selecting the rows in test_df that doesn't have a Fare value, and assign a value to it
# using its class as a criteria and using the median_Fare table that we computed before
# test_df.loc[test_df.Fare.isnull(), 'Fare'] = test_df.loc[test_df.Fare.isnull()].apply(lambda x: median_Fare[x.Pclass-1], axis = 1)
test_df.loc[test_df.Fare.isnull(),'Fare'] = test_df[test_df.Fare.isnull()].apply(lambda x: median_Fare[x.Pclass-1], axis = 1)

# Save the PassengerId for later use (generating file)... since it is not used as a parameter for our prediction model
test_ids = test_df.PassengerId

# # Fixing the Embarked feature like before
# test_df.Embarked = test_df.Embarked.map(Ports_dict)
# Transforming Embarked categorical variable to several binary variables
test_df = pd.concat([test_df, pd.get_dummies(test_df.Embarked).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)

test_df['Names'] = test_df['Name'].map(lambda x: len(re.split(' ', x)))
# All titles in data is preceded by a ',' and is followed by a '.'
# The .*? makes the '*' operation non greedy!
test_df['Title'] = test_df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
# Group low-occuring, related titles together
test_df['Title'][test_df.Title == 'Jonkheer'] = 'Master'
test_df['Title'][test_df.Title.isin(['Ms','Mlle'])] = 'Miss'
test_df['Title'][test_df.Title == 'Mme'] = 'Mrs'
test_df['Title'][test_df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
test_df['Title'][test_df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
# Changing Title variable to several binary variables and merge it back to test_df
test_df = pd.concat([test_df,pd.get_dummies(test_df['Title']).rename(columns=lambda x: 'Title_' + str(x))],axis=1)

# Replace missing values with "U0"
test_df['Cabin'][test_df.Cabin.isnull()] = 'U0'
# create feature for the alphabetical part of the cabin number
test_df['CabinLetter'] = test_df['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").findall(x)[0])
# convert the distinct cabin letters with incremental integer values
test_df['CabinLetter'] = pd.factorize(test_df['CabinLetter'])[0]
# # Divide all fares into quartiles
# test_df['Fare_bin'] = pd.qcut(test_df['Fare'], 4)
# # qcut() creates a new variable that identifies the quartile range, but we can't use the string so either
# # factorize or create dummies from the result
# test_df['Fare_bin_id'] = pd.factorize(test_df['Fare_bin'])[0]

# chopping off data that will not be used
test_df.drop(['Name','Sex','Ticket','Cabin','PassengerId', 'Age', 'Embarked', 'Title'],inplace=True, axis = 1)


'''
======= END of cleaning test data ====
'''


'''
======= Standardizing stuff =======
'''
comb_SibSp = pd.concat([df.SibSp, test_df.SibSp])
df.SibSp = (df.SibSp - comb_SibSp.mean())/comb_SibSp.std()
test_df.SibSp = (test_df.SibSp - comb_SibSp.mean())/comb_SibSp.std()
comb_Parch = pd.concat([df.Parch, test_df.Parch])
df.Parch = (df.Parch - comb_Parch.mean())/comb_Parch.std()
test_df.Parch = (test_df.Parch - comb_Parch.mean())/comb_Parch.std()

comb_AgeFill = pd.concat([df.AgeFill,test_df.AgeFill])
df.AgeFill = (df.AgeFill - comb_AgeFill.mean())/comb_AgeFill.std()
test_df.AgeFill = (test_df.AgeFill - comb_AgeFill.mean())/comb_AgeFill.std()

comb_Fare = pd.concat([df.Fare,test_df.Fare])
df.Fare = (df.Fare - comb_Fare.mean())/comb_Fare.std()
test_df.Fare = (test_df.Fare - comb_Fare.mean())/comb_Fare.std()

'''
======= Learning Curve =======
'''
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1,1.0,5)):
    P.figure()
    P.title(title)
    if ylim is not None:
        P.ylim(*ylim)
    P.xlabel("Training examples")
    P.ylabel("Score")
    train_sizes,train_scores,test_scores = learning_curve(estimator,X, y,
                                                                   cv=cv, n_jobs=n_jobs, train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    P.grid()

    P.fill_between(train_sizes, train_scores_mean-train_scores_std, train_scores_mean+train_scores_std,alpha=0.1,color='r')
    P.fill_between(train_sizes, test_scores_mean-test_scores_std, test_scores_mean+test_scores_std,alpha=0.1,color='r')

    P.plot(train_sizes,train_scores_mean,'o-',color='r', label='training score')
    P.plot(train_sizes,test_scores_mean,'o-',color='g', label='testing score')
    P.legend(loc='best')
    return P
    
    
    
# Utility function to report best scores
def report(grid_scores, n_top=1):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.5f} (std: {1:.5f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
# specify parameters and distributions to sample from
param_dist = {
              # "max_depth": [5, None],
              "max_features": [7,],
              "min_samples_split": [10,],
              # "min_samples_leaf": [1,2,5],
              # "bootstrap": [True, False],
              "criterion": ["entropy"]}


print ("Training...")
train_data = df.values
train_data = np.random.permutation(train_data[::,::])
temp = np.size(train_data,0)/5
cv_data = train_data[0:temp:,::]
train_data2 = train_data[temp::,::]


forest = RandomForestClassifier(n_estimators = 5000, oob_score=True)
# run randomized search
n_iter_search = 1
random_search = RandomizedSearchCV(forest, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=10)

start = time()
random_search.fit(train_data[::,1::], train_data[::,0])
# random_search.fit(pcaDataFrame.values, train_data[::,0])
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)

train_output = random_search.predict(train_data2[::,1::])
cv_output = random_search.predict(cv_data[::,1::])
print ("Training set accuracy: %.3f   CV set accuracy: %.3f"\
      %(len(train_data[train_output == train_data2[::,0]])/float(len(train_data2)),
      (len(cv_data[cv_output == cv_data[::,0]])/float(len(cv_data)))))

title = 'random forest'

cv = cross_validation.ShuffleSplit(train_data.shape[0], n_iter=10,
                                   test_size=0.2, random_state=0)
print ("cv stuff done")
plot_learning_curve(forest, title, train_data[::,1::], train_data[::,0],cv=cv, n_jobs=1)
print ("plotting done")
# Analyzing important features
forest = random_search.best_estimator_
feature_importance = forest.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())

feature_list = test_df.columns.values

df.drop(feature_list[feature_importance<5],inplace=True,axis=1)
test_df.drop(feature_list[feature_importance<5],inplace=True,axis=1)

print ("Predicting...")
test_data = test_df.values
output = random_search.predict(test_data).astype(int)
print ("Done...")







