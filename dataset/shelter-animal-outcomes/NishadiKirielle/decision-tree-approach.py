# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt


#Import 'tree' from scikit-learn library
from sklearn import tree
# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# Load the train and test datasets to create two DataFrames
train_url = "../input/train.csv"
train = pd.read_csv(train_url)

test_url = "../input/test.csv"
test = pd.read_csv(test_url)

#Data Preprocessing and Visualization
sns.set(style="darkgrid")

#Animal Type
#Categorical to Numerical conversion of Animal Type
train['AnimalType'][train['AnimalType']=='Dog']=0
train['AnimalType'][train['AnimalType']=='Cat']=1
#sns.countplot(x = train.AnimalType)

#Outcome Type
#sns.countplot(train.OutcomeType)

#Sex Upon Outcome

# functions to get new parameters from the column
def get_sex(x):
    x = str(x)
    if x.find('Male') >= 0: return 'male'
    if x.find('Female') >= 0: return 'female'
    return 'unknown'
def get_neutered(x):
    x = str(x)
    if x.find('Spayed') >= 0: return 'neutered'
    if x.find('Neutered') >= 0: return 'neutered'
    if x.find('Intact') >= 0: return 'intact'
    return 'unknown'

train['Sex'] = train.SexuponOutcome.apply(get_sex)
#Categorical to Numerical conversion of Sex
train['Sex'][train['Sex']=='male']=0
train['Sex'][train['Sex']=='female']=1
train['Sex'][train['Sex']=='unknown']=2

train['Neutered'] = train.SexuponOutcome.apply(get_neutered)
#Categorical to Numerical conversion of Sex
train['Neutered'][train['Neutered']=='neutered']=0
train['Neutered'][train['Neutered']=='intact']=1
train['Neutered'][train['Neutered']=='unknown']=2

#sns.countplot(train.Sex)
#sns.countplot(train.Neutered)

#Breed
def get_mix(x):
    x = str(x)
    if x.find('Mix') >= 0: return 'mix'
    return 'Not Mix'
train['Mix'] = train.Breed.apply(get_mix)

train['Mix'][train['Mix']=='mix']=0
train['Mix'][train['Mix']=='Not Mix']=1


#sns.countplot(train.Mix)

#Age
def calc_age_in_years(x):
    x = str(x)
    if x == 'nan': return 0
    age = int(x.split()[0])
    if x.find('year') > -1: return age 
    if x.find('month')> -1: return age / 12.
    if x.find('week')> -1: return age / 52.
    if x.find('day')> -1: return age / 365.
    else: return 0
    
train['AgeInYears'] = train.AgeuponOutcome.apply(calc_age_in_years)

def calc_age_category(x):
    if x < 3: return 'young'
    if x < 5: return 'young adult'
    if x < 10: return 'adult'
    return 'old'
train['AgeCategory'] = train.AgeInYears.apply(calc_age_category)

#Categorical to Numerical conversion of AGe
train['AgeCategory'][train['AgeCategory']=='young']=0
train['AgeCategory'][train['AgeCategory']=='young adult']=1
train['AgeCategory'][train['AgeCategory']=='adult']=2
train['AgeCategory'][train['AgeCategory']=='old']=3


 # Create the target and features numpy arrays: target, features_one
target = train["OutcomeType"].values
#features_one = train[["AnimalType",'Mix',"Neutered", "Sex","AgeInYears", "AgeCategory"]].values
features = train[["AnimalType", "Sex","AgeInYears"]].values

#  # Fit your first decision tree: my_tree_one
# my_tree_one = tree.DecisionTreeClassifier()
# my_tree_one = my_tree_one.fit(features_one, target)
forest = RandomForestClassifier(n_estimators = 100, random_state = 1)
my_forest = forest.fit(features, target)


# tree.export_graphviz(my_forest, out_file='tree.dot') 

 # Look at the importance and score of the included features
print("Feature Importance")
print(my_forest.feature_importances_)
#print("score")
print(my_forest.score(features, target))
#print(my_tree_one.score(features_one, target))

#Animal Type
#Categorical to Numerical conversion of Animal Type
test['AnimalType'][test['AnimalType']=='Dog']=0
test['AnimalType'][test['AnimalType']=='Cat']=1


test['Sex'] = test.SexuponOutcome.apply(get_sex)
#Categorical to Numerical conversion of Sex
test['Sex'][test['Sex']=='male']=0
test['Sex'][test['Sex']=='female']=1
test['Sex'][test['Sex']=='unknown']=2

test['AgeInYears'] = test.AgeuponOutcome.apply(calc_age_in_years)



# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["AnimalType", "Sex", "AgeInYears"]].values

# Make your prediction using the test set and print them.
my_prediction = my_forest.predict(test_features)
#print(my_prediction)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
ID =np.array(test["ID"]).astype(int)
my_solution = pd.DataFrame(my_prediction, ID, columns = ["OutcomeType"])
#print(my_solution)

# Check that your data frame has 418 entries
print("shape")
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv

my_solution["Adoption"] = 0
my_solution["Adoption"][my_solution["OutcomeType"]=="Adoption"]=1

my_solution["Died"]=0
my_solution["Died"][my_solution["OutcomeType"]=="Died"]=1

my_solution["Euthanasia"] = 0
my_solution["Euthanasia"][my_solution["OutcomeType"]=="Euthanasia"]=1

my_solution["Return_to_owner"]=0
my_solution["Return_to_owner"][my_solution["OutcomeType"]=="Return_to_owner"]=1

my_solution["Transfer"]=0
my_solution["Transfer"][my_solution["OutcomeType"]=="Transfer"]=1

my_solution = my_solution.drop('OutcomeType', 1)


my_solution.to_csv("my_solution_one.csv", index_label = ["ID"])

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#Any results you write to the current directory are saved as output.
#test.to_csv("my_solution_one.csv", index_label = ["AnimalType"])
