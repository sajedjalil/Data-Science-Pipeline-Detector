# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


#Import 'tree' from scikit-learn library
from sklearn import tree

# Load the train and test datasets to create two DataFrames
train_url = "../input/train.csv"
train = pd.read_csv(train_url)

test_url = "../input/test.csv"
test = pd.read_csv(test_url)


def get_animal_type(x):
    x = str(x)
    if x.find('Dog') >= 0: return 0
    if x.find('Cat') >= 0: return 1
    return 2
    
def get_sex(x):
    x = str(x)
    if x.find('Male') >= 0: return 0
    if x.find('Female') >= 0: return 1
    return 2
    
def get_neutered(x):
    x = str(x)
    if x.find('Neutered Male') >= 0: return 0
    if x.find('Spayed Female') >= 0: return 1
    if x.find('Intact Male') >= 0: return 2
    if x.find('Intact Female') >= 0: return 3
    return 4

# def get_neutered(x):
#     x = str(x)
#     if x.find('Neutered Male') >= 0: return 0
#     if x.find('Spayed Female') >= 0: return 0
#     if x.find('Intact Male') >= 0: return 1
#     if x.find('Intact Female') >= 0: return 2
#     return 3
    
def get_hasName(x):
    x = str(x)
    if len(x) >= 0: return 1
    return 0
    
def get_mix(x):
    x = str(x)
    if x.find('Mix') >= 0: return 1
    return 0
    
def get_simple_color(x):
    x = str(x)
    if x.find('/') >= 0: return 1
    return 0
    
def calc_age_in_days(x):
    x = str(x)
    if x == 'nan': return 0
    age = int(x.split()[0])
    if x.find('year') > -1: return age * 365
    if x.find('month')> -1: return age * 30
    if x.find('week')> -1: return age *7
    if x.find('day')> -1: return age 
    else: return 0
    
def calc_age_in_years(x):
    x = str(x)
    if x == 'nan': return 0
    age = int(x.split()[0])
    if x.find('year') > -1: return age 
    if x.find('month')> -1: return age / 12.
    if x.find('week')> -1: return age / 52.
    if x.find('day')> -1: return age / 365.
    else: return 0
    
def get_hour(x):
    x = str(x)
    split_x = x.split()
    hour = int(x[1].split(':')[0])
    if hour > 2 & hour < 5 : return 0
    if hour > 4 & hour < 7 : return 1
    if hour > 6 & hour < 9 : return 2
    if hour > 8 & hour < 11 : return 3
    if hour > 10 & hour < 13 : return 4
    if hour > 12 & hour < 15 : return 5
    if hour > 14 & hour < 17 : return 6
    if hour > 16 & hour < 19 : return 7
    if hour > 18 & hour < 21 : return 8
    if hour > 20 & hour < 23 : return 9
    if hour > 22 & hour < 24 : return 10
    else: return 11
    
def get_month(x):
    x = str(x)
    split_x = x.split()
    month = int(x[0].split('/')[0])
    return month

train['AnimalType'] = train.AnimalType.apply(get_animal_type)
train['Sex'] = train.SexuponOutcome.apply(get_sex)
train['Neutered'] = train.SexuponOutcome.apply(get_neutered)
train['Name'] = train.Name.apply(get_hasName)
train['Mix'] = train.Breed.apply(get_mix)
train['AgeInYears'] = train.AgeuponOutcome.apply(calc_age_in_years)
train['AgeInDays'] = train.AgeuponOutcome.apply(calc_age_in_days)
train['Month'] = train.DateTime.apply(get_month)
train['Color'] = train.Color.apply(get_simple_color)


target = train["OutcomeType"].values
features_one = train[["AnimalType", "Sex","AgeInDays",'Neutered']].values

#  # Fit your first decision tree: my_tree_one
# my_tree_one = tree.DecisionTreeClassifier()
# my_tree_one = my_tree_one.fit(features_one, target)

# Building and fitting my_forest
forest = RandomForestClassifier(n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_one, target)

#tree.export_graphviz(my_tree_one, out_file='tree.dot') 

 # Look at the importance and score of the included features
# print("Feature Importance")
# print(my_tree_one.feature_importances_)
# print(my_tree_one.score(features_one, target))
print(my_forest.score(features_one, target))

test['AnimalType'] = test.AnimalType.apply(get_animal_type)
test['Sex'] = test.SexuponOutcome.apply(get_sex)
test['AgeInYears'] = test.AgeuponOutcome.apply(calc_age_in_years)
test['AgeInDays'] = test.AgeuponOutcome.apply(calc_age_in_days)
test['Neutered'] = test.SexuponOutcome.apply(get_neutered)
test['Name'] = test.Name.apply(get_hasName)
test['Mix'] = test.Breed.apply(get_mix)
test['Month'] = test.DateTime.apply(get_month)
test['Color'] = test.Color.apply(get_simple_color)

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["AnimalType", "Sex", "AgeInDays",'Neutered']].values

#pred_forest = my_forest.predict(___)
my_prediction = my_forest.predict(test_features)
ID =np.array(test["ID"]).astype(int)
my_solution = pd.DataFrame(my_prediction, ID, columns = ["OutcomeType"])


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

my_solution.to_csv("Random_Forest_2.csv", index_label = ["ID"])
