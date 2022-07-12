# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier 
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
print(train)
train_copy=train
train_copy['OutcomeType'] = train['OutcomeType'].map( {'Adoption': 0, 'Died': 1,'Euthanasia':2,'Return_to_owner':3,'Transfer':4} ).astype(int)

'''outcomes=pd.get_dummies(train["OutcomeType"])
print (outcomes)

for row in outcomes:
    train_copy[row]=outcomes[row]

#train_copy.append()

subtype=pd.get_dummies(train["OutcomeSubtype"])

for row in subtype:
    train_copy[row]=subtype[row]'''

train_copy["AnimalType"]=pd.get_dummies(train["AnimalType"])

#assign id to outcome subtypes
subtypes_dict={}
subtypes=train['OutcomeSubtype'].unique()
counter=0
for item in subtypes:
    subtypes_dict[item]=counter
    counter+=1
    
train_copy['OutcomeSubtype'] = train['OutcomeSubtype'].map(subtypes_dict ).astype(int)



sex_dict={}
sex=train['SexuponOutcome'].unique()
counter=0
for item in sex:
    sex_dict[item]=counter
    counter+=1
    
train_copy['SexuponOutcome'] = train['SexuponOutcome'].map(sex_dict).astype(int)

breed_dict={}
breed=train['Breed'].unique()
counter=0
for item in breed:
    breed_dict[item]=counter
    counter+=1
    
train_copy['Breed'] = train['Breed'].map(breed_dict).astype(int)

color_dict={}
color=train['Color'].unique()
counter=0
for item in color:
    color_dict[item]=counter
    counter+=1
    
train_copy['Color'] = train['Color'].map(color_dict).astype(int)

print(color_dict)

#convert age to years

def age_to_years(item):
    # convert item to list if it is one string
    if type(item) is str:
        item = [item]
    ages = np.zeros(len(item))
    for i in range(len(item)):
        # check if item[i] is str
        if type(item[i]) is str:
            if 'day' in item[i]:
                ages[i] = int(item[i].split(' ')[0])/365
            if 'week' in item[i]:
                ages[i] = int(item[i].split(' ')[0])/52
            if 'month' in item[i]:
                ages[i] = int(item[i].split(' ')[0])/12
            if 'year' in item[i]:
                ages[i] = int(item[i].split(' ')[0])
        else:
            # item[i] is not a string but a nan
            ages[i] = 0
    return ages

train_copy['AgeuponOutcome']=age_to_years(train['AgeuponOutcome'])
#print(train.head())
print(train_copy.head())

plt.hist(train_copy['OutcomeType'])
plt.show()

forest=RandomForestClassifier(n_estimators=100)

#forest = forest.fit(train_copy[0::,5::],train_copy[0::,3])
