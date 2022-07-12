# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import tree

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

animals = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')

print(animals)

otv = animals["OutcomeType"].value_counts()
print(otv)
otv = animals["OutcomeSubtype"].value_counts()
print(otv)


labels =  'Return to Owner', 'NULL', 'SCRP', 'Barn', 'Partner', 'Foster', 'Offsite',  'Suffering', 'Aggressive', 'Behavior', 'Court/Investigation', 'Rabies Risk', 'Medical', 'At Vet', 'In Surgery', 'In Kennel', 'In Foster', 'Enroute'
sizes = [4786, 8872, 1599, 2, 7816, 1800, 165, 1002, 320, 86, 6, 74, 66, 4, 3, 114, 52, 8]
colors = ['yellowgreen', 'grey', 'mediumblue','mediumpurple', 'lightskyblue', 'lightcoral', 'orange', 'red', 'yellow', 'purple', 'blue', 'green', 'lightyellow', 'orangered', 'white', 'black', 'lightgreen', 'brown'] 
explode = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)   # proportion with which to offset each wedge

plt.pie(sizes,              # data
        explode=explode,    # offset parameters 
        labels=labels,      # slice labels
        colors=colors,      # array of colours
        autopct='%1.1f%%',  # print the values inside the wedges
        shadow=True,        # enable shadow
        startangle=60       # starting angle
        )

plt.axis('equal')

plt.show()

def names(x):
    x = str(x)
    if 'nan' in x:
        return 0
    else:
        return 1

animals["Names_No"] = animals.Name.apply(names)
test["Names_No"] = test.Name.apply(names)


# SEEING HOW THE AGE AFFECTS THE OUTCOME
#making all of the values into days
def calc_age(x):
    x = str(x)
    if x == 'nan': return 0
    elif 'day ' in x:
        return 1
        
    elif 'days' in x:
        x = x.replace("days","")
        x = int(x)
        return x
        
    elif 'week ' in x:
        return 7
        
    elif 'weeks' in x:
        x = x.replace("weeks", "")
        x = 7*int(x)
        return x
    
    elif 'month ' in x:
        return 30
        
    elif 'months' in x:
        x = x.replace("months", "")
        x = 30*int(x)
        return x 
        
    elif 'year ' in x:
        return 365
        
    elif 'years' in x:
        x = x.replace("years","")
        x = 365*int(x)
        return x
        
    else:
        return 0
animals["Age_in_Days"] = 0
animals["Age_in_Days"]= animals.AgeuponOutcome.apply(calc_age)
animals["Puppy"] = 0

animals["Puppy"][animals["Age_in_Days"] >= 56] = 1 #young adults
animals["Puppy"][animals["Age_in_Days"] >= 548] = 2 #adults
animals["Puppy"][animals["Age_in_Days"] >= 2373] = 3 #seniors
animals["Puppy"][animals["Age_in_Days"] < 56 ] = 0 #puppies

test["Age_in_Days"] = 0
test["Age_in_Days"]= test.AgeuponOutcome.apply(calc_age)
test["Puppy"] = 0

test["Puppy"][test["Age_in_Days"] >= 56] = 1 #young adults
test["Puppy"][test["Age_in_Days"] >= 548] = 2 #adults
test["Puppy"][test["Age_in_Days"] >= 2373] = 3 #seniors
test["Puppy"][test["Age_in_Days"] < 56 ] = 0 #puppies


#all others are puppies

y = animals["Puppy"].value_counts()
print(y)

#animals["Dog"] = float('NaN')
#animals["Dog"][animals["AnimalType"] == "Cat"] = 0
#animals["Dog"][animals["AnimalType"] == "Dog"] = 1
#animals["Dog"] = animals["Dog"].fillna(animals["Dog"].median())
#print(animals["Dog"].value_counts())


def dogs_or_cats(x):
    x = str(x)
    if 'Dog' in x:
        return 1
    elif 'Cat' in x:
        return 2
    else:
        return 0

animals["Dogs"]= animals.AnimalType.apply(dogs_or_cats)
test["Dogs"]= test.AnimalType.apply(dogs_or_cats)

def sex(x):
    x = str(x)
    if 'Male' and 'Neuter' in x:
        return 1
    if 'Female' and 'Spay' in x:
        return 2
    if 'Male' in x:
        return 3
    if 'Female' in x:
        return 4
    else:
        return 0

animals["Sex"] = float('Nan')
animals["Sex"] = animals.SexuponOutcome.apply(sex)
test["Sex"] = test.SexuponOutcome.apply(sex)

def adopt(x):
    x = str(x)
    if 'Adoption' in x:
        return 1
    else: return 0
def ret(x):
    if 'Return_to_owner' in x:
        return 1
    else: return 0
def euth(x):
    if 'Euthanasia' in x:
        return 1
    else: return 0
def trans(x):   
    if 'Transfer' in x:
        return 1
    else: return 0
def died(x):
    if 'Died' in x:
        return 1
    else: return 0

animals["Outcome"] = float('NaN')
animals["Adoption"] = animals.OutcomeType.apply(adopt)
animals["Euthanasia"] = animals.OutcomeType.apply(euth)
animals["Transfer"] = animals.OutcomeType.apply(trans)
animals["Died"] = animals.OutcomeType.apply(died)
animals["Return_to_owner"] = animals.OutcomeType.apply(ret)

#print(animals["Euthanasia"])

#print(animals["Breed"].value_counts())

def breed(x):
    x = str(x)
    if 'Pit Bull' in x:
        return 0
    if 'Shih Tzu' in x:
        return 0
    if 'Lhasa Apso' in x:
        return 0
    if 'Pekingese' in x:
        return 0
    if 'Welmarener' in x:
        return 0
    if 'Bloodhound' in x:
        return 0
    if 'Russian Blue' in x:
        return 0
    if 'Ragdoll' in x:
        return 0
    if 'Cairn Terrier' in x:
        return 1
    if 'Catahoula' in x:
        return 1
    if 'Australian Kelpie' in x:
        return 1
    if 'Redboone Hound' in x:
        return 1
    if 'Border Terrier' in x:
        return 1
    if 'Mix' in x:
        return 3
    else: 
        return 2

animals["Group_Breeds"] = float('NaN')
animals["Group_Breeds"] = animals.Breed.apply(breed)
test["Group_Breeds"] = test.Breed.apply(breed)


# Create the target and features numpy arrays: target, features_one
target = animals["Adoption"].values
target2 = animals["Euthanasia"].values
target3 = animals["Return_to_owner"].values
target4 = animals["Transfer"].values
target5 = animals["Died"].values
features_one = animals[["Names_No", "Sex", "Puppy","Dogs"]].values

# Fit your first decision tree: my_tree_one

#my_tree_one = tree.DecisionTreeClassifier()
#my_tree_one = my_tree_one.fit(features_one, target, target2, target3, target4)

# Look at the importance and score of the included features
#print(my_tree_one.feature_importances_)
#print(my_tree_one.score(features_one, target, target2, target3, target4))

#test_features = test[["Names_No", "Sex", "Puppy","Dogs"]].values

# Make your prediction using the test set
#my_prediction = my_tree_one.predict(test_features)


# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
#def ID(x):
    #x = str(x)
    #if 'A' in x:
       # x = x.replace("A", "")
        # return x
#animals["ID"] = float('NaN')
#animals["ID"] = animals.AnimalID.apply(ID)
#test_features = test[["Names_No", "Sex", "Puppy","Dogs"]].values

# Make your prediction using the test set
#my_prediction = my_tree_one.predict(test_features)


# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
#def ID(x):
   # x = str(x)
    #if 'A' in x:
        #x = x.replace("A", "")
        #return x
#animals["ID"] = float('NaN')
#animals["ID"] = animals.AnimalID.apply(ID)


#ID = np.array(test["ID"]).astype(int)
#my_solution = pd.DataFrame(my_prediction, ID, columns = ["OutcomeType"])
#print(my_solution)

# Check that your data frame has 418 entries
#print(my_solution.shape)

#my_solution.to_csv("try_one.csv", index_label = ["OutcomeType"])

labels =  'Suffering', 'Aggressive', 'Behavior', 'Court/Investigation', 'Rabies Risk', 'Medical', 'NULL'
sizes = [1002, 320, 86, 6, 74, 66, 1]
colors = ['lightskyblue', 'lightcoral', 'orange','yellow', 'purple', 'blue', 'green']
explode = (0, 0, 0, 0, 0, 0, 0)   # proportion with which to offset each wedge

plt.pie(sizes,              # data
        explode=explode,    # offset parameters 
        labels=labels,      # slice labels
        colors=colors,      # array of colours
        autopct='%1.1f%%',  # print the values inside the wedges
        shadow=True,        # enable shadow
        startangle=60       # starting angle
        )

plt.axis('equal')

plt.show()
labels =  'At Vet', 'In Surgery', 'In Kennel', 'In Foster', 'Enroute', 'NULL'
sizes = [4, 3, 114, 52, 8, 16]
colors = ['lightcoral', 'orange', 'yellow', 'purple', 'blue', 'green']
explode = (0, 0, 0, 0, 0, 0)   # proportion with which to offset each wedge

plt.pie(sizes,              # data
        explode=explode,    # offset parameters 
        labels=labels,      # slice labels
        colors=colors,      # array of colours
        autopct='%1.1f%%',  # print the values inside the wedges
        shadow=True,        # enable shadow
        startangle=60       # starting angle
        )

plt.axis('equal')

plt.show()
labels =  'At Vet', 'In Surgery', 'In Kennel', 'In Foster', 'Enroute', 'NULL'
sizes = [4, 3, 114, 52, 8, 16]
colors = ['lightcoral', 'orange', 'yellowgreen', 'purple', 'turquoise', 'grey']
explode = (0, 0, 0, 0, 0, 0)   # proportion with which to offset each wedge

plt.pie(sizes,              # data
        explode=explode,    # offset parameters 
        labels=labels,      # slice labels
        colors=colors,      # array of colours
        autopct='%1.1f%%',  # print the values inside the wedges
        shadow=True,        # enable shadow
        startangle=60       # starting angle
        )

plt.axis('equal')

plt.show()

labels =  'SCRP', 'Barn','Partner', 'NULL'
sizes = [1599, 1, 7816, 6]
colors = ['orange', 'yellowgreen', 'turquoise', 'grey']
explode = (0, 0, 0, 0)   # proportion with which to offset each wedge

plt.pie(sizes,              # data
        explode=explode,    # offset parameters 
        labels=labels,      # slice labels
        colors=colors,      # array of colours
        autopct='%1.1f%%',  # print the values inside the wedges
        shadow=True,        # enable shadow
        startangle=60       # starting angle
        )

plt.axis('equal')

plt.show()

labels =  'Foster', 'Offsite', 'NULL', 'Barn'
sizes = [1800, 165, 8803, 1]
colors = ['mediumpurple', 'green', 'grey', 'turquoise']
explode = (0, 0, 0, 0)   # proportion with which to offset each wedge

plt.pie(sizes,              # data
        explode=explode,    # offset parameters 
        labels=labels,      # slice labels
        colors=colors,      # array of colours
        autopct='%1.1f%%',  # print the values inside the wedges
        shadow=True,        # enable shadow
        startangle=60       # starting angle
        )

plt.axis('equal')

plt.show()

print(animals["Breed"].value_counts())
