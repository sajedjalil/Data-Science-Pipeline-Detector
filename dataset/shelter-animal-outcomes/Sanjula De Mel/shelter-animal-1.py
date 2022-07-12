# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from ipykernel import kernelapp as app
from sklearn.ensemble import RandomForestClassifier




# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#take a look at dataset

train.drop("DateTime",axis = 1, inplace = True)
test.drop("DateTime",axis = 1, inplace = True)
train.drop("OutcomeSubtype", axis = 1, inplace = True)

def set_name(name):
    if type(name) is str:
        return 1
    else:
        return 0
   
def age_group(age):
    try:
        age_list = age.split()
    except:
        return None
    if(age_list[1].find("s")):
        age_list[1] = age_list[1].replace("s","")
    age_list[0]= int(age_list[0])
    if age_list[1] == "day":
        return "Infant"
    elif age_list[1] == "month":
        return "Young"
    elif (age_list[0] >= 1 and age_list[0]<=10) and (age_list[1] == "year"):
        return "Adult"
    else:
        return "Senior"
    
train["Has_Name"]= train["Name"].apply(set_name)
test["Has_Name"] = test["Name"].apply(set_name) 

train["AgeuponOutcome"] = train["AgeuponOutcome"].apply(age_group)
test["AgeuponOutcome"] = test["AgeuponOutcome"].apply(age_group)

train_id = train["AnimalID"]
test_id = test["ID"]
train.drop("AnimalID",axis = 1, inplace = True)
test.drop("ID",axis = 1, inplace = True)

train.drop("Breed", axis = 1, inplace = True)
test.drop("Breed", axis = 1, inplace = True)

train.drop("Color", axis = 1, inplace = True)
test.drop("Color", axis = 1, inplace = True)

train.drop("Name", axis = 1 , inplace = True)
test.drop("Name",axis = 1, inplace = True)

train["train"] = 1
test["train"] = 0

lables = train["OutcomeType"]
train.drop("OutcomeType",axis = 1, inplace = True)

all = pd.concat([train,test])
all_encoded = pd.get_dummies(all, columns = all.columns)

model_train = all_encoded[all_encoded["train_1"]==1]
model_test = all_encoded[all_encoded["train_0"]==1]
print(model_test.head)
model_train.drop(["train_0","train_1"], axis=1, inplace=True)
model_test.drop(["train_0","train_1"], axis=1, inplace=True)

X_train, X_val, y_train, y_val = train_test_split(model_train, lables, test_size=0.1)
forest = RandomForestClassifier(n_estimators = 200, n_jobs =2)
forest.fit(X_train,y_train)

y_probs = forest.predict_proba(X_val)
y_pred = forest.predict(X_val)

new_frame = pd.DataFrame(y_probs,columns =  ["Adoption","Died","Euthanasia","Return_to_owner","Transfer"])

from sklearn.metrics import classification_report, accuracy_score,log_loss

print(classification_report(y_val,y_pred))
print(accuracy_score(y_val,y_pred))
print(log_loss(y_val,y_probs))

forest.fit(model_train,lables)
y_probs = forest.predict_proba(model_test)

results = pd.read_csv("../input/sample_submission.csv")

results["Adoption"] = y_probs[:,0]
results["Died"] = y_probs[:,1]
results["Euthanasia"] = y_probs[:,2]
results["Return_to_owner"] = y_probs[:,3]
results["Transfer"] = y_probs[:,4]

results.to_csv("submission1.csv",index = False)




