# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
import sklearn
np.random.seed(8)
def getint(s):
    if(s=="" or s==None):
        return str(0)
    return str(int(s.split()[-1]))
train_file = "../input/act_train.csv"
test_file  = "../input/act_test.csv"
res = open("train.csv",'w')
file = open(train_file,"r")
reader = csv.reader(file)
fieldnames = ["activity_category", "char_1", "char_2", "char_3", "char_4", "char_5", "char_6", "char_7", "char_8", "char_9", "char_10","outcome"]
writer = csv.DictWriter(res, fieldnames=fieldnames)
writer.writeheader()
check = 0
for row in reader:    
    if(check == 0):
        check = 1
        continue
    r = {}
    x = 3
    for i in fieldnames:
        r[i]=getint(row[x])
        x+=1
    writer.writerow(r)
file.close()
res.close()
act_id = []
res = open("test.csv",'w')
file = open(test_file,"r")
reader = csv.reader(file)
fieldnames = ["activity_category", "char_1", "char_2", "char_3", "char_4", "char_5", "char_6", "char_7", "char_8", "char_9", "char_10"]
writer = csv.DictWriter(res, fieldnames=fieldnames)
writer.writeheader()
check = 0
for row in reader:
    if(check == 0):
        check = 1
        continue
    r = {}
    act_id += [row[1]]
    x = 3
    for i in fieldnames:
        r[i]=getint(row[x])
        x+=1
    writer.writerow(r)
file.close()
res.close()
train= pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
features_forest = train[["activity_category", "char_1", "char_2", "char_3", "char_4", "char_5", "char_6", "char_7", "char_8", "char_9", "char_10"]].values
forest = RandomForestClassifier(max_depth = 16, min_samples_split=4, n_estimators = 80, random_state = 1)
target = train["outcome"].values
my_forest = forest.fit(features_forest, target)
test_features = test[["activity_category", "char_1", "char_2", "char_3", "char_4", "char_5", "char_6", "char_7", "char_8", "char_9", "char_10"]].values
pred = (my_forest.predict_proba(test_features))[:,1]


res = open("rf_bad_prob_v2.csv",'w')
fieldnames = ["activity_id","outcome"]
writer = csv.DictWriter(res, fieldnames=fieldnames)
writer.writeheader()
for j in range(len(pred)):
    x = [act_id[j],pred[j]]
    r = {}
    y = 0
    for i in fieldnames:
        r[i]=x[y]
        y+=1
    writer.writerow(r)
res.close()
# Any results you write to the current directory are saved as output.