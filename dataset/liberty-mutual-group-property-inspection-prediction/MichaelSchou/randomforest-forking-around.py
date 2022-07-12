from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

s_data = []
s_labels = []
t_data = []
t_labels = []
d={}
z=0

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
print("train")
print(train.head())
print("testhead")
test.head()

train.drop('T1_V4', axis=1, inplace=True)
train.drop('T1_V5', axis=1, inplace=True)
train.drop('T1_V6', axis=1, inplace=True)
train.drop('T1_V11', axis=1, inplace=True)
train.drop('T2_V13', axis=1, inplace=True)
train.drop('T2_V14', axis=1, inplace=True)
train.drop('T2_V15', axis=1, inplace=True)
train.drop('T1_V8', axis=1, inplace=True)
train.drop('T1_V9', axis=1, inplace=True)
train.drop('T1_V12', axis=1, inplace=True)
train.drop('T1_V14', axis=1, inplace=True)
train.drop('T1_V15', axis=1, inplace=True)
train.drop('T1_V17', axis=1, inplace=True)
train.drop('T2_V3', axis=1, inplace=True)
train.drop('T2_V5', axis=1, inplace=True)
train.drop('T2_V6', axis=1, inplace=True)
train.drop('T2_V11', axis=1, inplace=True)
train.drop('T2_V12', axis=1, inplace=True)
train.drop('T2_V8', axis=1, inplace=True)
train.drop('T1_V7', axis=1, inplace=True)
train.drop('T2_V10', axis=1, inplace=True)
train.drop('T2_V7', axis=1, inplace=True)
train.drop('T1_V13', axis=1, inplace=True)
train.drop('T1_V10', axis=1, inplace=True)

test.drop('T1_V4', axis=1, inplace=True)
test.drop('T1_V5', axis=1, inplace=True)
test.drop('T1_V6', axis=1, inplace=True)
test.drop('T1_V11', axis=1, inplace=True)
test.drop('T2_V13', axis=1, inplace=True)
test.drop('T2_V14', axis=1, inplace=True)
test.drop('T2_V15', axis=1, inplace=True)
test.drop('T1_V8', axis=1, inplace=True)
test.drop('T1_V9', axis=1, inplace=True)
test.drop('T1_V12', axis=1, inplace=True)
test.drop('T1_V14', axis=1, inplace=True)
test.drop('T1_V15', axis=1, inplace=True)
test.drop('T1_V17', axis=1, inplace=True)
test.drop('T2_V3', axis=1, inplace=True)
test.drop('T2_V5', axis=1, inplace=True)
test.drop('T2_V6', axis=1, inplace=True)
test.drop('T2_V11', axis=1, inplace=True)
test.drop('T2_V12', axis=1, inplace=True)
test.drop('T2_V8', axis=1, inplace=True)
test.drop('T1_V7', axis=1, inplace=True)
test.drop('T2_V10', axis=1, inplace=True)
test.drop('T2_V7', axis=1, inplace=True)
test.drop('T1_V13', axis=1, inplace=True)
test.drop('T1_V10', axis=1, inplace=True)

col = train.columns
for i in range(len(train.Id)):
    s=[]
    for j in range(2,len(col)):
        w=col[j]+"_"+str(train[col[j]][i])
        if w in d:
            y = int(d[w])
        else:
            z+=1
            d[w]=int(z)
            y=int(z)
        s.append(int(y))
    s_data.append(list(s))
    s_labels.append("X"+str(train["Hazard"][i]))

col = test.columns
for i in range(len(test.Id)):
    s=[]
    for j in range(1,len(col)):
        w=col[j]+"_"+str(test[col[j]][i])
        if w in d:
            y = int(d[w])
        else:
            z+=1
            d[w]=int(z)
            y=int(z)
        s.append(int(y))
    t_data.append(list(s))
    
print(("\n").join(str(d).split(",")))

s_data=np.array(s_data)
t_data=np.array(t_data)

clf = Pipeline([('rfc', RandomForestClassifier(n_estimators = 10, n_jobs=-1))])
clf.fit(s_data, s_labels)
t_labels = clf.predict(t_data)
t_labels = [z[1:] for z in t_labels]
submission = pd.DataFrame({"Id": test.Id, "Hazard": t_labels})
submission.to_csv("random_forest.csv", index=False)

print(clf.named_steps['rfc'].feature_importances_)
i = 0
print(clf.named_steps['rfc'].estimators_)
#for t in clf.named_steps['rfc'].estimators_:
#    f = open(str(i) + '.dot', 'w')
#    tree.export_graphviz(t, out_file = f)
#    f.close
#    i += 1