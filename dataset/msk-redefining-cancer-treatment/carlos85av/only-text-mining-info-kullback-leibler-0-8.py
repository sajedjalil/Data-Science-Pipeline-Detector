# coding: utf-8


import pandas as pd
import nltk, re, math, collections
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import lightgbm as lgb 
from sklearn import preprocessing




train_v = pd.read_csv('../input/training_variants')
test_v = pd.read_csv('../input/test_variants')
train_t = pd.read_csv('../input/training_text',sep='\|\|',skiprows=1,engine='python',names=["ID","Text"])
test_t = pd.read_csv('../input/test_text',sep='\|\|',skiprows=1,engine='python',names=["ID","Text"])

train = pd.merge(train_v, train_t, how='left', on='ID').fillna('')
y_labels = train['Class'].values

test = pd.merge(test_v, test_t, how='left', on='ID').fillna('')
test_id = test['ID'].values


#Clustering for classes

c1, c2, c3, c4, c5, c6, c7, c8, c9 = "", "", "", "", "", "", "", "", ""

for i in train[train["Class"]==1]["ID"]:
    c1+=train["Text"][i]+" "


for i in train[train["Class"]==2]["ID"]:
    c2+=train["Text"][i]+" "

for i in train[train["Class"]==3]["ID"]:
    c3+=train["Text"][i]+" "
    
for i in train[train["Class"]==4]["ID"]:
    c4+=train["Text"][i]+" "
    
for i in train[train["Class"]==5]["ID"]:
    c5+=train["Text"][i]+" "
    
    
for i in train[train["Class"]==6]["ID"]:
    c6+=train["Text"][i]+" "
    
for i in train[train["Class"]==7]["ID"]:
    c7+=train["Text"][i]+" "
    
for i in train[train["Class"]==8]["ID"]:
    c8+=train["Text"][i]+" "
    
    
for i in train[train["Class"]==9]["ID"]:
    c9+=train_t["Text"][i]+" "
 




def tokenize(_str):
    stops = set(stopwords.words("english"))
    tokens = collections.defaultdict(lambda: 0.)
    wnl = nltk.WordNetLemmatizer()
    for m in re.finditer(r"(\w+)", _str, re.UNICODE):
        m = m.group(1).lower()
        if len(m) < 2: continue
        if m in stops: continue
        m = wnl.lemmatize(m)
        tokens[m] += 1 
    return tokens



#Tokenizing texts and clusters

texts_for_training=[]
texts_for_test=[]
num_texts_train=len(train)

print("Tokenizing training texts")
for i in range(0,num_texts_train):
    if((i+1)%1000==0):
        print("Text %d of %d\n"%((i+1), num_texts_train))
    texts_for_training.append(tokenize(train["Text"][i]))
    
print("Tokenizing test texts")
num_texts_test=len(test)
for i in range(0,num_texts_test):
    if((i+1)%1000==0):
        print("Text %d of %d\n"%((i+1), num_texts_test))
    texts_for_test.append(tokenize(test["Text"][i]))

print("Generating cluster 1")
cluster1=tokenize(c1)

print("Generating cluster 2")
cluster2=tokenize(c2)

print("Generating cluster 3")
cluster3=tokenize(c3)

print("Generating cluster 4")
cluster4=tokenize(c4)

print("Generating cluster 5")
cluster5=tokenize(c5)

print("Generating cluster 6")
cluster6=tokenize(c6)

print("Generating cluster 7")
cluster7=tokenize(c7)

print("Generating cluster 8")
cluster8=tokenize(c8)

print("Generating cluster 9")
cluster9=tokenize(c9)







def kldiv(_s, _t):
    ssum = 0. + sum(_s.values())
 
    tsum = 0. + sum(_t.values())
   
    div = 0.
    for t, v in _s.items():
        pts = v / ssum
         
        if t in _t:
            ptt = (_t[t] + v) / (tsum + ssum)
        else:
            ptt = v / (tsum + ssum)
 
        ckl = pts  * math.log(pts / ptt)
       
        div +=  ckl
 
    return div




#Datasets for training


le = preprocessing.LabelEncoder()
target = le.fit_transform(y_labels)

XTrain=[]
num_texts_train=len(train)
for i in range(0,num_texts_train):
    XTrain.append([kldiv(texts_for_training[i],cluster1),kldiv(texts_for_training[i],cluster2),kldiv(texts_for_training[i],cluster3),kldiv(texts_for_training[i],cluster4),kldiv(texts_for_training[i],cluster5),kldiv(texts_for_training[i],cluster6),kldiv(texts_for_training[i],cluster7),kldiv(texts_for_training[i],cluster8),kldiv(texts_for_training[i],cluster9)])
    if((i+1)%250==0):
        print("Weights for text %d of %d\n"%((i+1), num_texts_train))
    
XTest=[]
num_texts_test=len(test)
for i in range(0,num_texts_test):
    XTest.append([kldiv(texts_for_test[i],cluster1),kldiv(texts_for_test[i],cluster2),kldiv(texts_for_test[i],cluster3),kldiv(texts_for_test[i],cluster4),kldiv(texts_for_test[i],cluster5),kldiv(texts_for_test[i],cluster6),kldiv(texts_for_test[i],cluster7),kldiv(texts_for_test[i],cluster8),kldiv(texts_for_test[i],cluster9)])
    if((i+1)%250==0):
        print("Weights for text %d of %d\n"%((i+1), num_texts_test))
    
    
train_data=lgb.Dataset(XTrain,label=target)


# Training

print("Training...")

param = {'num_leaves':130, 'num_trees':500,'objective':'multiclass','metric':'multi_logloss','learning_rate':.01,'max_bin':255,'num_class':9}

num_round=500
lgbm=lgb.train(param,train_data,num_round)


ypred2=lgbm.predict(XTest)
print(ypred2[0:5])  


submission = pd.DataFrame(ypred2, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = test_id
submission.to_csv('submission.csv', index=False)
