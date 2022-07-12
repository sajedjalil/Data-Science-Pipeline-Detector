import pandas as pd 
import numpy as np
from lasagne import layers  
from lasagne.updates import nesterov_momentum  
from nolearn.lasagne import NeuralNet  
from nolearn.lasagne import visualize  
import theano
import theano.tensor as T
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

def missing(x):
    return sum(x.isnull())

train.apply(missing, axis=0).to_csv('title.csv',index=False)

s=['Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32', 'Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41', 'Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']
for i in s:
    print(i,':',train[i].describe())

train.fillna(0,inplace=True)
test.fillna(0,inplace=True)

"""
The following variables are all categorical (nominal):

Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41

The following variables are continuous:

Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5

The following variables are discrete:

Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32

Medical_Keyword_1-48 are dummy variables.
"""

col=train.columns

def unique_data(label,data,length):
    unique_datas={}
    unique_count={}
    for i in label:
        j=len(data[i].unique())
#        print(i,j)
        if j<length:
            unique_datas[i]=data[i].unique()
#            print(data[i].value_counts())
            unique_count[i]=data[i].value_counts()
    return unique_datas    

continues=['Product_Info_4','Ins_Age','Ht','Wt','BMI','Employment_Info_1','Employment_Info_4','Employment_Info_6','Insurance_History_5','Family_Hist_2','Family_Hist_3','Family_Hist_4','Family_Hist_5']

#print(train[continues])

category=['Product_Info_1','Product_Info_2','Product_Info_3','Product_Info_5','Product_Info_6','Product_Info_7','Employment_Info_2','Employment_Info_3','Employment_Info_5','InsuredInfo_1','InsuredInfo_2','InsuredInfo_3','InsuredInfo_4','InsuredInfo_5','InsuredInfo_6','InsuredInfo_7','Insurance_History_1','Insurance_History_2','Insurance_History_3','Insurance_History_4','Insurance_History_7','Insurance_History_8','Insurance_History_9','Family_Hist_1','Medical_History_2','Medical_History_3','Medical_History_4','Medical_History_5','Medical_History_6','Medical_History_7','Medical_History_8','Medical_History_9','Medical_History_11','Medical_History_12','Medical_History_13','Medical_History_14','Medical_History_16','Medical_History_17','Medical_History_18','Medical_History_19','Medical_History_20','Medical_History_21','Medical_History_22','Medical_History_23','Medical_History_25','Medical_History_26','Medical_History_27','Medical_History_28','Medical_History_29','Medical_History_30','Medical_History_31','Medical_History_33','Medical_History_34','Medical_History_35','Medical_History_36','Medical_History_37','Medical_History_38','Medical_History_39','Medical_History_40','Medical_History_41']
cate_dict={}
for i in category:
    j=len(train[i].unique())
    if j<10:
        cate_dict[i]=train[i].unique()
#    print(i,train[i].value_counts())
#print(cate_dict)
    
descrete=['Medical_History_1','Medical_History_10','Medical_History_15','Medical_History_24','Medical_History_32']
des_dict={}
for i in descrete:
    des_dict[i]=train[i].unique()
#    print(i,train[i].value_counts())

a=unique_data(category,train,10)
#print(a)

k=train['Product_Info_2']
change=k.unique()
ran=range(len(change))
train['Product_Info_2'].replace(to_replace=change,value=ran,inplace=True)

#print(train['Product_Info_2'].unique(),train['Product_Info_2'].value_counts())

m=[]
'''
for i in descrete:
    m.append(i)

for i in category:
    m.append(i)
    
for i in continues:
    m.append(i)
'''

m.extend(descrete)
m.extend(category)
m.extend(continues)
print(m)


from sklearn.ensemble import RandomForestClassifier

#gb=RandomForestClassifier(n_estimators=500,max_depth=None, min_samples_split=1, random_state=0)
gb=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=3, random_state=0)
x=train.drop('Response',axis=1)
y=train['Response']
#train_use=x[m].drop('Product_Info_2',axis=1)
#test_use=test[m].drop('Product_Info_2',axis=1)
#print(x[m],y)
#print(train_use,test_use)
#first 34655
ff=[ 0,  2,  3,  6, 11, 27, 29, 31, 54, 62, 65, 66, 67, 68, 69, 70, 73,74, 75, 76]
fe=[]
for i in ff:
    fe.append(m[i])
#    print(m[i])
m=fe
#print(m)

train_use=x[m]
test_use=test[m]

q=gb.fit(train_use,y)
z=q.predict(train_use)
train['predict']=z
u=sum(train['Response']==train['predict'])

feature=q.feature_importances_
print(np.where(feature>0.01))
#print(len(col),len(q.feature_importances_),len(m))
print(u)
print(train)
train['right']=(train['Response']==train['predict'])
train.to_csv('first.csv',index=False)
#sample_submission['Response']=z
#sample_submission.to_csv('first.csv',index=False)

