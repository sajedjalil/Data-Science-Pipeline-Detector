# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import svm
import seaborn as sns
#%matplotlib inline 
from sklearn.neural_network import BernoulliRBM
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

T1_V4=train['T1_V4'].unique()
repl_T1_V4= np.arange(1,len(T1_V4)+1)
table_T1_V4 = pd.DataFrame({'T1_V4':T1_V4,'repl_T1_V4':repl_T1_V4})

T1_V5=train['T1_V5'].unique()
repl_T1_V5= np.arange(1,len(T1_V5)+1)
table_T1_V5 = pd.DataFrame({'T1_V5':T1_V5,'repl_T1_V5':repl_T1_V5})

T1_V6=train['T1_V6'].unique()
repl_T1_V6= np.arange(1,len(T1_V6)+1)
table_T1_V6 = pd.DataFrame({'T1_V6':T1_V6,'repl_T1_V6':repl_T1_V6})

T1_V7=train['T1_V7'].unique()
repl_T1_V7= np.arange(1,len(T1_V7)+1)
table_T1_V7 = pd.DataFrame({'T1_V7':T1_V7,'repl_T1_V7':repl_T1_V7})

T1_V8=train['T1_V8'].unique()
repl_T1_V8= np.arange(1,len(T1_V8)+1)
table_T1_V8 = pd.DataFrame({'T1_V8':T1_V8,'repl_T1_V8':repl_T1_V8})

T1_V9=train['T1_V9'].unique()
repl_T1_V9= np.arange(1,len(T1_V9)+1)
table_T1_V9 = pd.DataFrame({'T1_V9':T1_V9,'repl_T1_V9':repl_T1_V9})

T1_V11=train['T1_V11'].unique()
repl_T1_V11= np.arange(1,len(T1_V11)+1)
table_T1_V11 = pd.DataFrame({'T1_V11':T1_V11,'repl_T1_V11':repl_T1_V11})

T1_V12=train['T1_V12'].unique()
repl_T1_V12= np.arange(1,len(T1_V12)+1)
table_T1_V12 = pd.DataFrame({'T1_V12':T1_V12,'repl_T1_V12':repl_T1_V12})

T1_V15=train['T1_V15'].unique()
repl_T1_V15= np.arange(1,len(T1_V15)+1)
table_T1_V15 = pd.DataFrame({'T1_V15':T1_V15,'repl_T1_V15':repl_T1_V15})

T1_V16=train['T1_V16'].unique()
repl_T1_V16= np.arange(1,len(T1_V16)+1)
table_T1_V16 = pd.DataFrame({'T1_V16':T1_V16,'repl_T1_V16':repl_T1_V16})

T1_V17=train['T1_V17'].unique()
repl_T1_V17= np.arange(1,len(T1_V17)+1)
table_T1_V17 = pd.DataFrame({'T1_V17':T1_V17,'repl_T1_V17':repl_T1_V17})

T2_V3=train['T2_V3'].unique()
repl_T2_V3= np.arange(1,len(T2_V3)+1)
table_T2_V3 = pd.DataFrame({'T2_V3':T2_V3,'repl_T2_V3':repl_T2_V3})

T2_V5=train['T2_V5'].unique()
repl_T2_V5= np.arange(1,len(T2_V5)+1)
table_T2_V5 = pd.DataFrame({'T2_V5':T2_V5,'repl_T2_V5':repl_T2_V5})

T2_V11=train['T2_V11'].unique()
repl_T2_V11= np.arange(1,len(T2_V11)+1)
table_T2_V11 = pd.DataFrame({'T2_V11':T2_V11,'repl_T2_V11':repl_T2_V11})

T2_V12=train['T2_V12'].unique()
repl_T2_V12= np.arange(1,len(T2_V12)+1)
table_T2_V12 = pd.DataFrame({'T2_V12':T2_V12,'repl_T2_V12':repl_T2_V12})

T2_V13=train['T2_V13'].unique()
repl_T2_V13= np.arange(1,len(T2_V13)+1)
table_T2_V13=pd.DataFrame({'T2_V13':T2_V13,'repl_T2_V13':repl_T2_V13})




train_data=pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(train,table_T2_V13),table_T2_V12),table_T2_V11),table_T2_V5),table_T2_V3),
table_T1_V17),table_T1_V16),table_T1_V15),table_T1_V12),table_T1_V11),table_T1_V9),table_T1_V8),table_T1_V7),table_T1_V6),table_T1_V5),table_T1_V4)




train_data=pd.DataFrame(data={'Id':train_data['Id'], 'Hazard':train_data['Hazard'], 'T1_V1':train_data['T1_V1'], 
'T1_V2':train_data['T1_V2'],
 'T1_V3':train_data['T1_V3'], 'T1_V4':train_data['repl_T1_V4'], 'T1_V5':train_data['repl_T1_V5'], 
 'T1_V6':train_data['repl_T1_V6'],
       'T1_V7':train_data['repl_T1_V7'], 'T1_V8':train_data['repl_T1_V8'], 'T1_V9':train_data['repl_T1_V9'],
       'T1_V10':train_data['T1_V10'],
 'T1_V11':train_data['repl_T1_V11'], 'T1_V12':train_data['repl_T1_V12'], 'T1_V13':train_data['T1_V13'],
       'T1_V14':train_data['T1_V14'], 'T1_V15':train_data['repl_T1_V15'], 'T1_V16':train_data['repl_T1_V16'], 
       'T1_V17':train_data['repl_T1_V17'],
 'T2_V1':train_data['T2_V1'], 'T2_V2':train_data['T2_V2'], 'T2_V3':train_data['repl_T2_V3'],
       'T2_V4':train_data['T2_V4'], 'T2_V5':train_data['repl_T2_V5'], 'T2_V6':train_data['T2_V6'], 'T2_V7':train_data['T2_V7'],
 'T2_V8':train_data['T2_V8'], 'T2_V9':train_data['T2_V9'], 'T2_V10':train_data['T2_V10'],
       'T2_V11':train_data['repl_T2_V11'], 'T2_V12':train_data['repl_T2_V12'], 'T2_V13':train_data['repl_T2_V13'],
       'T2_V14':train_data['T2_V14'], 
'T2_V15':train_data['T2_V15']})


train_data1= train_data[[ 'T1_V1','T1_V10','T1_V2', 'T1_V11', 'T1_V12', 'T1_V13','T1_V3',
       'T1_V14', 'T1_V15', 'T1_V16', 'T1_V17', 'T1_V2', 'T1_V3', 'T1_V4',
       'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T2_V1', 'T2_V10',
       'T2_V11', 'T2_V12', 'T2_V13', 'T2_V14', 'T2_V15', 'T2_V2', 'T2_V3',
       'T2_V4', 'T2_V5', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9'
       ]]


target_data=train_data[['Hazard']]

#check=pd.DataFrame({'Hazard':selftest_Y['Hazard'],'prediction':prediction})


print("train: ",train_data1.shape,"target_data: ",target_data.shape)

train_X, test_X,train_Y, test_Y = train_test_split( train_data1, target_data, test_size=0.1, random_state=42)

print(train_X.shape, test_X.shape,train_Y.shape, test_Y.shape )

clf = DecisionTreeRegressor(max_depth=15)
clf.fit(train_X,train_Y)
pred=clf.predict(test_X)

print(np.unique(pred))

'''prediction=[]
for i in pred:
    z=i[0]
    prediction.append(z)
#print(prediction)'''

#print(clf.score(test_X,prediction))

comparision=pd.DataFrame({'Hazard':test_Y['Hazard'],'prediction':pred})
print(comparision.head(50))




test_data=pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(test,table_T2_V13),table_T2_V12),table_T2_V11),table_T2_V5),table_T2_V3),
table_T1_V17),table_T1_V16),table_T1_V15),table_T1_V12),table_T1_V11),table_T1_V9),table_T1_V8),table_T1_V7),table_T1_V6),table_T1_V5),table_T1_V4)




test_data=pd.DataFrame(data={'Id':test_data['Id'], 'T1_V1':test_data['T1_V1'], 'T1_V2':test_data['T1_V2'],
 'T1_V3':test_data['T1_V3'], 'T1_V4':test_data['repl_T1_V4'], 'T1_V5':test_data['repl_T1_V5'], 'T1_V6':test_data['repl_T1_V6'],
       'T1_V7':test_data['repl_T1_V7'], 'T1_V8':test_data['repl_T1_V8'], 'T1_V9':test_data['repl_T1_V9'], 'T1_V10':test_data['T1_V10'],
 'T1_V11':test_data['repl_T1_V11'], 'T1_V12':test_data['repl_T1_V12'], 'T1_V13':test_data['T1_V13'],
       'T1_V14':test_data['T1_V14'], 'T1_V15':test_data['repl_T1_V15'], 'T1_V16':test_data['repl_T1_V16'], 'T1_V17':test_data['repl_T1_V17'],
 'T2_V1':test_data['T2_V1'], 'T2_V2':test_data['T2_V2'], 'T2_V3':test_data['repl_T2_V3'],
       'T2_V4':test_data['T2_V4'], 'T2_V5':test_data['repl_T2_V5'], 'T2_V6':test_data['T2_V6'], 'T2_V7':test_data['T2_V7'],
 'T2_V8':test_data['T2_V8'], 'T2_V9':test_data['T2_V9'], 'T2_V10':test_data['T2_V10'],
       'T2_V11':test_data['repl_T2_V11'], 'T2_V12':test_data['repl_T2_V12'], 'T2_V13':test_data['repl_T2_V13'], 'T2_V14':test_data['T2_V14'], 
'T2_V15':test_data['T2_V15']})


test_data1= test_data[[ 'T1_V1','T1_V10','T1_V2', 'T1_V11', 'T1_V12', 'T1_V13','T1_V3',
       'T1_V14', 'T1_V15', 'T1_V16', 'T1_V17', 'T1_V2', 'T1_V3', 'T1_V4',
       'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T2_V1', 'T2_V10',
       'T2_V11', 'T2_V12', 'T2_V13', 'T2_V14', 'T2_V15', 'T2_V2', 'T2_V3',
       'T2_V4', 'T2_V5', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9']]


#print(test_data1.shape,test_data.shape,test.shape)



prediction=clf.predict(test_data1)

#print(len(prediction))

Submission=pd.DataFrame({'Id':test_data['Id'],'Hazard':prediction},columns=['Id','Hazard'],)

print(Submission.head())
Submission.to_csv('Submission.csv', index=False)


