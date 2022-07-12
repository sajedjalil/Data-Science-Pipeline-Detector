import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot  as plt

from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn import ensemble
from sklearn import datasets
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

import sklearn as sk

#np.set_printoptions(precision=2)

train = pd.read_csv("../input/train.csv", 
        header  =   0,
        usecols =   ['id','build_count_before_1920','build_count_1921-1945','build_count_1946-1970','build_count_1971-1995','build_count_after_1995', 'full_sq','life_sq','floor','material', 'kremlin_km', 'ID_metro','office_count_2000','church_count_2000','mosque_count_2000', 'university_km',   'green_part_2000', 'max_floor','state','build_year','num_room','price_doc'],
        )
        #'timestamp',
train = train.select_dtypes(['number'])         
test  = pd.read_csv("../input/test.csv", 
        header  =   0,
        usecols =   ['id','build_count_before_1920','build_count_1921-1945','build_count_1946-1970','build_count_1971-1995','build_count_after_1995', 'full_sq','life_sq','floor', 'material', 'kremlin_km',  'ID_metro','office_count_2000','church_count_2000','mosque_count_2000', 'university_km',   'green_part_2000', 'max_floor','state','build_year','num_room'],
        )
        #'timestamp', 

#a bit better version, which use all numeric data
"""train = pd.read_csv("../input/train.csv", 
        header  =   0,
        )
train = train.select_dtypes(['number'])         
test  = pd.read_csv("../input/test.csv", 
        header  =   0,
        )
test = test.select_dtypes(['number'])  
"""


#timeseries
"""
time_t  = pd.read_csv("../input/train.csv", 
        header  =   0,
        usecols =   ['timestamp', 'full_sq', 'price_doc'],)

def month(row):
    stri = row['timestamp'][:-3]
    stri = stri.split('-')
    row['timestamp'] = int(stri[0]) * 12 + int(stri[1]) - 2010 * 12
    return row
	
time_t = time_t.apply(month, axis=1)
time_t = time_t.groupby('timestamp').sum()

def average(row):
	row['full_sq'] = row['price_doc']/row['full_sq']
	return row
global_mean  = time_t["full_sq"].mean()

time_t = time_t.apply(average, axis=1)
del time_t['price_doc']
"""
#polynomial code
"""
#plt.plot(time_t)
#plt.savefig('price1.png')
poly_x = time_t.index.values
#print(poly_x)
poly_y = time_t.values
poly_y = [item for sublist in poly_y for item in sublist]
poly_wsp = np.polyfit(poly_x, poly_y, 11)
poly = np.poly1d(poly_wsp)

#xp = np.linspace(0, 12*9, 100)
#plt.plot(poly_x, poly_y, '.', xp, poly(xp), '-')
#plt.savefig('price2.png')
"""

headers = list(train.columns.values)


#imputation
from sklearn.preprocessing import Imputer

test["build_count_before_1920"].fillna(1)
test["build_count_1921-1945"].fillna(1)
test["build_count_1946-1970"].fillna(1)
test["build_count_1971-1995"].fillna(1)
test["build_count_after_1995"].fillna(1)

train["build_count_before_1920"].fillna(1)
train["build_count_1921-1945"].fillna(1)
train["build_count_1946-1970"].fillna(1)
train["build_count_1971-1995"].fillna(1)
train["build_count_after_1995"].fillna(1)

def imputation(row):
    if math.isnan(row["life_sq"]):
        row["life_sq"] = row["full_sq"]*0.7 #average diffrence is 21,6 
    
    if math.isnan(row["max_floor"]):
        row["max_floor"] = row["floor"]  
        
    if row["state"] > 5 or math.isnan(row["state"]):
        row["state"] = 2 #2,1 is average
    
    years_distribution = [1900, 1935, 1955, 1980, 2005]
    if math.isnan(row["build_year"]) or row["build_year"] < 1700: #there are planty 0s in data,
        sum_a  = row['build_count_before_1920'] + row['build_count_1921-1945'] + row['build_count_1946-1970'] + row['build_count_1971-1995'] + row['build_count_after_1995']
        row["build_year"] = int(np.random.choice(years_distribution, 1, p=[row['build_count_before_1920']/sum_a, row['build_count_1921-1945']/sum_a,row['build_count_1946-1970']/sum_a,row['build_count_1971-1995']/sum_a,row['build_count_after_1995']/sum_a]))
    
    if math.isnan(row["num_room"]) or row["num_room"] == 0:
        row["num_room"] = round(row["life_sq"]/15,0)
   
    #stri = row['timestamp'][:-3]
    #stri = stri.split('-')
    #row['timestamp'] = int(stri[0]) * 12 + int(stri[1]) - 2010 * 12
    return row
    
    
test = test.apply(imputation, axis=1)
train = train.apply(imputation, axis=1)


tra = train.values
tes_X = test.values
tra_Y = tra[:,-1]
tra_X = tra[:,:-1]

#templorary split tra into two sets
#import random
#random.shuffle(tra)
#tra_Y = tra[:,-1]
#tra_X = tra[:,:-1]
#tes_X, tra_X = tra_X[20000:, :], tra_X[:20000, :] #about 20k training, 10k test
#tes_Y, tra_Y = tra_Y[20000:], tra_Y[:20000]
    


#last chance imputation
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(tes_X)
tes_X = imp.transform(tes_X)

imp2 = Imputer(missing_values='NaN', strategy='median', axis=0)
imp2.fit(tra_X)
tra_X = imp2.transform(tra_X)


      
# create a 0 test_Y_pred
#tes_Y_pred = [0 for i in range(len(tes_Y))]
 
#First try - Lasso model
"""
reg = linear_model.Lasso(alpha = 1,  max_iter=1000)
reg.fit(tra_X[:,1:], tra_Y)
tes_Y_pred = reg.predict(tes_X[:,1:])
"""

#Second try - gradient boosting model
from sklearn import ensemble
params = {'n_estimators': 1000, 'max_depth': 10, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(tra_X[:,1:], tra_Y)
tes_Y_pred = clf.predict(tes_X[:,1:])


#RMSLE Metric
#rmsle = 0.0
#for i in range(len(tes_Y)):
#    rmsle += math.pow(float(math.log(max(0.00001, float(tes_Y_pred[i]*poly(tes_X[i][1])) + 1.0))) - float(max(0.00001, math.log(float(tes_Y[i]) + 1.0))),2)
#rmsle = math.sqrt(rmsle/len(tes_Y))
#print("RMSLE: %.4f" % rmsle) #lower is better


#save valuation to the file, 
output = open('result.csv', "w")
output.write("id,price_doc\n")

for i in range(len(tes_Y_pred)):
    output.write(str(int(tes_X[i][0])) + ',' +  str(tes_Y_pred[i]) +'\n')
    #output.write(str(int(tes_X[i][0])) + ',' +  str(tes_Y[i]) + ',' +  str(tes_Y_pred[i]*poly(tes_X[i][1])) +'\n')

output.close()