__author__ = 'tushar'
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import TheilSenRegressor
from sklearn.preprocessing import Imputer
dir = os.path.dirname(__file__)
####Linux Variables###############
train='../input/train.csv'
test='../input/test.csv'
store='../input/store.csv'
submission = 'submission.csv'
####Windows Variables###############
''''
train=os.path.join(dir,'./input/train.csv')
test=os.path.join(dir,'./input/test.csv')
store=os.path.join(dir,'./input/store.csv')
submission = os.path.join(dir,'./output/submission2.csv')'''

cols=['StoreType','Assortment','PromoInterval','StateHoliday']
dates=['']
PromoIntervalDict={'Jan,Apr,Jul,Oct':[1,4,7,10],'Feb,May,Aug,Nov':[2,5,8,11],'Mar,Jun,Sept,Dec':[3,6,9,12]}
#Thanks to chenglongchen RMSE calculation script

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def RMSPE(y, yhat):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe
    
def oneHotEncoding(data,cols,replace=False):
	vec=DictVectorizer()
	mkdict=lambda row:dict((col, row[col]) for col in cols)
	vecData=pd.DataFrame(vec.fit_transform(data[cols].to_dict(outtype='records')).toarray())
	vecData.columns=vec.get_feature_names()
	vecData.index=data.index
	if replace is True:
		data=data.drop(cols,axis=1)
		data=data.join(vecData)
	return data

def impute(data):
    clf=Imputer(missing_values='NaN', strategy='most_frequent', axis=0).fit(data)
    data=clf.transform(data)
    return pd.DataFrame(data)

def sales_date(data):
    data['Sales_year']=data['Date'].dt.year
    data['Sales_month']=data['Date'].dt.month
    data['Sale_day']=data['Date'].dt.day
    #data['Sale_YearMonth'] = data['Date'].map(lambda x: 1000*x.year + x.month)
    data['Sale_DaysInPeriod']=(data['Date']-min(data['Date'])).astype('timedelta64[D]')

    return data

def loadData():
    print('Loading data')
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    train_data=pd.read_csv(train,engine='python',nrows=150000,parse_dates=['Date'],date_parser=dateparse)
    test_data=pd.read_csv(test,engine='python',nrows=50000,parse_dates=['Date'], date_parser=dateparse)
    store_data=pd.read_csv(store,engine='python')
    train_data=train_data.join(store_data,on='Store',rsuffix='_Store')
    test_data=test_data.join(store_data,on='Store',rsuffix='_Store')

    ids=test_data['Id']
    target=train_data['Sales']
    train_data=oneHotEncoding(train_data,cols,replace=True)
    test_data=oneHotEncoding(test_data,cols,replace=True)
    train_data=sales_date(train_data)
    test_data=sales_date(test_data)
    train_data=train_data.drop(['Customers','StoreType','Assortment','Sales','Store','Date'],axis=1)
    test_data=test_data.drop(['Id','StoreType','Assortment','Store','Date'],axis=1)
    for col in train_data.columns:
            if col not in test_data.columns:
                test_data[col]=np.zeros(test_data.shape[0])
    #train_data=impute(train_data)
    #test_data=impute(test_data)
    return train_data.fillna(0),test_data.fillna(0),target,ids

if __name__ == "__main__":
    train_data,test_data,y,ids=loadData()
    
    print("Started doing cross validation")
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.33, random_state=0)
    model=RandomForestRegressor(n_estimators=15)
   
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    error = RMSPE(y_test,y_pred)
    print("RMSPE:", error)
   
    model=RandomForestRegressor(n_estimators=15)
    print('Writing into submission file')
    model.fit(train_data,y)
    y_submission=model.predict(test_data)
    print(y_submission)
    fo = open(submission, 'w')
    fo.write('ID,Sales\n')
    for i,id in enumerate(ids):
        fo.write('%s,%s\n' % ( id, y_submission[i]) )
    fo.close()
    print('Finished writing into submission file')