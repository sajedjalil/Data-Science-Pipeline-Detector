import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
import csv
from sklearn import metrics
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
import gzip
import zipfile

def loadData(df,scaler=None):
    data = pd.DataFrame(index=range(len(df)))
    
    data = df.get(['X','Y']) #2
    
    #data = data.join(pd.get_dummies(df['DayOfWeek'])) #7
    DayOfWeeks = df.DayOfWeek.unique()
    DayOfWeekMap = {}
    i = 0
    for s in DayOfWeeks:
        DayOfWeekMap[s] = i
        i += 1
    data = data.join(df['DayOfWeek'].map(DayOfWeekMap))
    
    #data = data.join(pd.get_dummies(df['PdDistrict'])) #10
    PdDistricts = df.PdDistrict.unique()
    PdDistrictMap = {}
    i = 0
    for s in PdDistricts:
        PdDistrictMap[s] = i
        i += 1
    data = data.join(df['PdDistrict'].map(PdDistrictMap))
        
    date_time = pd.to_datetime(df.Dates)
    year = date_time.dt.year
    data['Year'] = year
    month = date_time.dt.month
    data['Month'] = month
    day = date_time.dt.day
    data['Day'] = day
    hour = date_time.dt.hour
    data['hour'] = hour
    minute = date_time.dt.minute
    time = hour*60+minute
    data['Time'] = time
    #data = data.join(pd.get_dummies(year)) #13
    #data = data.join(pd.get_dummies(month)) #12
    
    data['StreetCorner'] = df['Address'].str.contains('/').map(int)
    data['Block'] = df['Address'].str.contains('Block').map(int)
    
    X = data.values
    Y = None
    if 'Category' in df.columns:
        Y = df.Category.values
    
    #scaler = preprocessing.StandardScaler().fit(data.values)
    #X = scaler.transform(data.values)
    return X,Y,scaler
    
def RF(X,Y):
    clf = ensemble.RandomForestClassifier()
    #cross validation
    #n_estimators = [20]
    #clf.set_params(n_estimators=500) #10
    #max_features = ['auto'] 
    #clf.set_params(max_features=None) #['auto','sqrt',None,'log2']
    #bootstrap = [True,False]
    #clf.set_params(bootstrap=False) #Ture, False
    #criterion = ['gini','entropy']
    #clf.set_params(criterion='entropy') #gini, entropy
    #min_samples_split = [9,11,12,13,14,15]
    #clf.set_params(min_samples_split=12) #2
    #min_samples_leaf = [1,2,3,4,5] #1
    scoreSum = 0
    kfold = cross_validation.KFold(len(X),n_folds=3)
    for train, test in kfold:
        #clf.set_params(max_features='log2')
        #clf.set_params(bootstrap=False)
        #clf.set_params(criterion='entropy')
        clf.set_params(min_samples_split=1000)
        clf.fit(X[train],Y[train])
        Yhat = clf.predict_proba(X[test])
        score = metrics.log_loss(Y[test],Yhat)
        print(score)
        scoreSum += score
    print(scoreSum/4.0)
    '''for i in range(len(max_features)):#
        clf.set_params(max_features=max_features[i])##
        score = cross_validation.cross_val_score(clf,X,Y,cv=4,scoring='log_loss')
        print(score)
        avgScore = np.mean(score)
        #print(max_features[i],avgScore)#
        if minScore>avgScore:
            minScore = avgScore
            best_param=max_features[i]#
    print(best_param,minScore)'''
    return clf

def RFpredict(X,Y,Xhat):
    clf = ensemble.RandomForestClassifier()
    #clf.set_params(n_estimators=20)
    #clf.set_params(max_features=None)
    #clf.set_params(criterion='entropy')
    clf.set_params(min_samples_split=1000)
    clf.fit(X,Y)
    Yhat = clf.predict_proba(Xhat)
    return Yhat,clf

def NB(X,Y):
    clf = GaussianNB()
    #clf = MultinomialNB()
    #clf = BernoulliNB()
    scoreSum = 0
    kfold = cross_validation.KFold(len(X),n_folds=3)
    for train, test in kfold:
        #clf.set_params(max_features=max_features[0])
        clf.fit(X[train],Y[train])
        Yhat = clf.predict_proba(X[test])
        score = metrics.log_loss(Y[test],Yhat)
        print(score)
        scoreSum += score
    print(scoreSum/4.0)
    return clf

def NBpredict(X,Y,Xhat):
    clf = GaussianNB()
    clf.fit(X,Y)
    Yhat = clf.predict_proba(Xhat)
    return Yhat,clf
    
#import os
#print(os.listdir('../input'))
#mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")

train = pd.read_csv("../input/train.csv")
X,Y,scaler = loadData(train)
#clf = RF(X,Y)
#clf = NB(X,Y)

test = pd.read_csv("../input/test.csv")
Xhat,_,__ = loadData(test,scaler)
Yhat,clf = RFpredict(X,Y,Xhat)
#Yhat,clf = NBpredict(X,Y,Xhat)

submission = pd.DataFrame(Yhat,columns=clf.classes_)
submission['Id'] = test.Id.tolist()
submission.to_csv(gzip.open('RF.csv.gz','wt'),index=False)

'''submission = pd.DataFrame({'Id':test.Id.tolist()})
for i in range(len(Yhat[0])):
    submission[clf.classes_[i]] = Yhat[:,i]
submission.to_csv('submission.csv',index=False)'''


'''with gzip.open('NB.csv.gz','wt') as fw: #must be 'wt', not 'w'. 't' means text, different from binary
    writer = csv.writer(fw,lineterminator='\n')
    writer.writerow(['Id']+clf.classes_.tolist())
    for i in range(len(Yhat)):
        writer.writerow([i]+Yhat[i].tolist())'''

#z = zipfile.ZipFile('../input/train.csv.zip')
#train = pd.read_csv(z.open('train.csv'))


