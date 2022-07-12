import os
import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
from sklearn.cross_validation import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm.classes import LinearSVC
from matplotlib import pyplot as plt
from _collections import defaultdict

def get_cv_roc(X,Y, nb_folds = 10):
    X = X.values
    X, Y = shuffle(X, Y)
    kfolds0 = KFold(len(X[Y==0]), nb_folds)
    kfolds1 = KFold(len(X[Y==1]), nb_folds)
    adaboost_roc = np.zeros(nb_folds)
    for fi,((k_train0, k_valid0),(k_train1,k_valid1)) in enumerate(zip(kfolds0,kfolds1)):
        X_train = np.concatenate([X[Y==0][k_train0],X[Y==1][k_train1]])
        X_valid = np.concatenate([X[Y==0][k_valid0],X[Y==1][k_valid1]])
        Y_train = np.concatenate([Y[Y==0][k_train0],Y[Y==1][k_train1]])
        Y_valid = np.concatenate([Y[Y==0][k_valid0],Y[Y==1][k_valid1]])
        clf = ensemble.AdaBoostClassifier(n_estimators=100  )            
        clf.fit(X_train, Y_train)
        valid_preds = clf.predict_proba(X_valid)
        valid_preds = valid_preds[:, 1]
        adaboost_roc[fi] = metrics.roc_auc_score(Y_valid, valid_preds)
        print ("----------> Fold no.",fi+1)
        print ("roc =", adaboost_roc[fi])
    adaboost_av_roc = np.mean(adaboost_roc)
    adaboost_std_roc = np.std(adaboost_roc)
    print ("Adaboost Average ROC:%s +- %s"%(adaboost_av_roc,adaboost_std_roc))
    return adaboost_av_roc
    
def get_equal_columns(train):
    """
    recevies training data
    return columns on which we can pass because the same data apperas in other columns
    """
    equal_columns = []
    equality = defaultdict(list)
    # we review all columns except for date, id, and latitude because we want to keep those anyway
    columns = [x for x in train.columns.values if x!='Id' and x!='Date' and x!='Longitude' and x!='Latitude']
    # we prefer to keep the longitude column so we will put it first
    if 'Longitude' in train.columns.values:
        columns = ['Longitude'] + columns
    # compare each of the columns pairs
    for i,first_column in enumerate(columns):
        # if we already have an equality for a certain column we can skip on it
        if first_column in equal_columns: 
            continue 
        for j in np.arange(i+1,len(columns)):
            second_column = columns[j]
            if check_two_columns_for_equality(train,first_column,second_column,str_columns = True):
                equality[first_column].append(second_column)
                equal_columns.append(second_column)
    return equal_columns

def remove_columns_which_are_all_nan(X, test = None):
    """
    receives X - dataframe
    remove columns which are all nan in X and in the test data if it is given as well
    """
    for column in X.columns.values:
        if isinstance(X[column].values[0],str): continue
        if np.all(np.isnan(X[column].values)):
            X.drop(column,axis=1,inplace=True)
            if test is not None:
                test.drop(column,axis=1,inplace=True)
    return X,test
    
def shuffle(X, y, seed=3):
    np.random.seed(seed)
    shuffle = np.arange(len(y))
    np.random.shuffle(shuffle)
    X = X[shuffle]
    y = y[shuffle]
    return X, y

def get_clf_discriminant_vector_and_score(clf,X,Y):
    clf.fit(X, Y)
    score = clf.score(X, Y)
    return clf.coef_[0],score

def get_columns_with_no_discriminant_knowledge(X,Y, classifiers_to_check = {'svm':LinearSVC(),
                                                                             'logistic regression':LogisticRegression(),
                                                                             'lda':LinearDiscriminantAnalysis()},
                                               coef_threshold = 1e-4, bad_coefficient = 0.3, good_coefficient=0.8):
    """
    Try to filter columns with no discriminant knowledge using linear classifier and their projection coefficients absolute value
    """
    columns = X.columns.values   
    plt.figure(figsize = (24,15))
    bad_fields = defaultdict(list)
    good_fields = defaultdict(list)
    proj_vec,x_proj = {},{}
    if not isinstance(X,np.ndarray):
        X = X.values
    for clf_name,clf in classifiers_to_check.items():
        coef,_ = get_clf_discriminant_vector_and_score(clf,X,Y)
        indices =  np.argsort(np.abs(coef))
        proj_vec[clf_name] = coef
        x_proj[clf_name] = np.array((np.matrix(X)*np.transpose(np.matrix(coef))))
        # the coeffificients
        for i,(column,coef) in enumerate(zip(columns[indices],coef[indices]/np.sum(np.abs(coef)))):
            if np.abs(coef)<coef_threshold and i<bad_coefficient*len(indices): 
                bad_fields[column].append(coef)
            if i>good_coefficient*len(indices): 
                good_fields[column].append(coef)
    print ("bad_fields: %s"%bad_fields)
    print ("good_fields: %s"%good_fields)
    return [k for k,v in bad_fields.items() if len(v)>=len(classifiers_to_check)]
       


def check_two_columns_for_equality(X,first_column,second_column,str_columns = True):
    """
    receives two vectors of data
    return True if they both contain exactly the same information (in different values)
    """
    x1 = X[first_column].values
    y1 = X[second_column].values
    if len(np.unique(zip(x1,y1)))==len(np.unique(x1)) and len(np.unique(zip(x1,y1)))==len(np.unique(y1)):# len(np.unique(x1))==len(np.unique(y1)):
        return True
    return False
    
os.system("ls ../input")

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample = pd.read_csv('../input/sampleSubmission.csv')

# keep the labels and drop them from the training data, drop also the number of mosquitos which do not exist in the test data
labels = train.WnvPresent.values
train.drop(['WnvPresent','NumMosquitos'],inplace=True,axis=1)

# remove empty columns
train,test = remove_columns_which_are_all_nan(train,test)

# keep the Id column from the test data and drop it from the training data
test_ids = test.Id.values
test.drop(['Id'],inplace=True,axis=1)


# remove columns with exactly the same information
equal_columns = ['Address', 'AddressNumberAndStreet'] #get_equal_columns(train)
train.drop(equal_columns,inplace=True,axis=1)
test.drop(equal_columns,inplace=True,axis=1)
# print ("removed columns %s which contain equal information to other columns"%equal_columns)

# arrange weather data
weather = pd.read_csv('../input/weather.csv')
weather_stn1 = weather[weather['Station']==1]
weather_stn2 = weather[weather['Station']==2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather = weather_stn1.merge(weather_stn2, on='Date')
for st in ['M','-','T',' T','  T']:
    weather = weather.replace(st, np.nan)
equal_columns = ['Depart_y', 'Sunrise_y', 'Sunset_y', 'Depth_y', 'Water1_y', 'SnowFall_y']
# get_equal_columns( weather)
weather.drop(equal_columns,inplace=True,axis=1)
# print ("removed columns %s which contain equal information to other columns"%equal_columns)


categorical_columns = ['CodeSum_x','CodeSum_y']
lbl = preprocessing.LabelEncoder()
for column in categorical_columns:
    if not column in weather.columns.values: continue
    lbl.fit(list(weather[column].values))
    weather[column] = lbl.transform(weather[column].values)


weather,_ = remove_columns_which_are_all_nan(weather)

all_but_date_columns = [x for x in weather.columns.values if x!='Date']


# fill missing values with median values
weather[all_but_date_columns] = weather[all_but_date_columns].astype(float).apply(lambda x: x.fillna(x.median()))


# Extract month and day from dataset
train['month'] = pd.Series(pd.DatetimeIndex(train.Date.values).month, index=train.index)
train['day'] = pd.Series(pd.DatetimeIndex(train.Date.values).day, index=train.index)
# train['day_in_week'] = pd.Series(pd.DatetimeIndex(train.Date.values).weekday, index=train.index)
test['month'] = pd.Series(pd.DatetimeIndex(test.Date.values).month, index=test.index)
test['day'] = pd.Series(pd.DatetimeIndex(test.Date.values).day, index=test.index)
# test['day_in_week'] = pd.Series(pd.DatetimeIndex(test.Date.values).weekday, index=test.index)

# Merge with weather data
train = train.merge(weather, on='Date')
test = test.merge(weather, on='Date')
train.drop(['Date'], axis = 1, inplace=True)
test.drop(['Date'], axis = 1, inplace=True)

# Convert categorical data to numbers
for column in train.columns.values:
    if isinstance(train[column].values[0],str):
        print ("converting %s"%column)
        lbl.fit(list(train[column].values) + list(test[column].values))
        train[column] = lbl.transform(train[column].values)
        test[column] = lbl.transform(test[column].values)

X = train
Y = labels

columns = X.columns.values                                             

# check the training roc of adaboost
clf = ensemble.AdaBoostClassifier(n_estimators=100) 
clf.fit(X, Y)
valid_preds = clf.predict_proba(X)
valid_preds = valid_preds[:, 1]
adaboost_train_roc = metrics.roc_auc_score(Y, valid_preds)  
print ("adaboost train roc = %s"%adaboost_train_roc)
cv_roc = get_cv_roc(X,Y)

clf = ensemble.AdaBoostClassifier(n_estimators=100)
clf.fit(X, labels)
         
# # create predictions and submission file
predictions = clf.predict_proba(test)[:,1]
sample['WnvPresent'] = predictions
sample.to_csv('west_nile_adaboost.csv' , index=False)
