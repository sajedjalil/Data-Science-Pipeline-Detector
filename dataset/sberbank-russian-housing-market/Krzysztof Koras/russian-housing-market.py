
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler   
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, SGDRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.cross_validation import cross_val_score, ShuffleSplit, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def removing_outliers(df,columns,n_stds=2):
    numeric = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for column in columns:
        if (str(train[column].dtype)) in numeric:
            df= df[(df[column]<=df[column].mean()+n_stds*df[column].std()) & (df[column]>=df[column].mean()-n_stds*df[column].std())]
    return df

    

def nans_in_columns(df,columns, display=False):
    dic={}
    for column in columns:
        nans=df[column].isnull().sum()
        if display:
            print("Column {}: {} NaN values".format(column, nans))
        dic[column]=nans
    return dic


def grid_search_mine(X_train,y_train, estimator, grid, cv, scores,display=False):
    for score in scores:
        print("Searching for best parameters in terms of {} ...".format(score))
        clf=GridSearchCV(estimator, grid, score, cv=cv)
        clf.fit(X_train, y_train)
        print("Best parameters found: {}".format(clf.best_params_))
        print("Performance for best parameters: {}".format(clf.best_score_))
        
        if display:
            print("Performance for all parameters setting:")
            print("")
            for params, mean, scores in clf.grid_scores_:
                print("Paramaters: {}     |     metrics: {} +/- {}".format(params, mean, scores.std()*2))
        print("")
                
    return clf.best_estimator_

def pca_mine(X,n_components=None,listing=False):
    pca=PCA(n_components=n_components)
    pca.fit(X)
    if listing:
        for i in range(len(pca.explained_variance_ratio_)):
            print("Variance explained by {}. component: {}".format(i, pca.explained_variance_ratio_[i]))
    return pca

def feature_select(X,y,column_names,score_func,k,listing=False):
    sel=SelectKBest(score_func,k)
    sel.fit(X,y)
    
    if listing:
        print("Scores for each feature...")
        for i in range(len(sel.scores_)):
            print("Score for {} feature: {}".format(column_names[i],sel.scores_[i]))
    return sel


#Loading all data
train=pd.read_csv("../input/train.csv",parse_dates=["timestamp"])
test=pd.read_csv("../input/test.csv", parse_dates=["timestamp"])
macro=pd.read_csv("../input/macro.csv", parse_dates=["timestamp"]) 

#Removing outliers from full_sq and sadovoe_km columns
train=removing_outliers(train, ["full_sq"])
train=removing_outliers(train, ["sadovoe_km"], n_stds=2.5)

#Scaling price (target) feature using log(1+x)
train.price_doc=np.log1p(train.price_doc)

#Dropping target feature from train df
y_train=train.price_doc
train=train.drop("price_doc", axis=1)

#Concatenating train and test data
all_data=pd.concat((train,test))

#Creating additional features
all_data["month"]=all_data.timestamp.dt.month
all_data["week_of_year"]=all_data.timestamp.dt.week
all_data["apartment"]=all_data.floor.notnull()

#Merging train and test with macro
all_data= pd.merge(all_data, macro, how='left', on='timestamp')

#Dropping columns with NaN values greater than given treshold
nan_dict=nans_in_columns(all_data, all_data.columns)
nan_thresh=0
for column in all_data.columns:
    if nan_dict[column]>nan_thresh:
        if column not in [ "num_room","floor"]:    #Keeping num_room despite many NaN values
            all_data.drop(column,axis=1,inplace=True)
            
#Filling num_room and floor with means
all_data=all_data.fillna(all_data.mean())

#Dropping id and timestamp
all_data.drop(["id", "timestamp"],axis=1,inplace=True)

#Converting categorical features (sub_area) into dummy variables
all_data=pd.get_dummies(all_data)

#Creating sklearn matrices
X_train, y_train=all_data[:train.shape[0]].values, y_train.values
X_test = all_data[train.shape[0]:].values

#Scaling
normal_scaler=StandardScaler()
normal_scaler.fit(X_train)
X_train_scaled=normal_scaler.transform(X_train)

#Feature selection
k=300
selec=feature_select(X_train_scaled,y_train,all_data.columns,f_regression,k=k, listing=False)
X_train_selected=selec.transform(X_train_scaled)

"""

#Cross validation for some classifiers
n_samples=X_train.shape[0]
n_iter=10   #number of re-shuffling & splitting iterations
test_size=0.20
cv=ShuffleSplit(n_samples, n_iter, test_size, random_state=33)

model_lasso = LassoCV(alphas=[1e-09, 1e-08, 1e-07, 1e-06, 0.001, 2.0, 10.0])
scores_lasso=cross_val_score(model_lasso, X_train_selected, y_train, cv=cv, scoring="mean_squared_error")

model_elastic = ElasticNetCV(alphas=[1e-09, 1e-08, 1e-07, 1e-06, 0.001, 2.0, 10.0])
scores_elastic=cross_val_score(model_elastic, X_train_selected, y_train, cv=cv, scoring="mean_squared_error")

model_LR = LinearRegression()
scores_regression=cross_val_score(model_LR, X_train_selected, y_train, cv=cv, scoring="mean_squared_error")

#Printing results
print("Mean squared error for lasso: {} +/- {}".format(-scores_lasso.mean(), 2*scores_lasso.std()))
print("Mean squared error for elastic net: {} +/- {}".format(-scores_elastic.mean(), 2*scores_elastic.std()))
print("Mean squared error for linear regression: {} +/- {}".format(-scores_regression.mean(), 2*scores_regression.std()))

"""

#Evaluation of elastic net using single train/test split procedure

#Shuffling the data
perm=np.random.permutation(X_train.shape[0])
X_train_shuffled=X_train_selected[perm]
y_shuffled=y_train[perm]

#Splitting
X_train_split, X_val, y_train_split, y_val=X_train_shuffled[:23000], X_train_shuffled[23000:], y_shuffled[:23000], y_shuffled[23000:]

#Setting up estimator
model_elastic=ElasticNetCV(alphas=[1e-09, 1e-08, 1e-07, 1e-06, 0.001, 2.0, 10.0])
model_elastic.fit(X_train_split, y_train_split)

#Predictions on validation set
pred=(model_elastic.predict(X_val))

#Root mean squared error on rescaled targets
print("Root mean squared error: {}". format(np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(pred)))))

#Predictions on test set
model_elastic=model_elastic=ElasticNetCV(alphas=[1e-09, 1e-08, 1e-07, 1e-06, 0.001, 2.0, 10.0])
model_elastic.fit(X_train_selected, y_train)

X_test_scaled=normal_scaler.transform(X_test)
X_test_selected=selec.transform(X_test_scaled)

test_predictions=np.expm1(model_elastic.predict(X_test_selected))

#Making submission file
submission = pd.DataFrame({"id": test["id"],"price_doc": test_predictions})
#submission.to_csv('submission.csv', index=False)






