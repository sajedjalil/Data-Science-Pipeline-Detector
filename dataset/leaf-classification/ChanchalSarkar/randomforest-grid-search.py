# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#https://www.kaggle.com/c/leaf-classification

from operator import itemgetter
from time import time
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split,StratifiedKFold,cross_val_score,StratifiedShuffleSplit
from sklearn.metrics import roc_curve,auc,accuracy_score
from scipy import interp
import pickle
from sklearn.externals import joblib

df_train=pd.read_csv('../input/train.csv',header=0)
df_test=pd.read_csv('../input/test.csv',header=0)

#print (df_train.head(5))

#import sys;sys.exit()

def get_features():
    print ("Getting the features to build the model...")
    train=list(df_train.columns.values)
    test=list(df_test.columns.values)
    fields=list(set(train)& set(test))
    fields.remove("id")
    list_of_features=sorted(fields)
    print ("List of Features :",list_of_features)
    return list_of_features


def standardize_values(features):
    standard= StandardScaler()
    standard.fit(df_train[features])
    df_train[features]=standard.transform(df_train[features])
    df_test[features]=standard.transform(df_test[features])

def get_feature_imporance(features):

    n_estimators=10000
    random_state=0
    n_jobs=-1
    print ("Running random forest to get the feature importance with the paramers.. n_estimators:{} random_state :{} n_jobs :{}".format(n_estimators,random_state,n_jobs))

    x_train=df_train[features]
    y_train=df_train["species"]
    feat_labels= df_train.columns[1:]


    forest = RandomForestClassifier(n_estimators=n_estimators,random_state=random_state,n_jobs=n_jobs) ## Create a Random Forest Classifier object
    forest.fit(x_train,y_train) ## Fit the data into the model

    importances=forest.feature_importances_ ## get the feature importance
    # print("Original ",np.argsort(importances))
    indices = np.argsort(importances)[::-1]
    # print (" importances ",importances)
    # print (" indices ",indices)

    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f+1,30,feat_labels[indices[f]],
                                        importances[indices[f]]))

    ## Plot the feature importance in the bar chart.
    plt.title("Feature Importance")
    plt.bar(range(x_train.shape[1]),importances[indices],color='lightblue',align='center')
    plt.xticks(range(x_train.shape[1]),feat_labels[indices],rotation=90)
    plt.xlim([-1,x_train.shape[1]])
    plt.tight_layout()
    #plt.show()  ## Remove the  comment to display the feature imporance in clumn chart.
    #print(forest.predict_proba(x_train)[0])




#### Develop the regression Model
def grid_search_model(features):

    n_estimators=500
    random_state=0
    n_jobs=-1
    print ("Running Random Forest (Grid Search) with parameters n_estimators:{} random_state:{} n_jobs:{} ..".format(n_estimators,random_state,n_jobs))
    x_train = df_train[features] ## get the training data with list of features
    y_train=df_train['species'] ## get target data from the training set
    x_test=df_test[features]## get the features from test data

    #forest=RandomForestClassifier(n_estimators=n_estimators,random_state=random_state, n_jobs=n_jobs)
    #forest.fit(x_train.as_matrix(),y_train)
    forest=RandomForestClassifier(n_estimators=n_estimators,random_state=random_state, n_jobs=n_jobs)
    param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10,30,40],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

    grid_search = GridSearchCV(forest, param_grid=param_grid)
    start = time()
    grid_search.fit(x_train, y_train)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (time() - start, len(grid_search.grid_scores_)))
    return report(grid_search.grid_scores_)

def report(grid_scores, n_top=4):

    print ("The following are the  top %d parameters ......" % n_top)
    param_dict={}
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top] ## get the top n_top parameters
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
        if i==0:
            param_dict=score.parameters
    return param_dict ## return the best parameters

def build_model(parameters,features):
    n_estimators=500
    random_state=0
    n_jobs=-1
    print ("Running Random Forest with parameters n_estimators:{} random_state:{} n_jobs:{} parameters : {} ..".format(n_estimators,random_state,n_jobs,parameters))
    x_train = df_train[features] ## get the training data with list of features
    y_train=df_train['species'] ## get target data from the training set
    x_test=df_test[features]## get the features from test data

    #forest=RandomForestClassifier(n_estimators=n_estimators,random_state=random_state, n_jobs=n_jobs)
    #forest.fit(x_train.as_matrix(),y_train)
    forest=RandomForestClassifier(n_estimators=n_estimators,random_state=random_state, n_jobs=n_jobs,
                                  min_samples_leaf=parameters["min_samples_leaf"],
                                  bootstrap=parameters["bootstrap"],
                                  min_samples_split=parameters["min_samples_split"],
                                  criterion=parameters["criterion"],
                                  max_features=parameters["max_features"],
                                  max_depth=parameters["max_depth"]
                                  )  ### Run the randomforesr model with the best parameters with the highest rank.

    forest.fit(x_train,y_train)  ## Fit the Data.

    print ("Wrting the predicted valus of the test set to the CSV file..........")
    output=pd.DataFrame(columns=[sorted(np.unique(y_train))])

    output= pd.DataFrame(forest.predict_proba(x_test),columns=[sorted(np.unique(y_train))])
    output["id"]=df_test["id"]

    ## Reagganging the columns so that "id"
    cols=output.columns.tolist()
    cols=cols[-1:] + cols[:-1]
    output=output[cols]
    output.to_csv('result.csv',index=False)
    #print("Saving the model to the diskstorage so that the model can use reused...")
    #joblib.dump(forest, 'filename.pkl')


def main():

    features=get_features()
    standardize_values(features)
    get_feature_imporance(features)
    parameters=grid_search_model(features)
    #parameters={'max_features': 10, 'min_samples_leaf': 1, 'bootstrap': True, 'min_samples_split': 1, 'criterion': 'gini', 'max_depth': None}
    build_model(parameters,features)
if __name__==main():
    main()

