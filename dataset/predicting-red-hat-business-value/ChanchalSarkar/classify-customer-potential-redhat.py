# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split,StratifiedKFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc , roc_auc_score, accuracy_score
from scipy import interp
import matplotlib.pyplot as plt

## read data file: people in a pandas dataframe
df_people=pd.read_csv("../input/people.csv",header=0,
                       dtype={'people_id': np.str,
                              'activity_id': np.str,
                              'char_38': np.int32},
                       parse_dates=['date'])


## read data file: activity: training in a pandas dataframe
df_train_people_activity = pd.read_csv("../input/act_train.csv",header=0,
                        dtype={'people_id': np.str,
                               'activity_id': np.str,
                               'outcome': np.int8},
                        parse_dates=['date'])


## read the activity : test in pandas dataframe
df_test_people_activity=pd.read_csv("../input/act_test.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str},
                       parse_dates=['date'])

### define function to pre-process the data
def pre_process_data():

    print ("Processing Training and Test Data set..")
    table_train_test =[df_train_people_activity,df_test_people_activity]

    ## Derive Year,month and Day from the data field. Drop the date field
    for data_frame in table_train_test:
        data_frame['year']=data_frame['date'].dt.year
        data_frame['month']=data_frame['date'].dt.month
        data_frame['day']=data_frame['date'].dt.day
        data_frame.drop('date',axis=1,inplace=True)
        ## remove the string 'type' from the value so that the field becomes an integer.
        data_frame['activity_category']=data_frame['activity_category'].str.lstrip('type ').astype(np.int32)

        ## fill missing values with default: "type -777" . then remove the string "type " from the field so the field can be convtered to interger.
        for i in range(1, 11):
            data_frame['char_' + str(i)].fillna('type -777', inplace=True)
            data_frame['char_' + str(i)]= data_frame['char_' + str(i)].str.lstrip('type ').astype(np.int32)

    ## Derive Year,month and Day from the data field. Drop the date field
    df_people['year']=df_people['date'].dt.year
    df_people['month']=df_people['date'].dt.month
    df_people['day']=df_people['date'].dt.day
    df_people.drop('date',axis=1,inplace=True)

    ## remove the string 'type' from the value so that the field becomes an integer.
    for i in range(1,10):
        df_people['char_'+str(i)]=df_people['char_'+str(i)].str.lstrip('type ').astype(np.int32)

    ## convert the fields to integer
    for i in range(10,38):
        df_people['char_'+str(i)]=df_people['char_'+str(i)].astype(np.int32)

    ## remove the string "group" from the values so the field can be converted to integer.
    df_people['group_1']=df_people['group_1'].str.lstrip('group ').astype(np.int32)


    print ("Merging the Dataframes based on people_id....")
    df_train= pd.merge(df_people,df_train_people_activity,how='inner',on='people_id') ## join df_train_people_activity and df_train_people
    df_train.fillna(-777, inplace=True)
    df_test= pd.merge(df_people,df_test_people_activity,how='inner',on='people_id')## join df_test_people_activity and df_train_people
    df_test.fillna(-777,inplace=True)

    ## get the features for model building. Exclude the fields people_id,activity_id. These fields do not seem to have good predictive power.
    print ("Getting the features that will be used to build the model......")
    l1=list(df_train.columns.values)
    l2=list(df_test.columns.values)
    fields=list(set(l1)& set(l2))
    fields.remove('people_id')
    fields.remove('activity_id')
    list_of_features=sorted(fields)
    return df_train,df_test,list_of_features



def RandomForest(df_train,df_test,list_features):

    n_estimators=100
    random_state=0
    n_jobs=5
    n_folds=3
    print ("Running Random Forest with parameters n_estimators:{} random_state:{} n_jobs:{} n_folds: {} ..".format(n_estimators,random_state,n_jobs,n_folds))
    x_train = df_train[list_features] ## get the training data with list of features
    y_train=df_train['outcome'] ## get target data from the training set
    x_test=df_test[list_features]## get the features from test data

    cv=StratifiedKFold(y_train,n_folds=n_folds,random_state=random_state) ## define the Kfold cross validation
    pipeline_lr = Pipeline([('forest',RandomForestClassifier(n_estimators=n_estimators,random_state=random_state, n_jobs=n_jobs))])

    ######### Begin :Draw the ROC curve for different fold validation.
    print ("At the end the process will draw the ROC curve")
    mean_tpr=0
    mean_fpr=np.linspace(0,1,100)
    all_tpr=[]
    fpr='' ## false positive rate
    tpr='' ## true positie rate
    for i, (train,test) in enumerate(cv):
        probas = pipeline_lr.fit(x_train[list_features].as_matrix()[train],y_train.as_matrix()[train]).predict_proba(x_train[list_features].as_matrix()[test] )
        fpr,tpr,threshold = roc_curve(y_train.as_matrix()[test],probas[:,1],pos_label=1)
        mean_tpr+=interp(mean_fpr,fpr,tpr)
        mean_tpr[0]=0.0
        roc_auc=auc(fpr,tpr)
        plt.plot(fpr, tpr,lw=1,label='ROC fold %d (area = %0.2f)' % (i+1,roc_auc))

    plt.plot([0,1],[0,1],linestyle='--',color=(0.6,0.6,0.6,0.6),label='Random Guessing')
    mean_tpr /= len(cv)
    mean_tpr[-1]=1.0
    mean_auc = auc(mean_fpr,mean_tpr)
    plt.plot(mean_fpr,mean_tpr,'k--',label='Mean ROC (area =%0.2f)' % mean_auc, lw=2 )
    plt.plot([0,0,1],[0,1,1],lw=2,linestyle=':',color='black',label='Perfect Performance')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False postive rate')
    plt.ylabel('True postive rate')
    plt.title('Receiver Operating Characteristics')
    plt.legend(loc="lower right")

    ## write the result to the file
    print ("Writing the result to the CSV file....")
    file=open('submission.csv','w')
    y_pred = pipeline_lr.predict(x_test[list_features])
    i=0
    header="activity_id,outcome"
    header=header+'\n'
    file.write(header)
    for activity_id in (df_test['activity_id']):
        str="{},{}".format(activity_id,y_pred[i])
        str=str+'\n'
        file.write(str)
        i+=1
    file.close()

    plt.show() ## Show the ROC curve.
    ######### End of : Draw the ROC curve for different fold validation.



def main():

    train,test,list_features=pre_process_data()
    print ("Features : [{}], {} ".format(len(list_features),list_features))
    RandomForest(train,test,list_features)
    #GradientBoosting(train,test,list_features)


if __name__ == '__main__':
    main()



