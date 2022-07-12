# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Class to predict the probability
class Class_Predict_Prudential:

 def Predict_Prudential(self,df_train,df_test,target):
    # Prepraing Train Test Set        
    y = target
    X = df_train.values

    # Select Model
    clf = ExtraTreesClassifier(n_estimators=750,max_features=50, \
                               criterion= 'entropy',min_samples_split= 4, \
                               max_depth= 35, min_samples_leaf= 2, \
                               n_jobs = -1, random_state=12)
                               
    clf.fit(X,y)

    # Prediction probability
    value_predict=clf.predict(df_test.values)
    
    # Submission
    submission=pd.read_csv('../input/sample_submission.csv')
    submission.index=submission.Id
    submission.Response=value_predict[:]
    submission.to_csv('./Prudential_ETC.csv', index=False)
    return;

# Load Data
df_train = pd.read_csv('../input/train.csv')     # 114321 rows x 133 columns
df_test = pd.read_csv('../input/test.csv')  # 114393 rows x 132 columns

# Drop columns
target = df_train['Response'].values
df_train=df_train.drop(['Id','Response'],axis=1)
df_test=df_test.drop(['Id'],axis=1)

# Feature Processing
refcols=df_train.columns
df_train=df_train.fillna(999)
df_test=df_test.fillna(999)

for elt in refcols:
    if df_train[elt].dtype=='O':
        df_train[elt], temp = pd.factorize(df_train[elt])
        df_test[elt]=temp.get_indexer(df_test[elt])
    else:
        df_train[elt]=df_train[elt].round(5)
        df_test[elt]=df_test[elt].round(5)
        
# Drop columns round 2

Xx=df_train.as_matrix()
yy = target
Selector = SelectKBest(chi2, k=100)
Selector.fit_transform(Xx,yy)
Col=df_train.columns.values
data = np.array([ np.append([''],Col),np.append(['0'],Selector.scores_)])
score= pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:])
score.iloc[0] = pd.to_numeric(score.iloc[0])
Sorted_score = score.sort_values(by='0', axis=1)
Drop_list = list(Sorted_score.columns[:24])
df_train=df_train.drop(Drop_list,axis=1)
df_test=df_test.drop(Drop_list,axis=1)


# Call the class and funtion to get prediction probability file.        
a = Class_Predict_Prudential()
a.Predict_Prudential(df_train,df_test,target)