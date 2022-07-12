# Importing necessary libraries 
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv
from sklearn.ensemble import RandomForestClassifier

np.random.seed(889)

# Importing training and test data
df_train = pd.read_csv("../input/train.csv")
df_test  = pd.read_csv("../input/test.csv")   
#df_train=df_train.iloc[1:8000,:]

tryfeatures = ['ID',
                    'TARGET',
                    #'saldo_var30',
                    #'var15',
                    #'saldo_var5',
                    #'ind_var30',
                   # 'var38',
                    #'saldo_medio_var5_ult3',
                    #'num_meses_var5_ult3',
                    #'var36',
                    'num_meses_var39_vig_ult3',	
                    'num_var5',
                    'var36',
                    'saldo_medio_var5_hace3'
                    #'num_meses_var39_vig_ult3'
                    ]
                    
tryfeaturesTest = ['ID',
                    #'saldo_var30',
                    #'var15',
                    #'saldo_var5',
                    #'ind_var30',
                    #'var38',
                    #'saldo_medio_var5_ult3',
                    #'num_meses_var5_ult3',
                    #'var36',
                    'num_meses_var39_vig_ult3',	
                    'num_var5',
                    'var36',
                    'saldo_medio_var5_hace3'
                    #'num_meses_var39_vig_ult3'
                    ]
df_train=df_train[tryfeatures]
df_test=df_test[tryfeaturesTest]

# Defining training and test sets
id_test = df_test['ID']
y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values
X_test = df_test.drop(['ID'], axis=1).values

# Applying the method
clf = RandomForestClassifier(n_estimators=4, max_depth=4)

# Cross validating and checking the score
scores = cv.cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=4) 
print(scores.mean())

# Fitting the model and making predictions
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

# Making submission
submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred[:,1]})
submission.to_csv("z_rf4_tryFeatures_Apr10.csv", index=False)