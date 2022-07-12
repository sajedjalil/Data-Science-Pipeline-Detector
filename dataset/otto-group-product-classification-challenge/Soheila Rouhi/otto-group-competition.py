
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns
sns.set()


''''
Steps:
1- check data, class distribution, feature scale etc.
2- Select best model: Here first decision tree is used, then KNN 
3- Decision tree for feature selection and then Random Forest classifier  
4- Try XGBoost on (60) selected features by Decision tree on step 3
'''

df=pd.read_csv('train.csv')
#print(df.info())  # No missing data; All integer; 93 feature; target is class name (string)

classes=df.target.nunique() # nine class!

#print(df.groupby('target').size())
#print(df.describe())

useless_cols=['id']
df=df.drop(columns=useless_cols)



# convert target to separate classes function
def convert(dataframe, column):
    dataframe[column]=dataframe[column].astype('category')
    df_new=pd.get_dummies(dataframe, column) # Convert target to numerical values
    cols=df_new.columns.tolist()
    cols_new=[]
    for col in cols:
        col=col.replace('target_', '')
        cols_new.append(col)
    df_new.columns=cols_new
    return(df_new)
X_old=convert(df, 'target')

df['target']=df['target'].astype('category')
X_old=pd.get_dummies(df, 'target') # Convert target to numerical values
cols=X_old.columns.tolist()

cols_new=[]
for col in cols:
    col=col.replace('target_', '')
    cols_new.append(col)
X_old.columns=cols_new

# ======================================================================================================================
# test dataset
dtest=pd.read_csv('test.csv')
print('Number of missing values in test data : ' + str(dtest.isnull().sum().max())) #just checking that there's no missing
#X=pd.read_csv('train.csv') # no need to drop anything or change any column
Classes=['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
y=X_old.loc[:, Classes]
X_old=X_old.iloc[:, 0:93]
X_test=dtest.drop(columns=['id'])
print('type x:' + str(type(X_old)))
print(y.head())
# ===================================== building decision tree model ===================================================
# Tuning the parameters of decision Tree
param_dist = {"max_depth": [250, None], "max_features": randint(1, 93), "min_samples_leaf": randint(5, 15), "criterion": ["gini", "entropy"]}
dtree = DecisionTreeClassifier()
dtree_cv = RandomizedSearchCV(dtree, param_dist, cv=10) # Instantiate the RandomizedSearchCV object: dtree_cv
dtree_cv.fit(X_old, y) # Fit it to the data

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(dtree_cv.best_params_))
print("Best score is {}".format(dtree_cv.best_score_))

print(X_test.shape)
print(X_test.info())
print(X_old.shape)
print(y.shape)

y_pred = dtree_cv.predict(X_test)

# Save the predicted values as csv file
Classes=['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
df2=pd.DataFrame(y_pred, columns=Classes)
df1=dtest[['id']]
result=pd.concat([df1, df2], axis=1)
result.to_csv('Classes_Submission_D3.csv', index=False) # save cleaned data as new csv file

# Create a pd.Series of features importance’s
importances = pd.Series(dtree_cv.best_estimator_.feature_importances_, index= X_old.columns)
pruned_cols=importances.nlargest(60).index.tolist()
print(pruned_cols)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='green')
#plt.title('Features Importances')
#plt.show()

# Use the selected features in the last step and then apply Random forest
X_new=X_old.loc[:, pruned_cols]
X_test_new=dtest.loc[:, pruned_cols]
print(X_new.shape)
print(X_test_new.shape)
print(y.shape)

# ======================= Apply XGBoost on selected features by D3 ===============

from xgboost import XGBClassifier

xgb = XGBClassifier(max_depth=90, n_estimators=500, learning_rate=0.02, silent=1, objective='multi:softprob', num_class= 9)

y = df[['target']]
xgb.fit(X_new, y.values.ravel())
preds=xgb.predict_proba(X_test_new)

Classes=['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
df2=pd.DataFrame(preds, columns=Classes)
df1=dtest[['id']]

result=pd.concat([df1, df2], axis=1)
result.to_csv('Classes_Submission_XGBoost.csv', index=False) # save cleaned data as new csv file