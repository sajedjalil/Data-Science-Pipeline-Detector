#Importing required libraries
import os
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
#import plotly.plotly as py
#import plotly.offline as py
#py.init_notebook_mode(connected=True)
#import plotly.graph_objs as go
#import plotly.offline as offline
#import plotly.tools as tls
#import plotly.graph_objs as go
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

#os.chdir("E:\Kaggle\DonorsChoose.org_Application_Screening")

# Importing data
res = pd.read_csv("../input/resources.csv")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")



train.head()
res.head()

# Missing values 
train.isnull().sum()    #4 # Ignoring missing values in
test.isnull().sum()     #2 # essay 3 and 4
res.isnull().sum()      #292


#  Combining test and train the data 
project = pd.concat([train, test], axis = 0)
#project = pd.merge(project, res, how = "left", on = "id")


# OverView of the combined data set 
project.head()

project.isnull().sum()


# Filling Missing Values 
project["teacher_prefix"].fillna(method = "ffill", inplace = True)

project["project_essay_3"].fillna(value = "", inplace = True)

project["project_essay_4"].fillna(value = "", inplace = True)


# Combining the project essay 1, 2, 3, 4 
project["essays"] = project["project_essay_1"] + project["project_essay_2"] + project["project_essay_3"] + project["project_essay_4"]
project["essays"][0]


# Exploaration and Feature Engineering 
project['teacher_id'].nunique() #132133

project['teacher_prefix'].unique()

# Does Teacher and Dr has any kind of privelages?
plt.figure()
sns.set(style="whitegrid", color_codes=True)
sns.countplot(x = "project_is_approved", hue = "teacher_prefix",
               data = project)

# School State wise projects 
plt.figure()
sns.countplot(x = 'school_state', data = project)

# Project Submitted Date Time
project["project_submitted_datetime"] = pd.to_datetime(
        project["project_submitted_datetime"])

project["month"] = project["project_submitted_datetime"].dt.month
project["year"] = project["project_submitted_datetime"].dt.year

plt.figure()
sns.countplot(x = "month", hue = "project_is_approved",
              data = project)

# Project Approval Rate
temp = train['project_is_approved'].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
#trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
#layout = go.Layout(title='Project proposal is approved or not')
#py.offline.plot({"data" : [trace], "layout" : layout})


# Project Approved Rate By Teacher Prefix 
temp = pd.crosstab(project["teacher_prefix"], project["project_is_approved"])
temp["ratio"] = (temp[1.0] / (temp[0.0]+temp[1.0])) * 100
#trace = go.Bar(x = temp.index, y = temp["ratio"])
#layout = go.Layout(title = "Approved rate by teacher prefix")
#py.offline.plot({"data" : [trace], "layout" : layout})

# Project Approved Rate by Teacher ID 
temp = project.groupby(["teacher_id", "project_is_approved"])\
["teacher_number_of_previously_posted_projects"].sum().unstack()

temp.fillna(value = 0, inplace = True)

temp.describe()

temp["ratio"] = (temp[1.0] / (temp[0.0]+temp[1.0]))

temp.fillna(value = 0, inplace = True)

project = pd.merge(project, temp[["ratio"]], how = "left",\
                       left_on = "teacher_id", right_index = True)

# Creating some for feature
project["essay1Len"] = project["project_essay_1"].str.len()

project["essay2Len"] = project["project_essay_2"].str.len()
    
# Preparing text data for features
# First Freeing up some memory
import gc
gc.collect()

project.drop(["project_essay_1", "project_essay_2", "project_essay_3",
              "project_essay_4"], inplace = True, axis = 1)

# ubique project id has many description in resources data
# aggregating it based on the project id
res["description"].fillna("Not Available", inplace = True)

res = pd.DataFrame(res.groupby("id").agg({"description" : lambda x : "".join(x),
                   "quantity": ["sum", "mean"],
                   "price" : ["sum", "mean"]}))

res.reset_index(inplace = True)

res.columns =  ["id", "description", "quantity_sum",
                "quantity_mean", "price_sum", "price_mean"]

# Merging the aggregated data with the combined Data frame
project = pd.merge(project, res, on = "id", how = "left")
project["ratio"].fillna(0, inplace = True)
# Preprocess text
print('Preprocessing text...')
cols = [
    'project_title', 
    'essays', 
    'project_resource_summary',
    'description'
]
n_features = [50, 50, 50, 50]

from tqdm import tqdm

for c_i, c in tqdm(enumerate(cols)):
    tfidf = TfidfVectorizer(
        max_features=n_features[c_i],
        norm='l2',
        )
    tfidf.fit(project[c])
    tfidf_train = np.array(tfidf.transform(project[c]).toarray(), dtype=np.float16)


    for i in range(n_features[c_i]):
        project[c + '_tfidf_' + str(i)] = tfidf_train[:, i]
      
        
    del tfidf, tfidf_train
    gc.collect()
    
print('Done.')


cols_to_drop = [
    'id',
    'teacher_id',
    'project_title', 
    'essays', 
    'project_resource_summary',
    'project_is_approved',
    'description',
    'project_submitted_datetime'
]




# Preparin Data For Modelling 

encoding = ["project_grade_category", "project_subject_categories",
            "project_subject_subcategories", "school_state", "teacher_prefix"]

le = LabelEncoder()

for ft in encoding:
    project[ft] = le.fit_transform(project[ft])
    
project.drop(cols_to_drop, axis = 1, inplace = True)

X = project.iloc[0:182080, :]
y = train["project_is_approved"]

print(X.shape, y.shape)

X_test = project.iloc[182080:260115, :]
ids = test["id"]

del [train, test]


#  Data Splitting for validation

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3)



""" Modelling using the random forest """
"""
# Cross validation usin Random Forest
rf_cv = RandomForestClassifier(random_state = 1)

rf_cv.fit(xTrain, yTrain)

roc_auc_score(yTest, rf_cv.predict(xTest)) # 0.6817699636254139
rf = RandomForestClassifier(random_state = 1)
rf.fit(X, y)
pred = rf.predict(X_test)
subFile = pd.DataFrame({"id" : ids, "project_is_approved" : pred})
subFile.to_csv("submission4.csv", index = False)
"""

""" Modelling using gbm """
"""
# Cross Validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(xTrain)

xTrainStd = sc.transform(xTrain)
xTestStd = sc.transform(xTest)

gbm_cv = GradientBoostingClassifier(max_depth = 6, n_estimators = 
                                 random_state = 1, verbose = 1)

gbm_cv.fit(xTrainStd, yTrain.values)

roc_auc_score(yTest, gbm_cv.predict(xTestStd)) # 0.71377


# Trainin on the complete training data set
gbm.fit(X.values, y.values)

pred = gbm.predict(X_test)

subFile = pd.DataFrame({"id" : ids, "project_is_approved" : pred})

subFile.to_csv("submission5.csv", index = False)# 
"""

""" Modelling using Xgboost """
import xgboost as xgb

train = xgb.DMatrix(X, label = y)
#test = xgb.DMatrix(xTest, yTest)

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'

#evallist = [(test, 'eval'), (train, 'train')]

num_round = 1000
#bst = xgb.train(param, train, num_round, evallist)
bst = xgb.train(param, train, num_round)

yPred = bst.predict(xgb.DMatrix(X_test))


subFile = pd.DataFrame({"id" : ids, "project_is_approved" : yPred})
subFile.to_csv("submission7.csv", index = False)
#roc_auc_score(yTest, bst.predict(test)) # 0.71377


### Will update with more features soon 