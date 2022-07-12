# The goal of this script is to just throw a bunch of models at the data 
# I did this to practice creating models using Sklearn
# Some likely make no sense from a data science perspective
# I welcome any and all feedback.
# My first Kaggle script! 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Create Enumerated Labels for Important Columns
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

def enumerate_labels(df2,outcome=True):
    # prepocess text labels to numeric
    cols = [u'AnimalType',u'Breed',u'Color']
    for c in cols: 
        label = "%s_N" % c
        textlabels = df2[c].unique()
        le.fit(textlabels)
        df2[label] = le.transform(df2[c])
    
    if outcome: 
        outcomes = df2['OutcomeType'].unique()
        le.fit(outcomes)
        df2['OutcomeType_N'] = le.transform(df2['OutcomeType'])
            
            
# Calculate Age In Days Using Pandas String Methods! 
def calc_age_in_days(df1): 
    ages = df1[u'AgeuponOutcome']
    splits = ages.str.extract('(\d{1,2})\s([year,month,day,week])s*')
    splits[1] = splits[1].fillna(0)

    df1.loc[splits[1] == 'y',('AgeInDays')] = splits.loc[splits[1] == 'y'][0].astype('int') * 365
    df1.loc[splits[1] == 'm',('AgeInDays')] = splits.loc[splits[1] == 'm'][0].astype('int') * 30
    df1.loc[splits[1] == 'w',('AgeInDays')] = splits.loc[splits[1] == 'w'][0].astype('int') * 7
    df1.loc[splits[1] == 'd',('AgeInDays')] = splits.loc[splits[1] == 'd'][0].astype('int')

    df1['AgeInDays'] = df1['AgeInDays'].fillna(method='pad').astype('int')
# Get train(df) and test data 
df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Create a Few New Variables
outcomes = df['OutcomeType'].unique()
df['NameLen'] = df['Name'].str.len()
test['NameLen'] = df['Name'].str.len()

# Prep Data
calc_age_in_days(df)
calc_age_in_days(test)

enumerate_labels(df)
enumerate_labels(test, outcome=False)

# plot Outcomes by Animal Type on bar graph
cat_outcomes = df[df['AnimalType'] == 'Cat']['OutcomeType_N']
dog_outcomes = df[df['AnimalType'] == 'Dog']['OutcomeType_N']

# make a general historgram function that we'll use later 
def make_hist(datadict,labels=['Cat','Dog']):
    ax = plt.subplot(111)
    plt.hist(datadict,alpha=0.5,normed=True,stacked=False,bins=5,label=labels)
    plt.xticks(cat_outcomes.unique(), outcomes, rotation='vertical')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,loc=2)
    return ax

make_hist([cat_outcomes,dog_outcomes])

# Data Prep: This helper function lets us pass colums to a dataframe and retrieve an X (factor array) and y (target outcome)
# This lets us shape the data to use in the sklearn models
# Note, this uses a the fillna(0) method to change any NaN to 0.0. This might effect results. Any suggestions to handle this differently are very welcome. 
# Columns to include in Model
def getXY(data, cols=[u'AnimalType_N',u'Breed_N',u'Color_N','AgeInDays','NameLen'],y='OutcomeType_N',animal=None): 
    if animal: 
        newdata = data.loc[data['AnimalType'] == animal]
    else:
        newdata = data

    X = newdata[cols]

    if y: 
        y = newdata[y].fillna(0)
    else: 
        y = np.zeros(len(newdata.index))

    return X.fillna(0),y
    
# Ok, let's party. 
# First, a quick OLS model. This is kind of garbage, but shows a little info
from pandas.stats.api import ols

X,y = getXY(df)
model = ols(x=X,y=y)
print(model)

# Hm, color and name length don't seem to be of much value, let's remove
newX,y = getXY(df,cols=[u'Breed_N','AgeInDays'],animal='Dog')
model2 = ols(x=newX,y=y)
print(model2)
# Somewhat better, but not great. Let's try logit from sklearn. 
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(newX,y) 
lr.coef_

# Use test data to create a prediction. 
Xhat,yhat = getXY(test,cols=[u'Breed_N','AgeInDays'],animal='Dog',y=None)
yhat = lr.predict(Xhat)

# I'm not looking closely at individual data, just making a similar historgram to above to see how it matches
# This one is OK, but not great. You'd expect them to be very similar. 
make_hist([dog_outcomes,yhat],labels=['Actual','Predicted'])
# Just messing around. Maybe a Lasso style (like a Logistic regression but with a few differences described in the sklearn documentation
from sklearn.linear_model import Lasso

X,y = getXY(df,cols=[u'Breed_N','AgeInDays'],animal='Dog')
lr = Lasso()
lr.fit(newX,y) 
lr.coef_

Xhat,yhat = getXY(test,cols=[u'Breed_N','AgeInDays'],animal='Dog',y=None)
yhat = lr.predict(Xhat)

make_hist([dog_outcomes,yhat],labels=['Actual','Predicted'])
# Chances are, this is a KNN problem -- animals that have similar ages and breeds may have similar outcomes.
# Let's try that. 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5,algorithm='ball_tree',weights='distance')
X,y = getXY(df,cols=[u'Breed_N','AgeInDays'])
knn.fit(X,y)

Xhat,yhat = getXY(df,cols=[u'Breed_N','AgeInDays'],y=None)
yhat = knn.predict(Xhat)

make_hist([dog_outcomes,yhat],labels=['Actual','Predicted'])


# Predicted is now closer to the Actual. Let's try Random Forest. 
from sklearn.ensemble import RandomForestClassifier

X,y = getXY(df,cols=[u'Breed_N','AgeInDays'],animal='Dog')
lr = RandomForestClassifier()
lr.fit(X,y) 

Xhat,yhat = getXY(test,cols=[u'Breed_N','AgeInDays'],animal='Dog',y=None)
yhat = lr.predict(Xhat)

logprob = lr.predict_log_proba(Xhat)
make_hist([dog_outcomes,yhat],labels=['Actual','Predicted'])

# This looks closest to me. Let's look at the probabilties and output those as our outcome
prob = lr.predict_proba(Xhat)
print(prob)