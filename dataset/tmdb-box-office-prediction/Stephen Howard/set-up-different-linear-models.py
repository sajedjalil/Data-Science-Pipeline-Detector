# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 21:16:28 2019

@author: Howard Family
"""
# Import packages
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import json
import ast

t0 = time.time()
#%%###############################################################################
##################################################################################
##################################################################################
print('Step 1 - Read in data')

#This code is required for running on local machine
#labelled = pd.read_csv('C:/Users/Howard Family/Documents/Kaggle competitions/TMDB Box Office Predictions - Feb 2018/input/train.csv')
#submission = pd.read_csv('C:/Users/Howard Family/Documents/Kaggle competitions/TMDB Box Office Predictions - Feb 2018/input/test.csv')

#This code is required for running in kernels

labelled = pd.read_csv('../input/train.csv')
submission = pd.read_csv('../input/test.csv')


m_lab = len(labelled)

t1 = time.time()
print('Step 1 complete - Time to complete = %3.1f seconds' % (t1-t0))
#%%###############################################################################
##################################################################################
##################################################################################
print('Step 2 - Add extra features to dataset')

# Create the log revenue column, as this is used for assessment
# Create a binary column to assess whether the film made a profit or not

# As revenue is not available in the submission data, this does not need
# to be part of a function
labelled['log_revenue'] = labelled['revenue'].apply(np.log)
labelled['profitable?'] = labelled['revenue'] > labelled['budget']


# Create some extra date features and reformat the existing release date column
# Define various functions to do this
# The creation of the extra features should also be a function, so it can be applied to 
# the submission data

def Release_Year(date):
    ''' (str) -> int
    
    Returns the year from a string of format 'mm/dd/yy', assuming yy greater than or 
    equal to 20 relates to 20th century and yy less than 20 relates to 21st century
    
    >>> Release_Year('2/12/15')
    2015
    >>> Release_Year('5/19/89')
    1989
    '''
    
    if type(date)!=str:
        return 1900
    elif int(date[-2:]) < 20:
        return int(date[-2:]) + 2000
    else:
        return int(date[-2:]) + 1900

def Release_Month(date):
    ''' (str) -> int
    
    Returns the month from a string of format 'mm/dd/yy', 'm/dd/yy',
    'mm/d/yy' or 'm/d/yy'.
    
    >>> Release_Month('2/12/15')
    2
    >>> Release_Year('5/19/89')
    5
    '''    
     
    if type(date)!=str:
        return 1
    else:
        end = date.find('/')
        return int(date[:end])
    
def Release_Day(date):
    ''' (str) -> int
    
    Returns the day from a string of format 'mm/dd/yy', 'm/dd/yy',
    'mm/d/yy' or 'm/d/yy'.
    
    >>> Release_Day('2/12/15')
    2
    >>> Release_Day('19/5/89')
    19
    '''    
    
    if type(date)!=str:
        return 1
    else:
        start = date.find('/')+1
        end = date.find('/',start)
        return int(date[start:end])

def Month_to_Quarter(month):
    ''' int -> int
    
    Returns the quarter for a given month.
    
    Precondition => month is an integer between 1 and 12 inclusive.
    
    >>> Month_to_Quarter(5)
    2
    >>> Month_to_Quarter(12)
    4
    '''      
    return int(1+((month-0.1) // 3))
    
def Date_Reformat(date):
    ''' (str) -> date
    
    Returns the month from a string of format 'dd/mm/yy', 'd/mm/yy',
    'dd/m/yy' or 'd/m/yy'.
    
    >>> Release_Day('2/12/15')
    ?
    >>> Release_Day('19/5/89')
    ?
    '''    
    day = Release_Day(date)
    month = Release_Month(date)
    year = Release_Year(date)
    
    return dt.datetime(year, month, day)
    
def Release_Weekday(date):
    ''' (str) -> str
    
    Returns the day of the week from a string of format 'dd/mm/yy', 'd/mm/yy',
    'dd/m/yy' or 'd/m/yy'.
    
    Monday is day 0, Tuesday day 1, through to Sunday as day 6.
    
    >>> Release_Day('2/12/15')
    ?
    >>> Release_Year('19/5/89')
    ?
    ''' 
    
    return Date_Reformat(date).weekday()
    
def Release_Year_Category(year):
    '''(int) -> str
    
    Returns the category to which the year has been assigned
    1) Pre 1980
    2) 1980 to 1999
    3) 2000 to 2009
    4) 2010 or after
    
    '''
    
    if year < 1980:
        return 'Pre 1980'
    elif year < 2000:
        return '1980 to 1999'
    elif year < 2010:
        return '2000 to 2009'
    else:
        return '2010 or after'

def Amend_Date_Features(movies_df):
    '''(df) -> NoneType
    
    Amends the movies dataframe by adding various new columns based on release_date
    in a string format and then reformats release_date to a DateTime format
    
    '''
    movies_df['release_year'] = movies_df['release_date'].apply(Release_Year)
    movies_df['release_month'] = movies_df['release_date'].apply(Release_Month)
    movies_df['release_day'] = movies_df['release_date'].apply(Release_Day)
    movies_df['release_quarter'] = movies_df['release_month'].apply(Month_to_Quarter)
    movies_df['release_weekday'] = movies_df['release_date'].apply(Release_Weekday)
    movies_df['release_date'] = movies_df['release_date'].apply(Date_Reformat)
    movies_df['release_period'] = movies_df['release_year'].apply(Release_Year_Category)

   
Amend_Date_Features(labelled)


# Create summary of various features and plot
# First output some boxplots - using log revenue works better for the scale

labelled.boxplot('log_revenue','release_month',grid=False)
labelled.boxplot('log_revenue','release_quarter',grid=False)
labelled.boxplot('log_revenue','release_weekday',grid=False)
labelled.boxplot('log_revenue','release_day',grid=False)

# The consider how many movies are in th data by release year and plot release 
# year against log revenue (with budget colour coded).
labelled.hist('release_year',bins=50)

# These plots suggests that older movies only make it onto the database if they 
# are relatively successful (with some exceptions). Therefore, create a 
# categoricaly based on release year. Moved up in code to include in function 
# Amend_Date_Features()

# There appears to be an inflationary effect within the date

# NEXT STEPS
# Sort out missing languages
# Creare two models and figure out how to do predictions efficiently

def Amend_Missing_Languages(df):
    '''(df)-> NoneType
    
    For instances where the spoken_languages column is empty, this replaces the
    value with the original_languages value
    '''
    
#    df['spoken_languages'][df['spoken_languages'] == []] = df['original_language'][df['spoken_languages'] == []]

def Add_Extra_Features(movies_df):
    '''(df) -> NoneType
    
    Amends the movies dataframe by adding various new columns
        
    '''
    movies_df['log_budget'] = movies_df['budget'].apply(np.log)
    movies_df['log_popularity'] = movies_df['popularity'].apply(np.log)
    movies_df['Budget_Known'] = movies_df['budget'].apply(lambda x: x != 0)
    movies_df['Originally_Eng'] = movies_df['original_language'].apply(lambda x: x == 'en')
    movies_df['spoken_languages'] = movies_df['spoken_languages'].apply(lambda x: list(map(lambda d: list(d.values())[0], ast.literal_eval(x)) if isinstance(x, str) else []))
    #movies_df['languages_spoken'][movies_df['languages_spoken']==[]] = movies_df['original_language'][movies_df['languages_spoken']==[]] # replace films with no languages spoken with their original language (20 cases)
    movies_df['no_languages'] = movies_df['spoken_languages'].apply(len)
    movies_df['english_spoken'] = movies_df['spoken_languages'].apply(lambda x: 'en' in x)
    movies_df['runtime'][movies_df['runtime'].isna()] = movies_df['runtime'].mean() # replace films with missing runtime with the average length (2 cases)
    movies_df['runtime'][movies_df['runtime']==0] = movies_df['runtime'].mean() # replace films with zero runtime with the average length (12 cases)
    
    
Add_Extra_Features(labelled)

labelled.boxplot('log_revenue','Budget_Known',grid=False)
labelled.boxplot('log_revenue','Originally_Eng',grid=False)
labelled.groupby(['original_language']).count()['id'].plot.bar()
labelled.boxplot('log_revenue','no_languages',grid=False)
labelled.boxplot('log_revenue','english_spoken',grid=False)
plt.show()
labelled[labelled['Originally_Eng']==False].groupby(['original_language']).count()['id'].plot.bar()
plt.show()
labelled[labelled['Budget_Known']==False].plot.scatter('release_year','log_revenue',c='black')
plt.show()
labelled[labelled['Budget_Known']==True].plot.scatter('release_year','log_revenue',c='budget', colormap='hsv')
plt.show()
labelled[labelled['Budget_Known']==True].plot.scatter('log_budget','log_revenue',c='release_year', colormap='hsv')
plt.show()
labelled.plot.scatter('log_popularity','log_revenue',c='budget', colormap='hsv')
plt.show()
labelled.plot.scatter('runtime','log_revenue',c='budget', colormap='hsv')
plt.show()
    
# Consider grouping by genre, production country and original languages
# Number of spoken languages may be a key factor
# Indicate whether part of a franchise or not
# Look for important keywords
# Pull out some important actors/directors
# Films without a homepage have lower revenue - binary variable??

t2 = time.time()
print('Step 2 complete - Time to complete = %3.1f seconds' % (t2-t1))
#%%###############################################################################
##################################################################################
##################################################################################
# Select the features
# At this stage of code creation, just keep a few features which can be easily used
# Make this into a function so that the same function can be run on test and submission data

print('Step 3 - Select the feature space for modelling')

def Select_Features(movies_df):
    ''' (df) --> df
    
    Returns a new dataframe which is a selection of the features from the full, 
    modified movies dataframe to take through to the modelling stage
    
    '''
    return movies_df[['Budget_Known','log_budget','release_year','no_languages','english_spoken','log_popularity']]

features = Select_Features(labelled)

t3 = time.time()
print('Step 3 complete - Time to complete = %3.1f seconds' % (t3-t2))
#%%###############################################################################
##################################################################################
##################################################################################
print('Step 4 - Create training and test datasets')
# Define features and target
# Use log of target given evaluation criteria

target_true = labelled['revenue']
target_lg = labelled['revenue'].apply(np.log)

# Create training, cross-validation and test data
# 80% of data used for training
# 10% used for cross-validation
# 10% used for final testing


train_X, holdout_X, train_y, holdout_y = train_test_split(features,target_lg,test_size=0.2,random_state=42)

cv_X, test_x, cv_y, test_y = train_test_split(holdout_X, holdout_y, test_size=0.5,random_state=42) 

t4 = time.time()
print('Step 4 complete - Time to complete = %3.1f seconds' % (t4-t3))
#%%###############################################################################
##################################################################################
##################################################################################
print('Step 5 - Parametrise and train a model')

model_budget_known = LinearRegression()
model_budget_missing = LinearRegression()

train_X_budget_known = train_X[train_X['Budget_Known']==True]
train_y_budget_known = train_y[train_X['Budget_Known']==True]

train_X_budget_missing = train_X[train_X['Budget_Known']==False].drop('log_budget',axis=1)
train_y_budget_missing = train_y[train_X['Budget_Known']==False]


model_budget_known.fit(train_X_budget_known,train_y_budget_known)
model_budget_missing.fit(train_X_budget_missing,train_y_budget_missing)

# show a scatter chart of actuals vs predictions for training dataset
def Make_Single_Prediction(datapoint):
    '''(observation from df) -> float
    
    Returns the predicted value from a number of fitted models, selecting the 
    appropriate model depending on the characteristics of the datapoint.
    
    In many cases, it will also be necessary to amend the datapoint.
    '''
    
    if datapoint['Budget_Known']==False:
        return float(model_budget_missing.predict(np.array(datapoint.drop('log_budget')).reshape(1,-1)))
    else:
        return float(model_budget_known.predict(np.array(datapoint).reshape(1,-1)))

def Make_Prediction(movies_df):
    '''(df) -> array
    
    Returns a columns of predicted values, based on the function Make_Single_Prediction
    '''
    
    movies_df.reset_index()
    preds = np.zeros(len(movies_df))
    for i in range(len(movies_df)):
        preds[i] = Make_Single_Prediction(movies_df.iloc[i])
    
    
    return preds

preds_train = np.exp(Make_Prediction(train_X))

actuals_train = np.exp(train_y)

plt.scatter(preds_train,actuals_train)
plt.show()

t5 = time.time()
print('Step 5 complete - Time to complete = %3.1f seconds' % (t5-t4))
#%%###############################################################################
##################################################################################
##################################################################################
print('Step 6 - Make predictions of CV database - calcualte MLS score and plot actuals vs predictions')

preds_CV = np.exp(Make_Prediction(cv_X))
actuals_CV = np.exp(cv_y)

plt.scatter(preds_CV,actuals_CV)
plt.show()

t6 = time.time()
print('Step 6 complete - Time to complete = %3.1f seconds' % (t6-t5))
#%%###############################################################################
##################################################################################
##################################################################################

# Test the model on test dataset

tA = time.time()
#%%###############################################################################
##################################################################################
##################################################################################
print('Step X - Create predictions for submission file')

# Need to modify submissions data first to be consistent with data used to train model
Amend_Date_Features(submission)
Add_Extra_Features(submission)
submission_modified = Select_Features(submission)

# Create predictions on submissions data
# Remember to convert predictions from log values
sub_preds = np.exp(Make_Prediction(submission_modified))

tX = time.time()
print('Step X complete - Time to complete = %3.1f seconds' % (tX-tA))
#%%###############################################################################
##################################################################################
##################################################################################
# Create submissions file
# Needs to be called submission.csv

my_submission = pd.DataFrame({'id': submission['id'],'revenue': sub_preds})
my_submission.to_csv('submission.csv', index=False)

tY = time.time()
print('Step Y complete - Time to complete = %3.1f seconds' % (tY-tX))
#%%###############################################################################
##################################################################################
##################################################################################
tEnd = time.time()
print('Total run time = %3.1f seconds' % (tEnd-t0))
