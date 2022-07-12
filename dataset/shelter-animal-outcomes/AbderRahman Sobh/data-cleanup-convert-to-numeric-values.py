# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Take care of missing values
tf = pd.read_csv('../input/test.csv',header=0)

tf['Name'] = tf['Name'].fillna('NoName')
tf2 = tf
tf2['SexuponOutcome'] = tf2['SexuponOutcome'].fillna('Unknown')
tf2['AgeuponOutcome'] = tf2['AgeuponOutcome'].fillna('Unknown')


# --- Color ---
# Reduce color set, somewhat arbitrarily
tf2['Color'] = tf2['Color'].str.replace('.*Brown.*','Brown')
tf2['Color'] = tf2['Color'].str.replace('.*Black.*','Black')
tf2['Color'] = tf2['Color'].str.replace('.*White.*','White')
tf2['Color'] = tf2['Color'].str.replace('.*Grey.*','Grey')
tf2['Color'] = tf2['Color'].str.replace('.*Gray.*','Grey')
tf2['Color'] = tf2['Color'].str.replace('.*Silver.*','Grey')
tf2['Color'] = tf2['Color'].str.replace('.*Tan.*','Brown')
tf2['Color'] = tf2['Color'].str.replace('.*Chocolate.*','Brown')
tf2['Color'] = tf2['Color'].str.replace('.*Blue.*','Blue')
tf2['Color'] = tf2['Color'].str.replace('.*Yellow.*','Yellow')
tf2['Color'] = tf2['Color'].str.replace('.*Gold.*','Yellow')
tf2['Color'] = tf2['Color'].str.replace('.*Red.*','Orange')
tf2['Color'] = tf2['Color'].str.replace('.*Orange.*','Orange')
tf2['Color'] = tf2['Color'].str.replace('.*Calico.*','Tricolor')
tf2['Color'] = tf2['Color'].str.replace('.*Torbie.*','Tricolor')
tf2['Color'] = tf2['Color'].str.replace('.*Tortie.*','Tricolor')

def function(data):
    if (data in ("Brown","Black","White","Grey","Yellow","Orange","Tricolor")):
        return data
    else:
        return "RandomColor"
    
tf2['Color'] = tf2['Color'].map(function)
tf2['Color'] = tf2['Color'].map( {'Brown': 0, 'Black': 1,\
"White": 2,"Grey": 3,"Yellow": 4,"Orange": 5,"Tricolor": 6,\
"RandomColor": 7} ).astype(int)

# --- Name ---
# Looking at the data there is also a large skew of NoNames.
# Thus, we'll be looking at "Named"=1 vs "NoName"=0
def functionb(data):
    if (data == "NoName"):
        return 0
    else:
        return 1

tf2['Name'] = tf2['Name'].map(functionb)

# --- Animal Type ---
tf2['AnimalType'] = tf2['AnimalType'].map( {'Dog': 0, 'Cat': 1} ).astype(int)

# --- Sex Type ---
# A few possible models for sex type, I'll choose "fixed" versus not (vs nan)
tf2['SexuponOutcome'] = tf2['SexuponOutcome'].map( {'Neutered Male': 1, \
'Spayed Female': 1, 'Intact Male': 0, 'Intact Female': 0,\
       'Unknown': 2} ).astype(int)

# --- DateTime ---
# DateTime is probably best portioned into "days into the year"
# Minutes, seconds, etc. is fairly irrelevant
# Year, month, day all hold a seperate relevancy
# I'll create seperate columns for each item.    
tf2['Year']  = tf2['DateTime']
tf2['Month'] = tf2['DateTime']
tf2['Day']   = tf2['DateTime']

tf2['Year'] = tf2['Year'].str.replace('2013.*','2013') 
tf2['Year'] = tf2['Year'].str.replace('2014.*','2014')      
tf2['Year'] = tf2['Year'].str.replace('2015.*','2015')      
tf2['Year'] = tf2['Year'].str.replace('2016.*','2016')      
tf2['Year'] = tf2['Year'].astype(int)

tf2['Month'] = tf2['Month'].str.replace('.*-01-.*','1') 
tf2['Month'] = tf2['Month'].str.replace('.*-02-.*','2')
tf2['Month'] = tf2['Month'].str.replace('.*-03-.*','3')
tf2['Month'] = tf2['Month'].str.replace('.*-04-.*','4')
tf2['Month'] = tf2['Month'].str.replace('.*-05-.*','5')
tf2['Month'] = tf2['Month'].str.replace('.*-06-.*','6')
tf2['Month'] = tf2['Month'].str.replace('.*-07-.*','7')
tf2['Month'] = tf2['Month'].str.replace('.*-08-.*','8')
tf2['Month'] = tf2['Month'].str.replace('.*-09-.*','9')
tf2['Month'] = tf2['Month'].str.replace('.*-10-.*','10')
tf2['Month'] = tf2['Month'].str.replace('.*-11-.*','11')
tf2['Month'] = tf2['Month'].str.replace('.*-12-.*','12')
tf2['Month'] = tf2['Month'].astype(int)

# Day should represent day of the year, not the month.
# Leap years are not accounted for.
# strptime crashes on Feb 29th of a leap year
# Thus I gave a unique code to the "leap day" when it appears
# Assuming of course...all the values input are real so the error catch works
def function2(adate):
    try:
        bdate = datetime.datetime.strptime(adate,'%Y-%m-%d %H:%M:%S')
        cdate = bdate.timetuple().tm_yday
    except ValueError:
        cdate = 357
    return cdate
    
tf2['Day'] = tf2['Day'].map(function2)
tf2 = tf2.drop(['DateTime'], axis=1)

# --- AgeuponOutcome ---
# AgeuponOutcome needs to be converted to days as the smallest unit.
# Unknown or 0 values need to be re-mapped to something reasonable as well.
# With no real info on where to predict an age from the data
# I'm just going to take a median of "known" values to reduce skew
# 

def function3(age):
    try:
        if re.search('year',age):
            dage = int(age.split()[0]) * 365
        elif re.search('month',age):
            dage = int(age.split()[0]) * 30
        elif re.search('week',age):
            dage = int(age.split()[0]) * 7
        elif re.search('day',age):
            dage = int(age.split()[0])    
        elif re.search('Unknown',age):
            dage = 0
    except TypeError:
        dage = 0
    return dage

tf2['AgeuponOutcome'] = tf2['AgeuponOutcome'].map(function3)
ageNZ = int(np.median(tf2['AgeuponOutcome'].nonzero()[0]))
tf2['AgeuponOutcome'] = tf2['AgeuponOutcome'].replace(0,ageNZ)

# --- Animal Breed ---
# I took the top 10 breeds in the training data, unfortunately a very static method.
# Everything else will be termed "Other".
# This part could be improved upon by using a more dynamic way to select top 10 breeds.

def function4(data):
    if (data in ('Domestic Shorthair Mix','Pit Bull Mix',\
    'Chihuahua Shorthair Mix','Labrador Retriever Mix',\
    'Domestic Medium Hair Mix', 'German Shepherd Mix',\
    'Domestic Longhair Mix', 'Siamese Mix','Australian Cattle Dog Mix',\
    'Dachshund Mix')):
        return data
    else:
        return "Other"
    
tf2['Breed'] = tf2['Breed'].map(function4)
tf2['Breed'] = tf2['Breed'].map( {'Domestic Shorthair Mix': 1,
    'Pit Bull Mix': 2, 'Chihuahua Shorthair Mix': 3,\
    'Labrador Retriever Mix': 4,\
    'Domestic Medium Hair Mix': 5, 'German Shepherd Mix': 6,\
    'Domestic Longhair Mix': 7, 'Siamese Mix': 8,\
    'Australian Cattle Dog Mix': 9,\
    'Dachshund Mix': 10, 'Other': 11} ).astype(int)


# This completes the Data Cleaning process and converting to categorical to numeric values.
#
# Features: Name, AnimalType, SexuponOutcome, AgeuponOutcome, Breed, Color,
# Year, Month, Day
#
# Predictions will be made for: Return_to_owner, Euthanasia, Adoption, Transfer, Died
