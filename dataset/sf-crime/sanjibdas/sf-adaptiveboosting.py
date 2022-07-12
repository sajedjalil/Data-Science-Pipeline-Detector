
#imports libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time as systime
import datetime as dtime
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import gc



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Make mapping for month
month_enum={1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'}

# Load input files
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'

train_raw = pd.read_csv('../input/'+TRAIN_FILENAME, parse_dates=['Dates'], index_col=False)
test_raw = pd.read_csv('../input/'+TEST_FILENAME, parse_dates=['Dates'], index_col=False)

# Binarize Days, Month, District, Time
def feature_engineering(data):
    
    days = pd.get_dummies(data.DayOfWeek)
    district = pd.get_dummies(data.PdDistrict)
    month = pd.get_dummies(data.Dates.dt.month.map(month_enum))
    hour = data.Dates.dt.hour
    hour = pd.get_dummies(hour) 
 
    #Append newly created dummy variables to dataframe
    new_data = pd.concat([hour, month, days, district], axis=1)
    
    return new_data

# Prepare the data
train = feature_engineering(train_raw)
#test = pd.concat([test_raw['Id'],feature_engineering(test_raw)], axis=1)
test = feature_engineering(test_raw)

# Encode distinct Categories into dummy variables
cat_enc = LabelEncoder()
cat_enc.fit(train_raw['Category'])
train['CategoryEncoded'] = cat_enc.transform(train_raw['Category'])

# Select the Predictors
x_cols = list(train.columns[0:53].values)

# Fit Logit model and estimate the class probability
#clf = xgb.XGBClassifier(n_estimators=5)

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 8),
                              n_estimators = 40,
                              learning_rate = 0.5, 
                              random_state = 1)

clf.fit(train[x_cols], train['CategoryEncoded'])
predicted = clf.predict_proba(test[x_cols])


# Make the output data frame by mapping the probability estimates to categories
crime = cat_enc.fit_transform(train_raw.Category)
result=pd.DataFrame(predicted, columns=cat_enc.classes_)

# I noticed that predicted estimates were having 10 decimal digits or even more.
# Which was giving me memory insufficient error while trying to save it as .csv
# For eg. I tried saving half of the output(442131 records) and .csv file generated
# was of size 370mb. So I rounded of the digits to 4-5 decimal points and output file
# size got reduced.
result=result.round(5)

# Appending the Index column
result= pd.concat([test_raw['Id'], result], axis=1)

del train
del test
del train_raw
del test_raw

gc.collect()

result.to_csv('submit.csv', index = False)



