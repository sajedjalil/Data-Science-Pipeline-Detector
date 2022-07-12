from pandas import Series,DataFrame
import pandas as pd

# numpy, matplotlib, seaborn
import numpy as np

# machine learning

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import maxabs_scale
import xgboost as xgb

# get homesite & test csv files as a DataFrame
homesite_df = pd.read_csv("../input/train.csv")
test_df     = pd.read_csv("../input/test.csv")

# preview the data
#homesite_df.head()

# drop unnecessary columns, these columns won't be useful in analysis and prediction
homesite_df = homesite_df.drop(['QuoteNumber'], axis=1)
# date

# Convert Date to Year, Month, and Week
homesite_df['Year']  = homesite_df['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
homesite_df['Month'] = homesite_df['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
homesite_df['Week']  = homesite_df['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))

test_df['Year']  = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
test_df['Month'] = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
test_df['Week']  = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))

homesite_df.drop(['Original_Quote_Date'], axis=1,inplace=True)
test_df.drop(['Original_Quote_Date'], axis=1,inplace=True)

# fill NaN values
homesite_df.fillna(-1, inplace=True)
test_df.fillna(-1, inplace=True)

# There are some columns with non-numerical values(i.e. dtype='object'),
# So, We will create a corresponding unique numerical value for each non-numerical value in a column of training and testing set.

from sklearn import preprocessing

for f in homesite_df.columns:
    if homesite_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(homesite_df[f].values) + list(test_df[f].values)))
        homesite_df[f] = lbl.transform(list(homesite_df[f].values))
        test_df[f] = lbl.transform(list(test_df[f].values))
        
# define training and testing sets
X_train = homesite_df.drop("QuoteConversion_Flag",axis=1)
Y_train = homesite_df["QuoteConversion_Flag"]
X_test  = test_df.drop("QuoteNumber",axis=1).copy()
names=[x for x in homesite_df.columns]

from sklearn.linear_model import RandomizedLasso
rlasso = RandomizedLasso(alpha=0.025)
rlasso.fit(X_train, Y_train)
res=sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), 
                 names), reverse=True)
res.to_csv('res.csv', index=False)

