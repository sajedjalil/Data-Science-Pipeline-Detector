
# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import xgboost as xgb

from sklearn import preprocessing

# get airbnb & test csv files as a DataFrame
airbnb_df  = pd.read_csv('../input/train_users_2.csv')
test_df    = pd.read_csv('../input/test_users.csv')

# drop unnecessary columns, these columns won't be useful in analysis and prediction
airbnb_df  = airbnb_df.drop(['date_account_created','timestamp_first_active'], axis=1)
test_df    = test_df.drop(['date_account_created','timestamp_first_active'], axis=1)

# age
# assign all age values > 100 to NaN, these NaN values will be replaced with real ages below
airbnb_df["age"][airbnb_df["age"] > 100] = np.NaN
test_df["age"][test_df["age"] > 100]     = np.NaN

# get average, std, and number of NaN values in airbnb_df
average_age_airbnb   = airbnb_df["age"].mean()
std_age_airbnb       = airbnb_df["age"].std()
count_nan_age_airbnb = airbnb_df["age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test   = test_df["age"].mean()
std_age_test       = test_df["age"].std()
count_nan_age_test = test_df["age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_airbnb - std_age_airbnb, average_age_airbnb + std_age_airbnb, size = count_nan_age_airbnb)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

# fill NaN values in Age column with random values generated
airbnb_df["age"][np.isnan(airbnb_df["age"])] = rand_1
test_df["age"][np.isnan(test_df["age"])]     = rand_2

# convert type to integer
airbnb_df['age'] = airbnb_df['age'].astype(int)
test_df['age']   = test_df['age'].astype(int)

# first_affiliate_tracked

# fill NaN values randomly
count_first_affiliate = 7    # len(np.unique(airbnb_df["first_affiliate_tracked"].value_counts()))

count_nan_department_airbnb = airbnb_df["first_affiliate_tracked"].isnull().sum()
count_nan_department_test   = test_df["first_affiliate_tracked"].isnull().sum()

rand_1 = np.random.randint(0, count_first_affiliate, size = count_nan_department_airbnb)
rand_2 = np.random.randint(0, count_first_affiliate, size = count_nan_department_test)

range_departments_airbnb = airbnb_df['first_affiliate_tracked'].value_counts().index
range_departments_test   = test_df['first_affiliate_tracked'].value_counts().index

airbnb_df["first_affiliate_tracked"][airbnb_df["first_affiliate_tracked"] != airbnb_df["first_affiliate_tracked"]] = range_departments_airbnb[rand_1]
test_df["first_affiliate_tracked"][test_df["first_affiliate_tracked"] != test_df["first_affiliate_tracked"]]       = range_departments_test[rand_2]
# signup_method
airbnb_df["signup_method"] = (airbnb_df["signup_method"] == "basic").astype(int)
test_df["signup_method"]   = (test_df["signup_method"] == "basic").astype(int)
# signup_flow
airbnb_df["signup_flow"] = (airbnb_df["signup_flow"] == 3).astype(int)
test_df["signup_flow"]   = (test_df["signup_flow"] == 3).astype(int)
# language
airbnb_df["language"] = (airbnb_df["language"] == 'en').astype(int)
test_df["language"]   = (test_df["language"] == 'en').astype(int)
# affiliate_channel
airbnb_df["affiliate_channel"] = (airbnb_df["affiliate_channel"] == 'direct').astype(int)
test_df["affiliate_channel"]   = (test_df["affiliate_channel"] == 'direct').astype(int)
# affiliate_provider
airbnb_df["affiliate_provider"] = (airbnb_df["affiliate_provider"] == 'direct').astype(int)
test_df["affiliate_provider"]   = (test_df["affiliate_provider"] == 'direct').astype(int)


for f in airbnb_df.columns:
    if f == "country_destination" or f == "id": continue
    if airbnb_df[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(airbnb_df[f].values) + list(test_df[f].values)))
        airbnb_df[f] = lbl.transform(list(airbnb_df[f].values))
        test_df[f]   = lbl.transform(list(test_df[f].values))
        
# country_destination

airbnb_df['booked'] = (airbnb_df['country_destination'] != 'NDF').astype(int)
# test_df['booked']   = (test_df['country_destination'] != 'NDF').astype(int)

# define training and testing sets

X_train = airbnb_df.drop(["country_destination", "id", 'booked'],axis=1)
Y_train = airbnb_df["country_destination"]
X_test  = test_df.drop("id",axis=1).copy()


# modify country_destination to numerical values

country_num_dic = {'NDF': 0, 'US': 1, 'other': 2, 'FR': 3, 'IT': 4, 'GB': 5, 'ES': 6, 'CA': 7, 'DE': 8, 'NL': 9, 'AU': 10, 'PT': 11}
num_country_dic = {y:x for x,y in country_num_dic.items()}

Y_train    = Y_train.map(country_num_dic)

# Xgboost 

params = {"objective": "multi:softmax", "num_class": 12}

T_train_xgb = xgb.DMatrix(X_train, Y_train)
X_test_xgb  = xgb.DMatrix(X_test)

gbm = xgb.train(params, T_train_xgb, 20)
Y_pred = gbm.predict(X_test_xgb)

# convert type to integer
Y_pred = Y_pred.astype(int)

# change values back to original country symbols
Y_pred = Series(Y_pred).map(num_country_dic)

# Create submission

country_df = pd.DataFrame({
        "id": test_df["id"],
        "country": Y_pred
    })

submission = DataFrame(columns=["id", "country"])

# sort countries according to most probable destination country 
for key in country_df['country'].value_counts().index:
    submission = pd.concat([submission, country_df[country_df["country"] == key]], ignore_index=True)

submission.to_csv('sub.csv', index=False)