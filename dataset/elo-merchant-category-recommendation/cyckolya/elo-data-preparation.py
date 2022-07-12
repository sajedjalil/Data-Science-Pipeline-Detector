import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import datetime
import gc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.stats import boxcox
from scipy import stats
import numpy as np
#####################################################################

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
############################################################################################################################    
    
train = pd.read_csv("../input/train.csv")
train = reduce_mem_usage(train)
test = pd.read_csv("../input/test.csv")
test = reduce_mem_usage(test)
#############################################################

historical_transactions = pd.read_csv("../input/historical_transactions.csv")
historical_transactions = reduce_mem_usage(historical_transactions)
historical_transactions = historical_transactions.sample(frac= 0.2, replace=False)
historical_transactions['category_3'] = historical_transactions['category_3'].fillna(
                                            historical_transactions['category_3'].mode()[0])
historical_transactions['category_2'] = historical_transactions['category_2'].fillna(
                                            historical_transactions['category_2'].mode()[0])
historical_transactions['merchant_id'] = historical_transactions['merchant_id'].fillna(
                                            historical_transactions['merchant_id'].mode()[0])

#############################################################################################
merchants = pd.read_csv("../input/merchants.csv")
merchants = reduce_mem_usage(merchants)
new_merchant_transactions = pd.read_csv("../input/new_merchant_transactions.csv")
new_merchant_transactions = reduce_mem_usage(new_merchant_transactions)

new_merchant_transactions['category_3'] = new_merchant_transactions['category_3'].fillna(
                                            new_merchant_transactions['category_3'].mode()[0])
new_merchant_transactions['category_2'] = new_merchant_transactions['category_2'].fillna(
                                            new_merchant_transactions['category_2'].mode()[0])
new_merchant_transactions['merchant_id'] = new_merchant_transactions['merchant_id'].fillna(
                                            new_merchant_transactions['merchant_id'].mode()[0])
###############################################################################################
def convert_dates(df, converts):
    for i in converts:
        df[i] = pd.to_datetime(df[i])
convert_dates(train, ["first_active_month"])
convert_dates(test, ["first_active_month"])
convert_dates(historical_transactions, ["purchase_date"])
convert_dates(new_merchant_transactions, ["purchase_date"])

for df in [historical_transactions,new_merchant_transactions]:
    df['year'] = df['purchase_date'].dt.year
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['month'] = df['purchase_date'].dt.month
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0})
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
################################################################################################

def create_features(df1, df2, column1,variables, statistics):
    start = 0
    mydict_names= {"card_id":"card_id"}
    mydict_stats = {}
    if len(variables):
        for i in statistics:
            count = 1
            mynames = list(df1)
            if not variables[0] + '_' + i in mynames:
                mydict_names[variables[0]] = variables[0] + '_' + i
            else:
                mydict_names[variables[0]] = variables[0] + '_' + i + str(count)
            mydict_stats[variables[0]] = i
            if i != "mode":
                df3 = (df2.groupby('card_id', as_index=False).agg(mydict_stats).rename
                       (columns=mydict_names))
            else:
                df3 = (df2.groupby('card_id',as_index=False).agg(
                       lambda x: stats.mode(x)[0][0]).rename(columns=mydict_names))
            df1 = pd.merge(df1, df3, on='card_id', how='left')
        del variables[0]
        return create_features(df1, df2, column1,variables, statistics)
    else:
        print(list(df1))
        del statistics
        return df1
#############################################################################################
variables = ["purchase_amount", "month_lag", "installments", 'month_diff']
statistics = ["sum", "mean", "max", "min", "var", "median"]
train = create_features(train, historical_transactions, "card_id", variables, statistics)

variables = ["purchase_amount", "month_lag", "installments", 'month_diff']
statistics = ["sum", "mean", "max", "min", "var", "median"]
train = create_features(train, new_merchant_transactions, "card_id", variables, statistics)
##############################################################################################

variables = ["purchase_amount", "month_lag", "installments", "month_diff"]
statistics = ["sum", "mean", "max", "min", "var", "median"]
test = create_features(test, historical_transactions, "card_id", variables, statistics)

variables = ["purchase_amount", "month_lag", "installments", "month_diff"]
statistics = ["sum", "mean", "max", "min", "var", "median"]
test = create_features(test, new_merchant_transactions, "card_id", variables, statistics)

def drop_columns(df, columns):
    try:
        for i in columns:
            df.drop(str(i), axis=1, inplace=True)
    except KeyError:
        print("column named ", i, " is missing in the data frame")
####################################################################################

variables = ["purchase_date"]
statistics = ["max", "min"]
train = create_features(train, historical_transactions, "card_id", variables, statistics)

variables = ["purchase_date"]
statistics = ["max", "min"]
train = create_features(train, new_merchant_transactions, "card_id", variables, statistics)

###########################################################################################

variables = ["purchase_date"]
statistics = ["max", "min"]
test = create_features(test, historical_transactions, "card_id", variables, statistics)

variables = ["purchase_date"]
statistics = ["max", "min"]
test = create_features(test, new_merchant_transactions, "card_id", variables, statistics)

############################################################################################

convert_dates(train, ["purchase_date_min", "purchase_date_max"])
convert_dates(test, ["purchase_date_min", "purchase_date_max"])

#########################################################################
def convert_deltas(df, converts):
    for i in converts:
        df[i] =  pd.to_datetime(df[i], format='%Y%d%b:%H:%M:%S.%f')
convert_deltas(train, ["purchase_date_min", "purchase_date_max"])
convert_deltas(test, ["purchase_date_min", "purchase_date_max"])


#########################################################################
train['purchase_date_max'] = train['purchase_date_max'].astype(int)
train['purchase_date_min'] = train['purchase_date_min'].astype(int)
test['purchase_date_max'] = test['purchase_date_max'].astype(int)
test['purchase_date_min'] = test['purchase_date_min'].astype(int)

#######################################################################
columns_to_drop = ['card_id', 'first_active_month']
drop_columns(train, columns_to_drop)
drop_columns

#####################################################################
print('''********************************************
            
            **************************
                Submission reached
            **************************
        **********************************************''')
clean_train = train
clean_test = test

clean_train.to_csv("clean_train.csv", index=False)
clean_test.to_csv("clean_test.csv", index=False)