import numpy as np
import pandas as pd

np.random.seed(0)


# Loading data
df_train = pd.read_csv('../input/train_users.csv')
df_test = pd.read_csv('../input/test_users.csv')
# Convert relevant columns to dates
df_train['date_account_created'] = pd.to_datetime(df_train['date_account_created'],
                                                  format='%Y-%m-%d')
df_train['timestamp_first_active'] = pd.to_datetime(df_train['timestamp_first_active'],
                                                    format='%Y%m%d%H%M%S')
df_train['date_first_booking'] = pd.to_datetime(df_train['date_first_booking'],
                                                  format='%Y-%m-%d')
# Try and impute the age values
median_age = int(df_train.age[~df_train.age.isnull()].astype('int').median())
df_train.age.fillna(median_age, inplace=True)
df_train['age'] = df_train.age.apply(lambda x: 2015 - int(x) if int(x) > 1000 else int(x))
df_train['age'] = df_train.age.apply(lambda x: median_age if (int(x) > 100) | (int(x) < 16) else int(x))

# Try and impute the gender based off the age
age_gender_lkup = df_train.ix[:, ['age', 'gender']].sort_values('age')\
                                                   .groupby('age')\
                                                   .agg(lambda x : x.value_counts().index[0])
age_gender_lkup = age_gender_lkup.replace('-unknown-', np.nan).fillna(method='bfill').to_dict()
df_train.gender.replace('-unknown-', 'rep', inplace=True)
df_train.gender.replace('OTHER', 'rep', inplace=True)
df_train['gender'] = pd.Series(map(lambda gen, age: gen if gen is not 'rep' else 
                                                    age_gender_lkup['gender'][age], 
                                   df_train['gender'], 
                                   df_train['age']))
df_train.gender.value_counts()
