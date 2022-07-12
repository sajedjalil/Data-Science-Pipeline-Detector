import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

np.random.seed(0)

# Loading data
df_train = pd.read_csv('../input/train_users.csv')
df_test = pd.read_csv('../input/test_users.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]

#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

# Function to try and impute the age values
def impute_age(df):
    median_age = int(df.age[~df.age.isnull()].astype('int').median())
    df.age.fillna(median_age, inplace=True)
    df['age'] = df.age.apply(lambda x: 2015 - int(x) if int(x) > 1000 else int(x))
    df['age'] = df.age.apply(lambda x: median_age if (int(x) > 100) | (int(x) < 16) else int(x))
    return df

# Function to try and impute the gender based off the age
def impute_gender(df):
    age_gender_lkup = df.ix[:, ['age', 'gender']].sort_values('age')\
                                                 .groupby('age')\
                                                 .agg(lambda x : x.value_counts().index[0])
    age_gender_lkup = age_gender_lkup.replace('-unknown-', np.nan).fillna(method='bfill').to_dict()
    df.gender.replace('-unknown-', 'rep', inplace=True)
    df.gender.replace('OTHER', 'rep', inplace=True)
    df['gender'] = pd.Series(map(lambda gen, age: gen if gen is not 'rep' else 
                                                  age_gender_lkup['gender'][age], 
                                 df['gender'], 
                                 df['age']))
    return df

# Function to one-hot-encoding features
def one_hot_df(df, ohe_feats, train_cols):
    df_temp = df.iloc[:, train_cols]
    
    for f in ohe_feats:
        df_dummy = pd.get_dummies(df_temp[f], prefix=f)
        df_temp = df_temp.drop([f], axis=1)
        df_temp = pd.concat((df_temp, df_dummy), axis=1)

    return df_temp

df_all = impute_age(df_all)
df_all = impute_gender(df_all)
df_all = one_hot_df(df_all, ['gender', 'language'], [4, 5, 8])

vals = df_all.values
X = vals[:piv_train]
X_test = vals[piv_train:]
y = labels

# Run the algo
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X, y)
score = clf.score(X, y)
print(score)

pred = clf.predict(X_test)
print(pred[:50])
