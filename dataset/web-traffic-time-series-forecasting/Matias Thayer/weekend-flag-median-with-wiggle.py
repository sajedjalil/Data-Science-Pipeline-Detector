import pandas as pd
import numpy as np

def get_raw_data():
    train = pd.read_csv("../input/train_2.csv")
    test = pd.read_csv("../input/key_2.csv")
    return train, test

def transform_data(train, test, periods=-49):
    train_flattened = pd.melt(train[list(train.columns[periods:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
    train_flattened = get_features(train_flattened)
    test['date'] = test.Page.apply(lambda a: a[-10:])
    test['Page'] = test.Page.apply(lambda a: a[:-11])
    test = get_features(test)
    return train_flattened, test

def get_features(df):
    df['date'] = df['date'].astype('datetime64[ns]')
    df['weekend'] = (df.date.dt.dayofweek // 5).astype(float)
    #df['shortweek'] = ((df.date.dt.dayofweek) // 4 == 1).astype(float)
    return df

def predict_using_median_weekend(train, test):
    df = train.copy()
    agg_train_weekend = df.groupby(['Page', 'weekend']).median().reset_index()
    test_df = test.merge(agg_train_weekend, how='left')
    result = test_df['Visits'].values
    return result

def wiggle_preds(df):
    second_term_ixs = df['date'] > '2017-10-13'
    adjusted = df['Visits'].values + df['Visits'].values*0.02
    adjusted[second_term_ixs] = df['Visits'].values[second_term_ixs] + df['Visits'].values[second_term_ixs]*0.04
    df['Visits'] = adjusted
    df.loc[df.Visits.isnull(), 'Visits'] = 0
    df['Visits'] = np.round(df['Visits'].values)
    return df

if __name__ == '__main__':
    train, test = get_raw_data()
    train, test = transform_data(train, test, periods=-49)
    preds_weekend = predict_using_median_weekend(train, test)
    test['Visits'] = preds_weekend
    test = wiggle_preds(test)

    test[['Id','Visits']].to_csv('sub_mads_new_data_ok.csv', index=False)
    print(test[['Id', 'Visits']].head(10))
    print(test[['Id', 'Visits']].tail(10))
