import pandas as pd
from time import process_time
import numpy as np
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Thanks to Chenglong Chen for providing this in the forum


def to_weight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind] ** 2)
    return w


def rmspe(yhat, y):
    w = to_weight(y)
    return np.sqrt(np.mean(w * (y - yhat) ** 2))


def build_features(df, store):
    # store = pd.read_csv('../input/store.csv')
    df = df.merge(store, on='Store', how="left")
    # Break down date column
    df['year'] = df.Date.apply(lambda x: x.year)
    df['month'] = df.Date.apply(lambda x: x.month)
    df['woy'] = df.Date.apply(lambda x: x.weekofyear)

    # Calculate time competition open time in months
    df['CompetitionOpen'] = 12 * (df.year - df.CompetitionOpenSinceYear) + (df.month - df.CompetitionOpenSinceMonth)
    df['CompetitionOpen'] = df.CompetitionOpen.apply(lambda x: x if x > 0 else 0)

    # Promo open time in months
    df['PromoOpen'] = 12 * (df.year - df.Promo2SinceYear) + (df.woy - df.Promo2SinceWeek) / float(4)
    df['PromoOpen'] = df.PromoOpen.apply(lambda x: x if x > 0 else 0)
    df['PromoInterval'] = df.PromoInterval.apply(lambda x: x[0] if type(x) == str else 'N')

    df['StateHoliday'] = df.StateHoliday.apply(lambda x: 1 if (x != '0') else 0)

    df = pd.get_dummies(df, columns=[
        'PromoInterval',
        'StoreType',
        'Assortment'
    ])

    df.drop([
        'Date',
        'Open',
        # 'Store',
        'Promo2SinceYear',
        'Promo2SinceWeek',
        'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear',
        'year'], axis=1, inplace=True)
    return df


# return X_train, y_train, X_test and features names
def get_sets(store_id=None):
    t = process_time()
    train = pd.read_csv('../input/train.csv', parse_dates=['Date'], dtype={'StateHoliday': str})
    test = pd.read_csv('../input/test.csv', parse_dates=['Date'], dtype={'StateHoliday': str})
    store = pd.read_csv('../input/store.csv')
    print('reading data processed in ', process_time() - t)

    if store_id is not None:
        print("found store id. only process model for store ", store_id)
        train = train[train['Store'] == store_id]
        test = test[test['Store'] == store_id]

    print("Consider only open stores for training. Closed stores wont count into the score.")
    train = train[train["Open"] != 0]
    print("Use only Sales bigger then zero. Simplifies calculation of rmspe")
    train = train[train["Sales"] > 0]
    # NaN in test should be handled as open stores
    test[['Open']] = test[['Open']].fillna(1)

    t = process_time()
    train = build_features(train, store)
    test = build_features(test, store)
    print('merging store processed in ', process_time() - t)

    # calculate customer and sales median for each store
    grouped_columns = ['Store', 'DayOfWeek', 'Promo']
    means = train.groupby(grouped_columns)['Sales', 'Customers'].median().reset_index()
    means['SalesMedian'] = means['Sales']
    means['CustomerMedian'] = means['Customers']
    means.drop(['Sales', 'Customers'], axis=1, inplace=True)

    # merge median with train set
    train.drop(['Customers'], axis=1, inplace=True)
    train = train.merge(means, on=grouped_columns, how='left')
    test = test.merge(means, on=grouped_columns, how='left')
    train = train.drop(['Store'], axis=1).astype(float)
    test = test.drop(['Store'], axis=1).astype(float)

    # on sunday there is no sales data, we can safely replace NAN with 0
    # print("any null test: ", np.unique(test[test.isnull().any(axis=1)][['DayOfWeek']].values))
    test.fillna(0, inplace=True)
    train.fillna(0, inplace=True)

    feature_names = train.columns.values.tolist()
    feature_names.remove('Sales')

    return train, test, feature_names


def kfold_validation(name, clf, X_train, y_train):
    t = process_time()
    print("------------- ", name, " --------------")
    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train), " = Coefficient of determination on training set")
    cv = KFold(X_train.shape[0], 5, shuffle=True)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print("Cross validation accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))
    print('train and evaluate model in ', process_time() - t)


def score(name, clf, X_train, y_train, X_test, y_test):
    t = process_time()
    print("------------- ", name, " --------------")
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print("RMSPE: %.6f" % rmspe(y_predict, y_test))
    # print("feature importance: " + np.array_str(clf.feature_importances_))
    print('train and evaluate model in ', process_time() - t)


def submission(clf, train, test, feature_names):
    t = process_time()
    clf.fit(train[feature_names], train.Sales.values)
    y_test = clf.predict(test[feature_names])
    result = pd.DataFrame({'Id': test.astype(int).Id.values, 'Sales': y_test}).set_index('Id')
    result = result.sort_index()
    result.to_csv('submission.csv')
    result.to_csv('submission.excel.csv', sep=";", decimal=",")
    print('submission created in ', process_time() - t)
    
    
train, test, feature_names = get_sets()
X_train, X_test, y_train, y_test = train_test_split(train[feature_names], train.Sales.values, test_size=0.10, random_state=2)

clf = make_pipeline(StandardScaler(), SGDRegressor(loss='squared_loss', penalty='l2'))
score("SGDRegressor", clf, X_train, y_train, X_test, y_test)
kfold_validation("SGDRegressor", clf, X_train, y_train)
submission(clf, train, test, feature_names)

clf = RandomForestRegressor(n_jobs=-1, n_estimators=25)
score("RandomForestRegressor", clf, X_train, y_train, X_test, y_test)
kfold_validation("RandomForestRegressor", clf, X_train, y_train)

# clf = GradientBoostingRegressor()
# score("GradientBoostingRegressor", clf, X_train, y_train, X_test, y_test)
# kfold_validation("GradientBoostingRegressor", clf, X_train, y_train)
