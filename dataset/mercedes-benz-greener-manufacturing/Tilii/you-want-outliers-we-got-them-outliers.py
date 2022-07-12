__author__ = 'Tilii: https://kaggle.com/tilii7'

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore',category=DeprecationWarning)
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import cross_val_predict
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.manifold import TSNE
    import pprint

# from https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
def is_outlier(points, thresh=3.5):
    '''
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), 'Volume 16: How to Detect and
        Handle Outliers', The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    '''
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return (modified_z_score, (modified_z_score > thresh) )

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: %i minutes and %s seconds.' % (tmin, round(tsec,2)))

if __name__ == '__main__':

    RFR = RandomForestRegressor(n_estimators=100)
    tsne = TSNE(n_components=2, n_iter_without_progress=50, init='pca', verbose=2, random_state=1001)

# Load data set and target values
    start_time = timer(None)
    print('\n# Reading and Processing Data')
    train = pd.read_csv('../input/train.csv', dtype={'ID': np.int32, 'y': np.float32})
    target = train['y'].values
    train_ids = train['ID'].values
    train = train.drop(['ID', 'y'], axis=1)
    print('\n Initial Train Set Matrix Dimensions: %d x %d' % (train.shape[0], train.shape[1]))
    train_len = len(train)
    test = pd.read_csv('../input/test.csv', dtype={'ID': np.int32})
    test_ids = test['ID'].values
    test = test.drop(['ID'], axis=1)
    print('\n Initial Test Set Matrix Dimensions: %d x %d' % (test.shape[0], test.shape[1]))

# Sort out numerical and categorical features
    all_data = pd.concat((train, test))
    numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
    categorical_feats = all_data.dtypes[all_data.dtypes == 'object'].index

    print('\n Converting categorical features:')
    for i, col_name in enumerate(categorical_feats):
        print(' Converting %s' % col_name)
        temp_df = pd.get_dummies(all_data[col_name])
        new_features = temp_df.columns.tolist()
        new_features = [col_name + '_' + w for w in new_features]
        temp_df.columns = new_features
        all_data.drop(col_name, axis=1, inplace=True)
        all_data = pd.concat((all_data, temp_df), axis=1)

# Remove columns where all data points have the same value
    print('\n Number of columns before cleaning: %d' % len(all_data.columns))
    cols = all_data.columns.tolist()
    for column in cols:
        if len(np.unique(all_data[column])) == 1:
            print(' Column %s removed' % str(column))
            all_data.drop(column, axis=1, inplace=True)

# Remove identical columns where all data points have the same value
    cols = all_data.columns.tolist()
    remove = []
    for i in range(len(cols)-1):
        v = all_data[cols[i]].values
        for j in range(i+1,len(cols)):
            if np.array_equal(v,all_data[cols[j]].values):
                remove.append(cols[j])
                print(' Column %s is identical to %s. Removing %s' % (str(cols[i]), str(cols[j]), str(cols[j])))

    all_data.drop(remove, axis=1, inplace=True)
    print('\n Number of columns after cleaning: %d' % len(all_data.columns))

    features = all_data.columns
    print('\n Final Matrix Dimensions: %d x %d' % (all_data.shape[0], all_data.shape[1]))
    train_data = pd.DataFrame(all_data[ : train_len].values, columns=features)
    test_data = pd.DataFrame(all_data[train_len : ].values, columns=features)
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    timer(start_time)

    start_time = timer(None)
    print('\n Calculating t-SNE embedding:')
    all_data_tsne = tsne.fit_transform(all_data)
    train_data_tsne = pd.DataFrame(all_data_tsne[ : train_len], columns=['tsne_x','tsne_y'])
    test_data_tsne = pd.DataFrame(all_data_tsne[train_len : ], columns=['tsne_x','tsne_y'])
    train_data_tsne.reset_index(drop=True, inplace=True)
    test_data_tsne.reset_index(drop=True, inplace=True)
    timer(start_time)

# Running isolation forest to remove outliers
    start_time = timer(None)
    clf = IsolationForest(n_estimators=500, max_samples=1.0, random_state=1001, bootstrap=True, contamination=0.02, verbose=0, n_jobs=-1)
    print('\n Running Isolation Forest:')
    clf.fit(train_data.values, target)
    isof = clf.predict(train_data.values)
    train.insert(0, 'y', target)
    train.insert(0, 'ID', train_ids)
    train['isof'] = isof
    myindex = train['isof'] < 0
    train_IF = train.loc[myindex]
    train_IF.reset_index(drop=True, inplace=True)
    train_IF.drop('isof', axis=1, inplace=True)
    train_IF.to_csv('train-isof-outliers.csv', index=False)
    test.insert(0, 'ID', test_ids)
    test['isof'] = clf.predict(test_data.values)
    myindex = test['isof'] < 0
    test_IF = test.loc[myindex]
    test_IF.reset_index(drop=True, inplace=True)
    test_IF.drop('isof', axis=1, inplace=True)
    test_IF.to_csv('test-isof-outliers.csv', index=False)
    print('\n Found %d outlier points' % len(train_IF))
    timer(start_time)

    start_time = timer(None)
    threshold = 2.0
    print('\n Running Random Forest Regressor (10-fold):')
    target_pred = cross_val_predict(estimator=RFR, X=train_data.values, y=target, cv=10, n_jobs=-1)
    rfr_pred = pd.DataFrame({'ID': train_ids, 'y': target, 'y_pred': target_pred})
    rfr_pred.to_csv('prediction-train-oof-10fold-RFR.csv', index=False)
    yvalues = np.vstack((target, target_pred)).transpose()
    OL_score, OL = is_outlier(yvalues, threshold)
    train['outlier_score'] = OL_score
    myindex = train['outlier_score'] >= threshold
    train_OL = train.loc[myindex]
    train_OL.reset_index(drop=True, inplace=True)
    train_OL.drop(['isof','outlier_score'], axis=1, inplace=True)
    train_OL.to_csv('train-outliers.csv', index=False)
    timer(start_time)

    start_time = timer(None)
    train_outliers_tsne = train_data_tsne.loc[myindex]
    test_outliers_tsne = test_data_tsne.values
    outlier_list = []
    for k in range(len(train_outliers_tsne)):
        d = ((test_outliers_tsne-train_outliers_tsne.values[k])**2).sum(axis=1)  # compute distances
        ndx = d.argsort() # sort so that smallest distance is first
        print(' Presumed outlier point for train ID = %d is test ID = %d ; their Euclidean distance from t-SNE embedding is %.8f' % (train_OL.iloc[k]['ID'], test.iloc[ndx[0]]['ID'], d[ndx[0]]))
        outlier_list.append(ndx[0])
        print(' Ten closest test points (ID, distance):')
        pprint.pprint(zip(test.iloc[ndx[:10]]['ID'], d[ndx[:10]]))

    test_OL = test.iloc[outlier_list]
    test_OL.drop(['isof'], axis=1, inplace=True)
    test_OL.sort_values(['ID'], inplace=True)
    test_OL.reset_index(drop=True, inplace=True)
    test_OL.to_csv('test-outliers.csv', index=False)

    timer(start_time)
