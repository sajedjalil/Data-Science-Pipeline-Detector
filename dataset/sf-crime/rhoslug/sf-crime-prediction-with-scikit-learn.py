from __future__ import print_function, division
import numpy as np

import pandas as pd

from patsy import dmatrices, dmatrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def main():
    # Read in the training and testing data
    df_train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])
    df_train.drop(['Descript', 'Dates', 'Resolution'], axis=1, inplace=True)
    df_test = pd.read_csv('../input/test.csv', parse_dates=['Dates'])
    df_test.drop(['Dates'], axis=1, inplace=True)

    # Select training and validation sets
    inds = np.arange(df_train.shape[0])
    np.random.shuffle(inds)
    train_inds = inds[:int(0.2 * df_train.shape[0])]
    val_inds = inds[int(0.2 * df_train.shape[0]):]

    # Extract the column names
    col_names = np.sort(df_train['Category'].unique())

    # Recode categories to numerical
    df_train['Category'] = pd.Categorical.from_array(df_train['Category']).codes
    df_train['DayOfWeek'] = pd.Categorical.from_array(df_train['DayOfWeek']).codes
    df_train['PdDistrict'] = pd.Categorical.from_array(df_train['PdDistrict']).codes
    df_test['DayOfWeek'] = pd.Categorical.from_array(df_test['DayOfWeek']).codes
    df_test['PdDistrict'] = pd.Categorical.from_array(df_test['PdDistrict']).codes
    
    # Extract the text frequencies for use
    cvec = CountVectorizer()
    bows_train = cvec.fit_transform(df_train['Address'].values)
    bows_test = cvec.transform(df_test['Address'].values)

    # Split up the training and validation sets
    df_val = df_train.ix[val_inds]
    df_train = df_train.ix[train_inds]

    # Construct the design matrix and response vector for the
    # training data and the design matrix for the test data
    y_train, X_train = dmatrices('Category ~ X + Y + DayOfWeek + PdDistrict', df_train)
    X_train = np.hstack((X_train, bows_train[train_inds, :].toarray()))
    y_val, X_val = dmatrices('Category ~ X + Y + DayOfWeek + PdDistrict', df_val)
    X_val = np.hstack((X_val, bows_train[val_inds, :].toarray()))
    X_test = dmatrix('X + Y + DayOfWeek + PdDistrict', df_test)
    X_test = np.hstack((X_test, bows_test.toarray()))
    
    # Use PCA to reduce the dimensionality
    ipca = IncrementalPCA(n_components=4, batch_size=10)
    X_train = ipca.fit_transform(X_train)
    X_val = ipca.transform(X_val)
    X_test = ipca.transform(X_test)

    # Create the logistic classifier and fit it
    logistic = LogisticRegression()
    logistic.fit(X_train, y_train.ravel())
    print('Mean accuracy (Logistic): {:.4f}'.format(logistic.score(X_val, y_val.ravel())))

    # Fit the random forest
    randforest = RandomForestClassifier(n_estimators=11)
    randforest.fit(X_train, y_train.ravel())
    print('Mean accuracy (Random Forest): {:.4f}'.format(randforest.score(X_val, y_val.ravel())))
    #
    # # Make predictions
    predict_probs = logistic.predict_proba(X_test)
    
    # Add the id numbers for the incidents and construct the final df
    df_pred = pd.DataFrame(data=predict_probs, columns=col_names)
    df_pred['Id'] = df_test['Id'].astype(int)
    df_pred.to_csv('output.csv', index=False)

if __name__ == '__main__':
    main()
