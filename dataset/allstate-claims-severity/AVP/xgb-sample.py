
"""
Based on Vladimir Iglovikov' script 
https://www.kaggle.com/iglovikov/allstate-claims-severity/xgb-1114/discussion
"""
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import mean_absolute_error

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#################################
# Modificación de columnas:
def modificaColumnas(matriz):
    
    matriz.cat73 = matriz.cat73.replace('C','A')
    matriz.cat76 = matriz.cat76.replace('C','A')
    matriz.cat79 = matriz.cat79.replace('A','B')
    matriz.cat79 = matriz.cat79.replace('C','D')
    matriz.cat80 = matriz.cat80.replace('B','C')
    matriz.cat80 = matriz.cat80.replace('A','D')
    matriz.cat86 = matriz.cat86.replace('D','C')
    matriz.cat86 = matriz.cat86.replace('B','C')
    matriz.cat87 = matriz.cat87.replace('A','B')
    matriz.cat87 = matriz.cat87.replace('C','D')
    matriz.cat89 = matriz.cat89.replace('C','B')
    matriz.cat89 = matriz.cat89.replace('D','B')
    matriz.cat89 = matriz.cat89.replace('H','B')
    matriz.cat89 = matriz.cat89.replace('I','B')
    matriz.cat89 = matriz.cat89.replace('G','B')
    matriz.cat89 = matriz.cat89.replace('E','A')
    matriz.cat90 = matriz.cat90.apply(lambda x: 'C' if not(x =='A') else 'A')
    matriz.cat91 = matriz.cat91.apply(lambda x: 'A' if (x in ['A','B','G']) else 'C')
    matriz.cat100 = matriz.cat100.apply(lambda x: 'O' if (x in ['F','N','K']) else x)
    matriz.cat100 = matriz.cat100.apply(lambda x: 'J' if (x in ['I','A']) else x)
    matriz.cat100 = matriz.cat100.apply(lambda x: 'M' if (x in ['L']) else x)
    matriz.cat100 = matriz.cat100.apply(lambda x: 'A' if not(x in ['O','J','M']) else x)
    matriz.cat101 = matriz.cat101.apply(lambda x: 'SS' if (x in ['K','U','B','H','N','E','R','S']) else x)
    matriz.cat102 = matriz.cat102.apply(lambda x: 'C' if not(x =='A') else 'A')
    matriz.cat103 = matriz.cat103.apply(lambda x: 'S' if (x in ['N','J','K','L','I','H','G','F']) else x)
    matriz.cat114 = matriz.cat114.apply(lambda x: 'C' if (x in ['E','B','D']) else x)
    matriz.cat114 = matriz.cat114.apply(lambda x: 'N' if (x in ['I','W','G']) else x)
    matriz.cat114 = matriz.cat114.apply(lambda x: 'S' if (x in ['V','Q','X']) else x)
    matriz.cat114 = matriz.cat114.apply(lambda x: 'R' if (x in ['L','U','O']) else x)
    matriz.cat114 = matriz.cat114.apply(lambda x: 'F' if (x in ['J']) else x)

    matriz2 = matriz
    
    return matriz2

# In[0]: SPLIT DATASET AND SCALE VARIABLES:
train = modificaColumnas(train)
test = modificaColumnas(test)
print('########################')
#################################

test['loss'] = np.nan
joined = pd.concat([train, test])
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con =2
    x =preds-labels
    grad =con*x / (np.abs(x)+con)
    hess =con**2 / (np.abs(x)+con)**2
    return grad, hess 


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))

if __name__ == '__main__':
    for column in list(train.select_dtypes(include=['object']).columns):
        if train[column].nunique() != test[column].nunique():
            set_train = set(train[column].unique())
            set_test = set(test[column].unique())
            remove_train = set_train - set_test
            remove_test = set_test - set_train

            remove = remove_train.union(remove_test)
            def filter_cat(x):
                if x in remove:
                    return np.nan
                return x

            joined[column] = joined[column].apply(lambda x: filter_cat(x), 1)
            
        joined[column] = pd.factorize(joined[column].values, sort=True)[0]

    train = joined[joined['loss'].notnull()]
    test = joined[joined['loss'].isnull()]

    shift = 200
    y = np.log(train['loss'] + shift)
    ids = test['id']
    X = train.drop(['loss', 'id'], 1)
    X_test = test.drop(['loss', 'id'], 1)

    from sklearn.model_selection import train_test_split
    x_train, x_valid, y_train, y_valid =train_test_split(X, y, test_size=0.2, random_state=2016)
    RANDOM_STATE = 2016
    params = {
        'min_child_weight': 1,
        'eta': 0.03,
        'colsample_bytree': 0.5,
        'max_depth': 5,
        'subsample': 0.9,
        'alpha': 1,
        'gamma': 1.5,
        'silent': 1,
        'verbose_eval': True,
        'seed': RANDOM_STATE,
        'nthread':7,
        'base_score':7.76
    }

    xgtrain = xgb.DMatrix(x_train, label=y_train)
    xgval = xgb.DMatrix(x_valid, label=y_valid)
    xgtest = xgb.DMatrix(X_test)
    watchlist = [ (xgtrain,'train'),(xgval,'eval')]
    model = xgb.train(params, xgtrain,5000,watchlist,obj=logregobj,feval=evalerror,early_stopping_rounds=10)

    prediction = np.exp(model.predict(xgtest)) - shift

    submission = pd.DataFrame()
    submission['loss'] = prediction
    submission['id'] = ids
    submission.to_csv('sub_fair_obj.csv', index=False)