# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import xgboost as xgb

SEED = 42
np.random.seed(SEED)

interest_level = ['high', 'medium', 'low']
n_class = len(interest_level)
interest_level_dict = {w:i for i, w in enumerate(interest_level)}
cat_feats = ['display_address', 'manager_id', 'building_id', 'street_address']
encoders = {}
tfidf = TfidfVectorizer(stop_words='english', max_features=250)

def prepare_encoders():
    test_df = pd.read_json(open('../input/test.json'))
    train_df = pd.read_json(open('../input/train.json'))
    for name in cat_feats:
        lbl = preprocessing.LabelEncoder()
        encoders[name] = lbl.fit(list(train_df[name].values) + 
                                 list(test_df[name].values))
                                 
    tfidf_col = pd.concat([train_df['features'].apply(lambda x: ' '.join(x)), 
                           test_df['features'].apply(lambda x: ' '.join(x))])
    tfidf.fit(tfidf_col)
     
    

def load_data(datatype='test'):

    df = pd.read_json(open('../input/{}.json'.format(datatype)))

    df['num_photos'] = df['photos'].apply(len)
    df['num_features'] = df['features'].apply(len)
    df['num_description_words'] = df['description'].apply(lambda x: len(x.split(' ')))

    desc_feats = {'bathroom_mentions': ['bathroom', 'bthrm', 'ba '],
                  'bedroom_mentions': ['bedroom', 'bdrm', 'br '],
                  'kitchen_mentions': ['kitchen', 'kit ']}

    # Add decsription features
    for name, kwords in desc_feats.items():
        df[name] =  df['description'].apply(lambda x: sum([x.count(w) for w in kwords]))

    # Add time features
    df['created'] = pd.to_datetime(df['created'])
    df['created_year'] = df['created'].dt.year
    df['created_month'] = df['created'].dt.month
    df['created_day'] = df['created'].dt.day
    df['created_hour'] = df['created'].dt.hour

    num_feats = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                 'num_photos', 'num_features', 'num_description_words',
                 'bathroom_mentions', 'bedroom_mentions', 'kitchen_mentions',
                 'created_year', 'created_month', 'created_day', 'created_hour']
                 
    # Add categorical features
    for name in cat_feats:
        df[name] = encoders[name].transform(list(df[name].values))
    
    X = np.array(df[num_feats + cat_feats]).astype('float32')
    
    # Add features from 'features' column. 250 tfidf features
    # We use scipy.sparse because tfidf.transform return a sparse object
    X = sparse.hstack([X,  tfidf.transform(df['features'].apply(lambda x: ' '.join(x)))]).tocsr()
    
    y = np.array(df['interest_level'].replace(interest_level_dict)).astype('int8') if datatype == 'train' else None
    ids = df['listing_id'].values
    
    print('{} shape: {}'.format(datatype, X.shape))

    return X, y, ids


def train_xgb(X, y, params):
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    #xg_train = xgb.DMatrix(X_train, label=y_train)
    #xg_val = xgb.DMatrix(X_val, label=y_val)
    #watchlist  = [(xg_val,'eval'), (xg_train,'train')]
    #clr = xgb.train(params, xg_train, params['num_rounds'], watchlist)
    xg_train = xgb.DMatrix(X, label=y)
    clr = xgb.train(params, xg_train, params['num_rounds'])

    return clr

def predict_xgb(clr, X_test):
    xg_test = xgb.DMatrix(X_test)
    preds = clr.predict(xg_test)

    return preds


def main():
    prepare_encoders()
    X, y, ids = load_data(datatype='train')

    params = {}
    params['objective'] = 'multi:softprob'
    params['eval_metric'] = 'mlogloss'
    params['num_class'] = n_class
    params['eta'] = 0.07
    params['max_depth'] = 6
    params['subsample'] = 0.8
    params['colsample_bytree'] = 0.8
    params['min_child_weight'] = 1
    params['silent'] = 1
    params['num_rounds'] = 500
    params['seed'] = SEED

    clr = train_xgb(X, y, params)
    X, y, ids = load_data(datatype='test')
    preds = predict_xgb(clr, X)

    with open('submission.csv', 'w') as wf:
        wf.write('listing_id,{}\n'.format(','.join(interest_level)))
        for i, pred in enumerate(preds):
            wf.write('{},{}\n'.format(ids[i], ','.join(map(str, pred))))

if __name__ == '__main__':
    main()