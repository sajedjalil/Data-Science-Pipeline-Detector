import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import time
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
import gc

NR_MAX_TEXT_FEATURES = 55000
NR_TOP_BRANDS = 4000
NR_TOP_CATEGORIES = 1000
NR_DISCOVER_BRANDS = 50
NAME_MIN_DF = 10

# results
# ? () added item_condition_id
# 0.57362 (55000 text features) added count vectorized product name
# 0.66818 (55000 text features) combined sparse matrix
# 0.71877 (55000 text features)
# 0.71940 (5000 text features)
# 0.70099 (1000 text features)
# 0.71975 (1 text feature)
# 0.68758

### feature engineering section

# 1. feature engineering: convert category to single subcategory strings
def label_category (row, index):
    if 'category_name' in row:
        cats = str(row['category_name']).split('/')
    if len(cats) > index:
        return cats[index]

# 2. feature engineering: positive words
positiveWords = ['brand new', 'great condition', 'excellent condition','flawless condition', 'authentic', 'factory sealed']
def label_positive_wording (row):
    if any(s in str(row['item_description']).lower() for s in positiveWords):
        return 1
    if any(s in str(row['name']).lower() for s in positiveWords):
        return 1
    return 0

def find_brand (row, top_brands):
    if pd.isnull(row['brand_name']):
        for brand in top_brands:
            if brand in str(row['item_description']):
                return brand
        return row['brand_name']

# evaluation
def eval_error (d):
    sum = 0.0;
    for index, row in d.iterrows():
        sum = sum + (math.log(row['price']+1) - math.log(row['real_price']+1))**2
    return math.sqrt(sum / (index + 1))

def main():
    start_time = time.time()

    ### Read data
    train = pd.read_csv('../input/train.tsv', sep='\t')
    test = pd.read_csv('../input/test.tsv', sep='\t')
    result = test[['test_id']].copy()
    result['test_id'] = result['test_id'].astype(int) 
    # delete rows where price is zero
    train = train[train['price'] > 0]
    train['price'].dropna()
    y = np.log1p(train['price']) # get the result to fit for
    # merge train and test for feature engineering process
    nrow_train = train.shape[0]
    data = pd.concat([train, test])
    # free import data structures again and run GC
    del train
    del test
    gc.collect()
    
    print("[%d] Finished to read data" % int(time.time() - start_time))
    # calc top brands
    top_brands = data['brand_name'].value_counts().index[:NR_DISCOVER_BRANDS]
    # discover missing brands
    data['brand_name'] = data.apply (lambda row: find_brand (row, top_brands),axis=1)
    print("[%d] Finished to identify missing brands" % int(time.time() - start_time))
    # fill empty values
    data['category_name'].fillna(value='none', inplace=True)
    data['brand_name'].fillna(value='none', inplace=True)
    data['item_description'].fillna(value='none', inplace=True)
    print("[%d] Finished to fill missing values" % int(time.time() - start_time))
    # cap categories to top N
    top_categorys = data['category_name'].value_counts().loc[lambda x: x.index != 'none'].index[:NR_TOP_CATEGORIES]
    data.loc[~data['category_name'].isin(top_categorys), 'category_name'] = 'none'
    print("[%d] Finished to categorize categories" % int(time.time() - start_time))
    # cap brands to top N
    top_brands = data['brand_name'].value_counts().loc[lambda x: x.index != 'none'].index[:NR_TOP_BRANDS]
    data.loc[~data['brand_name'].isin(top_brands), 'brand_name'] = 'none'
    print("[%d] Finished to cap brands" % int(time.time() - start_time))
    # convert brands to numeric values
    data['brand_i'] = data.brand_name.astype("category").cat.codes
    print("[%d] Finished to categorize brands" % int(time.time() - start_time))
    # convert categories to numeric values
    data['cat_i'] = data.category_name.astype("category").cat.codes
    
    
    # identify positive words
    data['pos_words'] = data.apply (lambda row: label_positive_wording (row),axis=1)
    print("[%d] Finished to identify positive words" % int(time.time() - start_time))

    tv = TfidfVectorizer(max_features=NR_MAX_TEXT_FEATURES,
                         ngram_range=(1, 3),
                         stop_words='english')
                         
    text_features = tv.fit_transform(data['item_description'])
    print("[%d] Finished transform product description" % int(time.time() - start_time))
    
    cv = CountVectorizer(min_df=NAME_MIN_DF)
    X_name = cv.fit_transform(data['name'])
    print("[%d] Finished vectorize product name" % int(time.time() - start_time))

    # convert to sparse matrix
    other_features = data.as_matrix(columns=['pos_words', 'brand_i', 'cat_i', 'shipping', 'item_condition_id'])
    sparse_matrix = hstack((other_features, X_name, text_features)).tocsr()
    print("[%d] Finished to stack and create sparse matrix" % int(time.time() - start_time))
    
    # split train from test rows
    X = sparse_matrix[:nrow_train]
    X_test = sparse_matrix[nrow_train:]
    print("[%d] Finished to split sparse matrix into train and test" % int(time.time() - start_time))

    # learn a model
    #model = RandomForestRegressor()
    model = linear_model.Ridge(solver="sag")
    #model.fit(train.as_matrix(columns=feature_columns), y)
    model.fit(X, y)
    print("[%d] Finished to training the model" % int(time.time() - start_time))
    predictions = model.predict(X_test)
    print("[%d] Finished to predict result" % int(time.time() - start_time))
    # write result
    result['price'] = np.expm1(predictions)
    result.to_csv('submission.csv', encoding='utf-8', index=False)
    print("[%d] Finished to store result" % int(time.time() - start_time))
    
if __name__ == '__main__':
    main()