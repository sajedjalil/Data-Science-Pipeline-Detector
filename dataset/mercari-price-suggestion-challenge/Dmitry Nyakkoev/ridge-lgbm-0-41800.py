import re
import gc
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import lightgbm as lgb

start_time = datetime.now()
print(start_time)

train = pd.read_table('../input/train.tsv', engine='c')
test = pd.read_table('../input/test.tsv', engine='c')

print('[{}] Finished to load data'.format( datetime.now() - start_time))


#submission: pd.DataFrame = test[['test_id']]

# DROP 
#dftt = train[(train.price < 1.0)]
train = train.drop(train[(train.price < 2)].index)
#del dftt['price']
row_train = train.shape[0]
#row_train2 = train2.shape[0]
train['train_id'] =np.arange(row_train)
train.index =  pd.RangeIndex(row_train)
# END

y_train = np.log1p(train['price'])
row_train = train.shape[0]

merge = pd.concat([train, test])

del train
gc.collect()


merge['category_name'] = merge['category_name'].fillna('Other').astype(str)
merge['brand_name'] = merge['brand_name'].fillna('missing').astype(str)
merge['shipping'] = merge['shipping'].astype(str)
merge['item_condition_id'] = merge['item_condition_id'].astype(str)
merge['item_description'] = merge['item_description'].fillna('None')


# Manipulation with DataFrame
start_manip = datetime.now()
merge['nameL'] = (merge['name'].str.len()).astype(str)
merge['item_descriptionL'] = (merge['item_description'].str.len()).astype(str)

merge['nameWL'] = (merge['name'].str.split().str.len() )
merge['nameWL'] = merge['nameWL'].fillna(0)
merge['nameWL'] = merge['nameWL'].astype(str)

merge['item_descriptionWL'] = (merge['item_description'].str.split().str.len() )
merge['item_descriptionWL'] = merge['item_descriptionWL'].fillna(0)
merge['item_descriptionWL'] = merge['item_descriptionWL'].astype(str)



print('[{}] Finish to Manipulation. Total time {}'.format( datetime.now() - start_manip, datetime.now() - start_time ))

# FINDING BRAND
start_brfind = datetime.now()
'''
all_brands = set(merge['brand_name'].values)
all_brands.remove('missing')
premissing = len(merge.loc[merge['brand_name'] == 'missing'])
def brandfinder(line):
    brand = line[0]
    name = line[1]
    descript = line[2]
    namesplit = name.split(' ')
    descsplit = descript.split(' ')
    if brand == 'missing':
        for x in namesplit:
            if x in all_brands:
                #print(x)
                return name
        #    else:
        #        if x in descsplit:
                   # print(x)
         #           return name
    if name in all_brands:
        return name
    return brand
merge['brand_name'] = merge[['brand_name','name','item_description']].apply(brandfinder, axis = 1)
#merge['brand_name'] = merge[['brand_name','name']].apply(brandfinder, axis = 1)
found = premissing-len(merge.loc[merge['brand_name'] == 'missing'])

print(found)
'''
###############################################################
all_cats = set(merge['category_name'].values)
print('lenCats {}'.format(len(all_cats)))
trainca = merge.groupby('category_name', as_index=False)['price'].median()
trainca.columns = ['category_name', 'cn']
merge=pd.merge(merge, trainca, how = 'left')

all_brs = set(merge['brand_name'].values)
print('lenBrands {}'.format(len(all_brs)))
trainbr = merge.groupby('brand_name', as_index=False)['price'].median()
trainbr.columns = ['brand_name', 'bm']
merge=pd.merge(merge,trainbr,how = 'left')

trainbr = merge.groupby('brand_name', as_index=False)['price'].max()
trainbr.columns = ['brand_name', 'bmmax']
merge=pd.merge(merge,trainbr,how = 'left')

trainca = merge.groupby('category_name', as_index=False)['price'].max()
trainca.columns = ['category_name', 'cnmax']
merge=pd.merge(merge, trainca, how = 'left')

trainbr = merge.groupby('brand_name', as_index=False)['price'].min()
trainbr.columns = ['brand_name', 'bmmin']
merge=pd.merge(merge,trainbr,how = 'left')

trainca = merge.groupby('category_name', as_index=False)['price'].min()
trainca.columns = ['category_name', 'cnmin']
merge=pd.merge(merge, trainca, how = 'left')




#merge['cnbm_max'] = merge[['bmmax','cnmax']].min(axis=1)
merge['cnbm_min'] = merge[['bmmin','cnmin']].max(axis=1)
#merge['cnbm'] =  merge['cnbm_max']/ merge['cnbm_min']

####merge['cnbm'] = merge['bm'] + merge['cn']
merge['cnbm'] = merge[['cn','bm']].min(axis=1)
merge['cnbm_max'] = merge['bmmax'] + merge['cnmax']
#merge['cnbm_min'] = merge['bmmin'] + merge['cnmin']
#merge['cnbm_min'] = merge[['bmmin','cnmin']].max(axis=1)
merge['maxs'] =  (merge['bmmax']+ merge['bmmin'] + merge['bm'])/ (merge['cnmax'] + merge['cnmin'] + merge['cn'])

merge['cnbm_max'] = merge['cnbm_max'].astype(str)
merge['cnbm_min'] = merge['cnbm_min'].astype(str)
merge['maxs'] = merge['maxs'].astype(str)


merge['bmcn'] = merge['bm'] - merge['cn']
merge['bmcn'] = merge['bmcn'].astype(str)
merge['cnbm'] = merge['cnbm'].astype(str)

####################################################################

print('[{}] Finish to Brand Finding. Total time {}'.format( datetime.now() - start_brfind, datetime.now() - start_time ))

def split_cat(text):
    clear_arr = ['Other','Other','Other']
    try: 
        arrtext = text.split("/")
        clear_arr[:len(arrtext)] = arrtext
        return clear_arr
    except: 
        return ("Other", "Other", "Other")

#merge['subcat_1'], merge['subcat_2'], merge['subcat_3'] = zip(*merge['category_name'].apply(lambda x: split_cat(x)))

def repack(word):
    word =  word.replace('purpley','purple')
    word=re.sub(r"(\d+)([a-zA-Z])", " \\1 \\2", word)
    return word

def build_preprocessor(field,arr):
    if field == ('name'):
        print(field)
        field_idx = list(arr.columns).index(field)
        return lambda x: ((x[field_idx]).lower()).replace('\'','')
    elif field == ('nameL'):
        print(field)
        field_idx = list(arr.columns).index(field)
        return lambda x: x[field_idx]
    elif field == ('item_descriptionL'):
        print(field)
        field_idx = list(arr.columns).index(field)
        return lambda x: x[field_idx]
    else:
        field_idx = list(arr.columns).index(field)
        return lambda x: (x[field_idx]).lower()

MAX_FEATURES_NM = 920000
MAX_FEATURES_ITEM_DESCR  = 2050000 #row_train  * 2

def vector(arr):
    vectorizer = FeatureUnion([
        ('name', CountVectorizer(
            ngram_range=(1, 2),
            max_features=MAX_FEATURES_NM,
            preprocessor=build_preprocessor('name',arr))),
        ('nameD', CountVectorizer(
            token_pattern='\d+',
            preprocessor=build_preprocessor('name',arr))),
        ('nameC', TfidfVectorizer(
            token_pattern='[a-zA-Z]+',
            max_features=700000,
            preprocessor=build_preprocessor('name',arr))),
        ('nameL', TfidfVectorizer(
           # token_pattern='[a-zA-Z]+',
            max_features=1000,
            preprocessor=build_preprocessor('nameL',arr))),
        ('nameWL', TfidfVectorizer(
            token_pattern='\d+',
            max_features=100,
            preprocessor=build_preprocessor('nameWL',arr))),
        ('category_name', CountVectorizer(
            token_pattern='.+',
            preprocessor=build_preprocessor('category_name',arr))),
        ('brand_name', TfidfVectorizer(
            token_pattern='.+',
            preprocessor=build_preprocessor('brand_name',arr))),
        ('bmcn', TfidfVectorizer(
            token_pattern='.+',
            preprocessor=build_preprocessor('bmcn',arr))),
        ('cnbm_max', TfidfVectorizer(
            token_pattern='.+',
            preprocessor=build_preprocessor('cnbm_max',arr))),
        ('cnbm_min', TfidfVectorizer(
            token_pattern='.+',
            preprocessor=build_preprocessor('cnbm_min',arr))),
        ('maxs', TfidfVectorizer(
            token_pattern='.+',
            preprocessor=build_preprocessor('maxs',arr))),
  #      ('mins', TfidfVectorizer(
  #          token_pattern='.+',
  #          preprocessor=build_preprocessor('mins',arr))),
            
        ('shipping', CountVectorizer(
            token_pattern='\d+',
            preprocessor=build_preprocessor('shipping',arr))),
        ('item_condition_id', CountVectorizer(
            token_pattern='\d+',
            preprocessor=build_preprocessor('item_condition_id',arr))),
        ('item_description', TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=MAX_FEATURES_ITEM_DESCR,
            preprocessor=build_preprocessor('item_description',arr))),
        ('item_descriptionD', TfidfVectorizer(
            token_pattern='\d+',
            max_features=100000,
            preprocessor=build_preprocessor('item_description',arr))),
        ('item_descriptionC', TfidfVectorizer(
            token_pattern='[a-zA-Z]+',
            max_features=MAX_FEATURES_ITEM_DESCR,
            preprocessor=build_preprocessor('item_description',arr))),
        ('item_descriptionL', TfidfVectorizer(
           # token_pattern='[a-zA-Z]+',
            max_features=1000,
            preprocessor=build_preprocessor('item_descriptionL',arr))),
        ('item_descriptionWL', TfidfVectorizer(
            token_pattern='\d+',
            max_features=100,
            preprocessor=build_preprocessor('item_descriptionWL',arr)))
    ])
    return vectorizer.fit_transform(arr.values)
    



X_merge = vector(merge)

X_train = X_merge[:row_train]
X_test = X_merge[row_train:]

print('merge {}    X_train {}'.format(X_merge.shape, X_train.shape))
print('[{}] Finished to vectorizer data'.format( datetime.now() - start_time ))


model = Ridge(
        solver='auto',
        fit_intercept=True,
         alpha = 5.2,
       # alpha=4.5,
        max_iter= 100,
        normalize=False
        )
        
model.fit(X_train, y_train)
predR = model.predict(X_test)


MAX_FEATURES_NM = 10000
MAX_FEATURES_ITEM_DESCR  = MAX_FEATURES_NM * 2 
del X_train
del X_test
del X_merge
gc.collect()

def vector2(arr):
    vectorizer = FeatureUnion([
    #    ('name', CountVectorizer(
    #        ngram_range=(1, 2),
    #        max_features=MAX_FEATURES_NM,
    #        preprocessor=build_preprocessor('name',arr))),
        ('nameD', CountVectorizer(
            token_pattern='\d+',
            preprocessor=build_preprocessor('name',arr))),
        ('nameC', TfidfVectorizer(
            token_pattern='[a-zA-Z]+',
            max_features=MAX_FEATURES_NM,
            preprocessor=build_preprocessor('name',arr))),
        ('nameL', TfidfVectorizer(
           # token_pattern='[a-zA-Z]+',
            max_features=1000,
            preprocessor=build_preprocessor('nameL',arr))),
        ('nameWL', TfidfVectorizer(
            token_pattern='\d+',
            max_features=100,
            preprocessor=build_preprocessor('nameWL',arr))),
        ('category_name', CountVectorizer(
            token_pattern='.+',
            max_features=1000,
            preprocessor=build_preprocessor('category_name',arr))),
        ('brand_name', TfidfVectorizer(
            token_pattern='.+',
            max_features=5004,
            preprocessor=build_preprocessor('brand_name',arr))),
        ('bmcn', TfidfVectorizer(
            token_pattern='.+',
            preprocessor=build_preprocessor('bmcn',arr))),
   #     ('cnbm', TfidfVectorizer(
   #         token_pattern='.+',
  #          preprocessor=build_preprocessor('cnbm',arr))),
        ('cnbm_max', TfidfVectorizer(
            token_pattern='.+',
            preprocessor=build_preprocessor('cnbm_max',arr))),
        ('cnbm_min', TfidfVectorizer(
            token_pattern='.+',
            preprocessor=build_preprocessor('cnbm_min',arr))),
        ('maxs', TfidfVectorizer(
            token_pattern='.+',
            preprocessor=build_preprocessor('maxs',arr))),
        ('shipping', CountVectorizer(
            token_pattern='\d+',
            preprocessor=build_preprocessor('shipping',arr))),
        ('item_condition_id', CountVectorizer(
            token_pattern='\d+',
            preprocessor=build_preprocessor('item_condition_id',arr))),
        ('item_descriptionD', TfidfVectorizer(
            token_pattern='\d+',
            max_features=1000,
            preprocessor=build_preprocessor('item_description',arr))),
        ('item_descriptionC', TfidfVectorizer(
            token_pattern='[a-zA-Z]+',
            max_features=MAX_FEATURES_ITEM_DESCR,
            preprocessor=build_preprocessor('item_description',arr))),
        ('item_descriptionL', TfidfVectorizer(
           # token_pattern='[a-zA-Z]+',
            max_features=1000,
            preprocessor=build_preprocessor('item_descriptionL',arr))),
        ('item_descriptionWL', TfidfVectorizer(
            token_pattern='\d+',
            max_features=100,
            preprocessor=build_preprocessor('item_descriptionWL',arr)))
    ])
    return vectorizer.fit_transform(arr.values)

print('[{}] Start vector2'.format( datetime.now() - start_time))
X_merge = vector2(merge)
X_train = X_merge[:row_train]
X_test = X_merge[row_train:]


print('[{}] Start LGB'.format( datetime.now() - start_time))

params = {
        'boosting': 'gbdt',
        'max_depth': 10,
        'min_data_in_leaf': 50,
        'num_leaves': 40,
       # 'learning_rate': 0.40,
        'learning_rate': 0.75,
        'objective': 'regression',
        'metric': 'rmse',
        'nthread': 4,
        'bagging_freq': 1,
        'subsample': 0.94,
        'colsample_bytree':0.68,
        'min_child_weight': 10,
        'is_unbalance': False,
        'verbose': -1,
        'seed': 1001
    }
dtrain = lgb.Dataset(X_train, label=y_train)
#dval = lgb.Dataset(Xval, label=yval)
watchlist = [dtrain]
#watchlist_names = ['train', 'val']

model = lgb.train(params,
                      train_set=dtrain,
                      num_boost_round=3250,
                      valid_sets=watchlist,
                 #     valid_names=watchlist_names,
                      early_stopping_rounds=100,
                      verbose_eval=250)
lgb_scores_val = model.predict(X_test)

print('[{}] Finished LGB work '.format(datetime.now() - start_time))
Lv = 0.32
Rv = 1 - Lv
y_pred = (lgb_scores_val * Lv + predR * Rv)


submission = pd.concat([test[['test_id']],pd.DataFrame(np.expm1(y_pred))],axis=1)
submission.columns = ['test_id', 'price']
submission.to_csv('submission_ridge_lgb82.csv',index=False)




print('[{}] Finished to all work data'.format(datetime.now() - start_time))