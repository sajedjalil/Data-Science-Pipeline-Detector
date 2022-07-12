import numpy as np
import pandas as pd
import gc
import lightgbm as lgb
from sklearn.model_selection import train_test_split

#print("Read Done")
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
 
def featureModify(isTrain, numRows):
    if isTrain:
        df = pd.read_csv('../input/train.csv',nrows=numRows) 
        df = reduce_mem_usage(df)
        df['event'] = df['event'].map({
            'A':0,
            'B':1,
            'C':2,
            'D':3
        })
    else:
        df = pd.read_csv('../input/test.csv',nrows=numRows)
        df = reduce_mem_usage(df)
        
    return df 
   
train = featureModify(True, None)
y = train['event']
train = train.drop('event',axis=1)
print(train.shape)
print(train.columns)

train, train_test, y, y_test = train_test_split(train, y, test_size=0.25, shuffle=False)
train = lgb.Dataset(train, label=y,categorical_feature=[1])
del y
gc.collect()


train_test = lgb.Dataset(train_test, label=y_test,categorical_feature=[1])
del y_test
gc.collect()

params = {
        "objective" : "multiclass", 
        "metric" : "multi_error", 
        'num_class':4,
        "num_leaves" : 30, 
        "learning_rate" : 0.01, 
        "bagging_fraction" : 0.9,
        "bagging_seed" : 0, 
        "num_threads" : 4,
        "colsample_bytree" : 0.5,
        'min_data_in_leaf':100, 
        'min_split_gain':0.00019
}
model = lgb.train(  params, 
                    train_set = train,
                    num_boost_round=2000,
                    early_stopping_rounds=200,
                    verbose_eval=100, 
                    valid_sets=[train,train_test]
                  )


test = featureModify(False, None)
print("Done test read")
df_sub = pd.DataFrame()
df_sub['id'] = test['id']
test = test.drop('id',axis=1)

y_pred = model.predict(test, num_iteration=model.best_iteration)

df_sub = pd.DataFrame(np.concatenate((np.arange(len(test))[:, np.newaxis], y_pred), axis=1), columns=['id', 'A', 'B', 'C', 'D'])
df_sub['id'] = df_sub['id'].astype(int)

print(df_sub)
df_sub.to_csv("submission.csv", index=False)