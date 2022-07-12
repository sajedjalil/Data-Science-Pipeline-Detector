import gc
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

#Great snippet from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
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
        #else:
        #    df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
    
    
    
train = reduce_mem_usage(pd.read_csv("../input/application_train.csv"))
test = reduce_mem_usage(pd.read_csv("../input/application_test.csv"))
test.loc[:, "is_test"] = True

alldata = pd.concat([train, test], axis=0)
alldata.loc[:, "is_test"] = alldata.loc[:, "is_test"].fillna(False)

num_cols = alldata.select_dtypes(exclude=["object"]).columns
num_cols = [col for col in num_cols if col not in ["SK_ID_CURR", "is_test", "TARGET"]]

del train, test; gc.collect()


bureau_balance = reduce_mem_usage(pd.read_csv("../input/bureau_balance.csv"))
bureau =  reduce_mem_usage(pd.read_csv('../input/bureau.csv'))
full_bureau = pd.merge(bureau, bureau_balance, on="SK_ID_BUREAU", how="left")

del bureau_balance, bureau
gc.collect()

dataframes = [
    (
        "previous_application",  
        "SK_ID_PREV", 
        reduce_mem_usage(pd.read_csv('../input/previous_application.csv'))
    ),
    (
        "bureau", 
        "SK_ID_BUREAU",
        full_bureau
    ),
    (
        "POS_CASH_balance",
        "SK_ID_PREV",
        reduce_mem_usage(pd.read_csv("../input/POS_CASH_balance.csv"))
    ),
    (
        "credit_card_balance",
        "SK_ID_PREV",
        reduce_mem_usage(pd.read_csv("../input/credit_card_balance.csv"))
    ),
    (
        "installments_payments",
        "SK_ID_PREV",
        reduce_mem_usage(pd.read_csv("../input/installments_payments.csv"))
    )
]


for name, key, df in dataframes:
    print("Working on %s..." % name, end="")
    cat_cols = df.select_dtypes(include=["object"]).columns
    
    df = pd.get_dummies(
        df, 
        columns=cat_cols, 
        drop_first=True, 
        dummy_na=True
    )
    
    tmp_df_mean = df.groupby("SK_ID_CURR").agg(["mean", "max"]).drop(key, axis=1)
    tmp_df_mean.columns = ["_".join(col) for col in tmp_df_mean.columns.ravel()]
    #tmp_df_mean.loc[:, "%s_count"%name] = df.loc[:, "SK_ID_CURR"].map(df.groupby('SK_ID_CURR').count()[key])
    
    cols_to_keep = [col for col in tmp_df_mean.columns if col not in alldata.columns]
    alldata = pd.merge(
        alldata, 
        tmp_df_mean[cols_to_keep].reset_index(), 
        on="SK_ID_CURR", 
        how="left"
    )
    del tmp_df_mean
    gc.collect()
    print("done")


for name, key, df in dataframes:
    del df; gc.collect()
    
del dataframes; gc.collect()


categorical_cols = [col for col in alldata.select_dtypes(include=["object"]).columns]

#Mean encoding of categorical variables
for col in categorical_cols:
    means = alldata.loc[~alldata.is_test, :].groupby(col)["TARGET"].mean()
    alldata.loc[:, "%s_MEAN" % col] = alldata.loc[:, col].map(means)
    
    #Missing values is filled with global mean
    alldata.loc[:, "%s_MEAN" % col] = alldata.loc[:, "%s_MEAN" % col].fillna(means.mean())
    
    
    
alldata.loc[:, categorical_cols] = alldata.loc[:, categorical_cols].apply(lambda x: LabelEncoder().fit_transform(x.astype(str)))

cols_to_drop = ["SK_ID_CURR", "TARGET", "is_test"]
X_train = alldata.loc[~alldata.is_test, :].drop(cols_to_drop, axis=1)
y_train = alldata.loc[~alldata.is_test, "TARGET"]

X_test = alldata.loc[alldata.is_test, :].drop(cols_to_drop, axis=1)

n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, random_state=42)

oof_preds = np.zeros(X_train.shape[0])

sub = pd.read_csv("../input/sample_submission.csv")
sub["TARGET"] = 0
feature_importances = pd.DataFrame()

for i, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    
    X_fit = X_train.iloc[fit_idx]
    y_fit = y_train.iloc[fit_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx]
    
    model = LGBMClassifier(
        max_depth=5,
        num_leaves=5 ** 2 - 1,
        learning_rate=0.007,
        n_estimators=30000,
        min_child_samples=80,
        subsample=0.8,
        colsample_bytree=1,
        reg_alpha=0,
        reg_lambda=0,
        random_state=np.random.randint(10e6)
    )

    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_fit, y_fit), (X_val, y_val)],
        eval_names=('fit', 'val'),
        eval_metric='auc',
        early_stopping_rounds=150,
        verbose=False
    )
    
    
    oof_preds[val_idx] = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
    sub['TARGET'] += model.predict_proba(X_test, num_iteration=model.best_iteration_)[:,1]
    
    fi = pd.DataFrame()
    fi["feature"] = X_train.columns
    fi["importance"] = model.feature_importances_
    fi["fold"] = (i+1)
    
    feature_importances = pd.concat([feature_importances, fi], axis=0)
    
    print("Fold {} AUC: {:.8f}".format(i+1, roc_auc_score(y_val, oof_preds[val_idx])))
    
print('Full AUC score %.8f' % roc_auc_score(y_train, oof_preds))   
    
sub["TARGET"] /= n_splits
sub.to_csv("lgbm.csv", index=None, float_format="%.8f")