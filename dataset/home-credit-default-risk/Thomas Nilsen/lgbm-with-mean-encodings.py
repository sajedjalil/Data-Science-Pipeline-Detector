import gc
import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv("../input/application_train.csv")
test = pd.read_csv("../input/application_test.csv")
test.loc[:, "is_test"] = True

alldata = pd.concat([train, test], axis=0)

dataframes = [
    (
        "previous_application",  
        "SK_ID_PREV", 
        pd.read_csv('../input/previous_application.csv')
    ),
    (
        "bureau", 
        "SK_ID_BUREAU",
        pd.read_csv('../input/bureau.csv')
    )
]

for name, key, df in dataframes:
    cat_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    tmp_df_mean = df.groupby("SK_ID_CURR").mean().drop(key, axis=1)
    tmp_df_mean.loc[:, "%s_count"%name] = df.groupby('SK_ID_CURR').count()[key].values
    
    cols_to_keep = [col for col in tmp_df_mean.columns if col not in alldata.columns]
    alldata = pd.merge(
        alldata, 
        tmp_df_mean[cols_to_keep].reset_index(), 
        on="SK_ID_CURR", 
        how="left"
    )
    del tmp_df_mean
    gc.collect()

alldata.loc[:, "is_test"] = alldata.loc[:, "is_test"].fillna(False)
del train, test
gc.collect()


categorical_cols = [col for col in alldata.select_dtypes(include=["object"]).columns]
numerical_cols = [col for col in alldata.select_dtypes(exclude=["object"]).columns]
numerical_cols = [col for col in numerical_cols if col not in ["SK_ID_CURR", "TARGET", "is_test"]]


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
cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(X_train.shape[0])

sub = pd.read_csv("../input/sample_submission.csv")
sub["TARGET"] = 0


for i, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    
    X_fit = X_train.iloc[fit_idx]
    y_fit = y_train.iloc[fit_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx]
    
    model = LGBMClassifier(
        max_depth=5,
        num_leaves=5**2 - 1,
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
        early_stopping_rounds=200,
        verbose=False
    )
    
    
    oof_preds[val_idx] = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
    sub['TARGET'] += model.predict_proba(X_test, num_iteration=model.best_iteration_)[:,1]
    
    print("Fold {} AUC: {:.8f}".format(i+1, roc_auc_score(y_val, oof_preds[val_idx])))
    
print('Full AUC score %.8f' % roc_auc_score(y_train, oof_preds))   
    
sub["TARGET"] /= n_splits
sub.to_csv("lgbm_mean_encondings.csv", index=None, float_format="%.8f")