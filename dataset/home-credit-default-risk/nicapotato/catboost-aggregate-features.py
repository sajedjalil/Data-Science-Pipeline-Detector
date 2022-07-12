# Classification CatBoost for Home Credit Default Risk Kaggle Competition
# By Nick Brooks, May 2018 [Other Kaggler's work also used, citation throughout]
# https://www.kaggle.com/c/home-credit-default-risk

import time
notebookstart = time.time()

# General
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

# Viz
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")

# Thanks You Guillaume Martin for the Awesome Memory Optimizer!
# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtypes
        
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
        #else: df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df

# Load Files - Thanks Cafeal, very nifty
# https://www.kaggle.com/cafeal/lightgbm-trial-public-0-742
input_files = os.listdir("../input")
for filename in input_files:
    locals()[filename.rstrip('.csv')] = import_data(f'../input/{filename}')#.sample(1000)
    print(filename.rstrip('.csv'), "## Loaded and Optimized ##\n")
    
traindex = application_train.SK_ID_CURR
testdex = application_test.SK_ID_CURR
print('Train shape: {} Rows, {} Columns'.format(*application_train.shape))
print('Test shape: {} Rows, {} Columns'.format(*application_test.shape))

# Dependent Variable
y = application_train["TARGET"].copy()
application_train.drop("TARGET",axis=1,inplace= True)
df = pd.concat([application_train,application_test],axis=0)
del application_train, application_test ; gc.collect();
df.columns = ["Main_" + e for e in df.columns]

# Encoder:
categorical_columns = [f for f in df.columns if df[f].dtype == 'object']
lbl = preprocessing.LabelEncoder()
for col in categorical_columns:
    df[col] = lbl.fit_transform(df[col].astype(str))

# Aggregate Bureau_balance into Balance, and merge that into the Central Dataframe
print("Aggregate Bureau Balance DF")
agg_bureau_balance = bureau_balance.reset_index().groupby('SK_ID_BUREAU').agg(
    dict(MONTHS_BALANCE = ["sum","mean","max","min","std"],
         SK_ID_BUREAU = 'count'))
# Collapse Multi-Index and Preserve Origin Column Name
agg_bureau_balance.columns = pd.Index(["bureau_balance_" + e[0] +"_"+ e[1] for e in agg_bureau_balance.columns.tolist()])
STATUS = pd.get_dummies(bureau_balance[["SK_ID_BUREAU","STATUS"]], columns=["STATUS"]).groupby('SK_ID_BUREAU').sum()
# Float to Interger
for col in STATUS.columns: STATUS[col] = STATUS[col].astype(int)
agg_bureau_balance = pd.merge(agg_bureau_balance,STATUS,left_on="SK_ID_BUREAU",right_on="SK_ID_BUREAU", how= "left")
# Bureau Balance into Bureau Df
bureau = pd.merge(bureau,agg_bureau_balance, on="SK_ID_BUREAU", how= "left")
# Now Aggregate the Bureau Dataset
bureau.drop("SK_ID_BUREAU",axis=1,inplace=True)
cat = ["CREDIT_ACTIVE","CREDIT_CURRENCY","CREDIT_TYPE"]
notcat = [x for x in bureau.columns if x not in cat + ["SK_ID_CURR"]]

# Bureau Continous Variables
print("Aggregate Bureau DF")
agg_bureau = bureau.groupby('SK_ID_CURR').agg({k:["sum","mean","max","min","std"] for k in notcat})
agg_bureau.columns = pd.Index(["bureau_" + e[0] +"_"+ e[1] for e in agg_bureau.columns.tolist()])
df = pd.merge(df,agg_bureau, left_on="Main_SK_ID_CURR", right_on="SK_ID_CURR", how= "left")
# Bureau Categorical Variables
dummy_temp = pd.get_dummies(bureau[["SK_ID_CURR"]+cat], columns=cat).groupby('SK_ID_CURR').sum()
for col in dummy_temp.columns: dummy_temp[col] = dummy_temp[col].astype(int)
df = pd.merge(df,dummy_temp,left_on="Main_SK_ID_CURR", right_on="SK_ID_CURR", how= "left")
del dummy_temp,bureau, agg_bureau_balance, bureau_balance, agg_bureau; gc.collect();

# Aggregate and merge POS_CASH_balance into Central Dataframe
print("Aggregate POS CASH DF")
agg_POS_CASH_balance = POS_CASH_balance.reset_index().groupby('SK_ID_CURR').agg(
    dict(MONTHS_BALANCE = ["sum","mean","max","min","std"],
         CNT_INSTALMENT = ["sum","mean","max","min","std"],
         CNT_INSTALMENT_FUTURE = ["sum","mean","max","min","std"],
         SK_DPD = ["sum","mean","max","min","std"],
         SK_DPD_DEF = ["sum","mean","max","min","std"],
         SK_ID_CURR = 'count'))
agg_POS_CASH_balance.columns = pd.Index(["PCASH_" + e[0] +"_"+ e[1] for e in agg_POS_CASH_balance.columns.tolist()])
NAME_CONTRACT_STATUS = pd.get_dummies(POS_CASH_balance[["SK_ID_CURR","NAME_CONTRACT_STATUS"]], columns=["NAME_CONTRACT_STATUS"]).groupby('SK_ID_CURR').sum()
for col in NAME_CONTRACT_STATUS.columns: NAME_CONTRACT_STATUS[col] = NAME_CONTRACT_STATUS[col].astype(int)
agg_POS_CASH_balance = pd.merge(agg_POS_CASH_balance,NAME_CONTRACT_STATUS, left_on="SK_ID_CURR", right_on="SK_ID_CURR", how= "left")
df = pd.merge(df,agg_POS_CASH_balance,left_on="Main_SK_ID_CURR", right_on="SK_ID_CURR", how= "left")
del agg_POS_CASH_balance,NAME_CONTRACT_STATUS,POS_CASH_balance; gc.collect();

# Aggregate and merge Previous Application into Central Dataframe
# Distinguish Column Types
print("Aggregate Previous Application DF")
continuous_var = [x for x in previous_application.select_dtypes(include=['float16','float32','int8','int16','int32']).columns
                  if x not in ["SK_ID_PREV","SK_ID_CURR", "SELLERPLACE_AREA","NFLAG_LAST_APPL_IN_DAY","NFLAG_INSURED_ON_APPROVAL"]]
categorical_var = [x for x in previous_application.columns if x not in continuous_var + ['SK_ID_CURR']]
                   
 # previous_application Categorical Variables Aggregation
lbl = preprocessing.LabelEncoder()
for col in categorical_var: previous_application[col] = lbl.fit_transform(previous_application[col].astype(str))
agg_previous_application = previous_application.groupby('SK_ID_CURR').agg({k: lambda x: x.mode().iloc[0] for k in categorical_var})
agg_previous_application.columns = ['PREV1_{}_AGGMODE'.format(a) for a in agg_previous_application.columns]
df = pd.merge(df,agg_previous_application,left_on="Main_SK_ID_CURR", right_on="SK_ID_CURR", how= "left")
categorical_columns.extend(agg_previous_application.columns)
del agg_previous_application
agg_previous_application = previous_application.groupby('SK_ID_CURR').agg({k: ["nunique"] for k in categorical_var})
agg_previous_application.columns = pd.Index(["PREV2_" + e[0] +"_"+ e[1] for e in agg_previous_application.columns.tolist()])
df = pd.merge(df,agg_previous_application,left_on="Main_SK_ID_CURR", right_on="SK_ID_CURR", how= "left")
del agg_previous_application; gc.collect();
                   
# previous_application Continous Variables Aggregation
agg_previous_application = previous_application.groupby('SK_ID_CURR').agg({k:["sum","mean","max","min","std"] for k in continuous_var})
agg_previous_application.columns = pd.Index(["PAPP_" + e[0] +"_"+ e[1] for e in agg_previous_application.columns.tolist()])
df = pd.merge(df,agg_previous_application, left_on="Main_SK_ID_CURR", right_on="SK_ID_CURR", how= "left")
del previous_application,agg_previous_application; gc.collect();


# Optimize DF Once More
print("\n")
print("Pre-processing Finishing Touches")
# Set Index (out of the way)
df.set_index("Main_SK_ID_CURR",inplace=True)
# Fill Missing Values with 999
df.fillna(999,inplace=True)
df = reduce_mem_usage(df)

# Final Train and Test Set
X = df.loc[traindex,:]
print("Train Set Shape: {} Rows, {} Columns".format(*X.shape))
test = df.loc[testdex,:]
print("Submission Set Shape: {} Rows, {} Columns".format(*test.shape))
del df; gc.collect();

# Train/Valid
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=17)

print("\nCatboost Stage")   
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]
cat_features_inds = column_index(X, categorical_columns)

# Thank you Olivier for the K-fold OOF notebook, where I adapted his model evaluation graph and OOF method
# Adapted his method to CatBoost - Merci mon ami!
# https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm
folds = KFold(n_splits=4, shuffle=True, random_state=23)
oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(test.shape[0])
feature_importance_df = pd.DataFrame()

modelstart = time.time()
catboost_params= {"iterations": 1600,
                  "learning_rate": 0.09,
                  "depth": 6,
                  "bagging_temperature": 0.5,
                  "loss_function":'Logloss',
                  "eval_metric": 'CrossEntropy',
                  "metric_period":400,
                  "od_type":'Iter',
                  "od_wait": 200,
                  "random_seed":17,
                  "allow_writing_files":False}

# Out-of-Fold Method
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X)):
    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

    modelstart = time.time()
    cb_model = CatBoostClassifier(**catboost_params)
    cb_model.fit(X_train, y_train,
             eval_set=(X_valid, y_valid),
             cat_features=cat_features_inds,
             use_best_model=True,
             verbose=True)
    
    oof_preds[val_idx] = cb_model.predict_proba(X_valid)[:,1]
    sub_preds += cb_model.predict_proba(test)[:,1] / int(folds.n_splits)
    
    fold_importance_df = pd.DataFrame({'importance': cb_model.feature_importances_, 'feature': X.columns})
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    print('\nFold %2d AUC : %.6f\n' % (n_fold + 1, roc_auc_score(y_valid, oof_preds[val_idx])))
    del X_train, y_train, X_valid, y_valid, cb_model
    gc.collect()

print("\nModel Evaluation: Metrics and Visualization")
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))
cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:75].index
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
plt.figure(figsize=(10,16))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')

# Plot ROC curves
plt.figure(figsize=(6,6))
scores = [] 
for n_fold, (_, val_idx) in enumerate(folds.split(X)):  
    # Plot the roc curve
    fpr, tpr, thresholds = roc_curve(y.iloc[val_idx], oof_preds[val_idx])
    score = roc_auc_score(y.iloc[val_idx], oof_preds[val_idx])
    scores.append(score)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
fpr, tpr, thresholds = roc_curve(y, oof_preds)
score = roc_auc_score(y, oof_preds)
plt.plot(fpr, tpr, color='b',
         label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('CatBoost ROC Curve')
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig('roc_curve.png')

# Submit
cat_sub = pd.DataFrame(sub_preds,columns=["TARGET"],index=testdex)
cat_sub.to_csv("cat_sub_oof.csv",index=True,float_format='%.8f')
print("\nNotebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))