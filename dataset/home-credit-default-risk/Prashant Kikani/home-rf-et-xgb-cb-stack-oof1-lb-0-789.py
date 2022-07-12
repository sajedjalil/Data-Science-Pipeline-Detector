import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import gc
import os
print("my 5th OOF...!!")
SEED = 7    # CR7 always..
subm = pd.read_csv("../input/sample_submission.csv")

#----------
input_dir = os.path.join(os.pardir, 'input')
#print('Input files:\n{}'.format(os.listdir(input_dir)))
#print('Loading data sets...')

sample_size = None
app_train_df = pd.read_csv(os.path.join(input_dir, 'application_train.csv'), nrows=sample_size)
app_test_df = pd.read_csv(os.path.join(input_dir, 'application_test.csv'), nrows=sample_size)
bureau_df = pd.read_csv(os.path.join(input_dir, 'bureau.csv'), nrows=sample_size)
bureau_balance_df = pd.read_csv(os.path.join(input_dir, 'bureau_balance.csv'), nrows=sample_size)
credit_card_df = pd.read_csv(os.path.join(input_dir, 'credit_card_balance.csv'), nrows=sample_size)
pos_cash_df = pd.read_csv(os.path.join(input_dir, 'POS_CASH_balance.csv'), nrows=sample_size)
prev_app_df = pd.read_csv(os.path.join(input_dir, 'previous_application.csv'), nrows=sample_size)
install_df = pd.read_csv(os.path.join(input_dir, 'installments_payments.csv'), nrows=sample_size)
#print('Data loaded.\nMain application training data set shape = {}'.format(app_train_df.shape))
#print('Main application test data set shape = {}'.format(app_test_df.shape))
#print('Positive target proportion = {:.2f}'.format(app_train_df['TARGET'].mean()))

def agg_and_merge(left_df, right_df, agg_method, right_suffix):
    """ Aggregate a df by 'SK_ID_CURR' and merge it onto another.
    This method allows feature name """
    
    agg_df = right_df.groupby('SK_ID_CURR').agg(agg_method)
    merged_df = left_df.merge(agg_df, left_on='SK_ID_CURR', right_index=True, how='left',
                              suffixes=['', '_' + right_suffix + agg_method.upper()])
    return merged_df

def feature_engineering(app_data, bureau_df, bureau_balance_df, credit_card_df,
                        pos_cash_df, prev_app_df, install_df):
    """ 
    Process the input dataframes into a single one containing all the features. Requires
    a lot of aggregating of the supplementary datasets such that they have an entry per
    customer.
    
    Also, add any new features created from the existing ones
    """
    
    # # Add new features
    
    # Amount loaned relative to salary
    app_data['LOAN_INCOME_RATIO'] = app_data['AMT_CREDIT'] / app_data['AMT_INCOME_TOTAL']
    app_data['ANNUITY_INCOME_RATIO'] = app_data['AMT_ANNUITY'] / app_data['AMT_INCOME_TOTAL']
    
    # Number of overall payments (I think!)
    app_data['ANNUITY LENGTH'] = app_data['AMT_CREDIT'] / app_data['AMT_ANNUITY']
    
    # Social features
    app_data['WORKING_LIFE_RATIO'] = app_data['DAYS_EMPLOYED'] / app_data['DAYS_BIRTH']
    app_data['INCOME_PER_FAM'] = app_data['AMT_INCOME_TOTAL'] / app_data['CNT_FAM_MEMBERS']
    app_data['CHILDREN_RATIO'] = app_data['CNT_CHILDREN'] / app_data['CNT_FAM_MEMBERS']
    
    # A lot of the continuous days variables have integers as missing value indicators.
    prev_app_df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev_app_df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    prev_app_df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev_app_df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev_app_df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    
    # # Aggregate and merge supplementary datasets

    # Previous applications
    #print('Combined train & test input shape before any merging  = {}'.format(app_data.shape))
    agg_funs = {'SK_ID_CURR': 'count', 'AMT_CREDIT': 'sum'}
    prev_apps = prev_app_df.groupby('SK_ID_CURR').agg(agg_funs)
    prev_apps.columns = ['PREV APP COUNT', 'TOTAL PREV LOAN AMT']
    merged_df = app_data.merge(prev_apps, left_on='SK_ID_CURR', right_index=True, how='left')

    # Average the rest of the previous app data
    for agg_method in ['mean', 'max', 'min']:
        merged_df = agg_and_merge(merged_df, prev_app_df, agg_method, 'PRV')
    #print('Shape after merging with previous apps num data = {}'.format(merged_df.shape))
    
    # Previous app categorical features
    prev_app_df, cat_feats, _ = process_dataframe(prev_app_df)
    prev_apps_cat_avg = prev_app_df[cat_feats + ['SK_ID_CURR']].groupby('SK_ID_CURR')\
                             .agg({k: lambda x: str(x.mode().iloc[0]) for k in cat_feats})
    merged_df = merged_df.merge(prev_apps_cat_avg, left_on='SK_ID_CURR', right_index=True,
                            how='left', suffixes=['', '_BAVG'])
    #print('Shape after merging with previous apps cat data = {}'.format(merged_df.shape))

    # Credit card data - numerical features
    wm = lambda x: np.average(x, weights=-1/credit_card_df.loc[x.index, 'MONTHS_BALANCE'])
    credit_card_avgs = credit_card_df.groupby('SK_ID_CURR').agg(wm)   
    merged_df = merged_df.merge(credit_card_avgs, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_CC_WAVG'])
    for agg_method in ['mean', 'max', 'min']:
        merged_df = agg_and_merge(merged_df, credit_card_avgs, agg_method, 'CC')
    #print('Shape after merging with previous apps num data = {}'.format(merged_df.shape))
    
    # Credit card data - categorical features
    most_recent_index = credit_card_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    cat_feats = credit_card_df.columns[credit_card_df.dtypes == 'object'].tolist()  + ['SK_ID_CURR']
    merged_df = merged_df.merge(credit_card_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR', right_on='SK_ID_CURR',
                       how='left', suffixes=['', '_CCAVG'])
    #print('Shape after merging with credit card data = {}'.format(merged_df.shape))

    # Credit bureau data - numerical features
    for agg_method in ['mean', 'max', 'min']:
        merged_df = agg_and_merge(merged_df, bureau_df, agg_method, 'B')
    #print('Shape after merging with credit bureau data = {}'.format(merged_df.shape))
    
    # Bureau balance data
    most_recent_index = bureau_balance_df.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].idxmax()
    bureau_balance_df = bureau_balance_df.loc[most_recent_index, :]
    merged_df = merged_df.merge(bureau_balance_df, left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU',
                            how='left', suffixes=['', '_B_B'])
    #print('Shape after merging with bureau balance data = {}'.format(merged_df.shape))

    # Pos cash data - weight values by recency when averaging
    wm = lambda x: np.average(x, weights=-1/pos_cash_df.loc[x.index, 'MONTHS_BALANCE'])
    f = {'CNT_INSTALMENT': wm, 'CNT_INSTALMENT_FUTURE': wm, 'SK_DPD': wm, 'SK_DPD_DEF':wm}
    cash_avg = pos_cash_df.groupby('SK_ID_CURR')['CNT_INSTALMENT','CNT_INSTALMENT_FUTURE',
                                                 'SK_DPD', 'SK_DPD_DEF'].agg(f)
    merged_df = merged_df.merge(cash_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_CAVG'])
                                
    # Unweighted aggregations of numeric features
    for agg_method in ['mean', 'max', 'min']:
        merged_df = agg_and_merge(merged_df, pos_cash_df, agg_method, 'PC')
    
    # Pos cash data data - categorical features
    most_recent_index = pos_cash_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    cat_feats = pos_cash_df.columns[pos_cash_df.dtypes == 'object'].tolist()  + ['SK_ID_CURR']
    merged_df = merged_df.merge(pos_cash_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR', right_on='SK_ID_CURR',
                       how='left', suffixes=['', '_CAVG'])
    #print('Shape after merging with pos cash data = {}'.format(merged_df.shape))

    # Installments data
    for agg_method in ['mean', 'max', 'min']:
        merged_df = agg_and_merge(merged_df, install_df, agg_method, 'I')    
    #print('Shape after merging with installments data = {}'.format(merged_df.shape))
    
    # Add more value counts
    merged_df = merged_df.merge(pd.DataFrame(bureau_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_CNT_BUREAU'])
    merged_df = merged_df.merge(pd.DataFrame(credit_card_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_CNT_CRED_CARD'])
    merged_df = merged_df.merge(pd.DataFrame(pos_cash_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_CNT_POS_CASH'])
    merged_df = merged_df.merge(pd.DataFrame(install_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_CNT_INSTALL'])
    #print('Shape after merging with counts data = {}'.format(merged_df.shape))

    return merged_df

def process_dataframe(input_df, encoder_dict=None):
    """ Process a dataframe into a form useable by LightGBM """

    # Label encode categoricals
    #print('Label encoding categorical features...')
    categorical_feats = input_df.columns[input_df.dtypes == 'object']
    for feat in categorical_feats:
        encoder = LabelEncoder()
        input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))
    #print('Label encoding complete.')

    return input_df, categorical_feats.tolist(), encoder_dict

# Merge the datasets into a single one for training
len_train = len(app_train_df)
app_both = pd.concat([app_train_df, app_test_df])
print('before feat engg')
merged_df = feature_engineering(app_both, bureau_df, bureau_balance_df, credit_card_df,
                                pos_cash_df, prev_app_df, install_df)
print('after feat engg')
merged_df.to_csv('processed_input_data.csv', index=False)

# Separate metadata
meta_cols = ['SK_ID_CURR']
meta_df = merged_df[meta_cols]
merged_df.drop(columns=meta_cols, inplace=True)

# Process the data set.
merged_df, categorical_feats, encoder_dict = process_dataframe(input_df=merged_df)

# Capture other categorical features not as object data types:
non_obj_categoricals = [
    'FONDKAPREMONT_MODE', 'HOUR_APPR_PROCESS_START', 'HOUSETYPE_MODE',
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
    'NAME_INCOME_TYPE', 'NAME_TYPE_SUITE', 'OCCUPATION_TYPE',
    'ORGANIZATION_TYPE', 'STATUS', 'NAME_CONTRACT_STATUS_CAVG',
    'WALLSMATERIAL_MODE', 'WEEKDAY_APPR_PROCESS_START', 'NAME_CONTRACT_TYPE_BAVG',
    'WEEKDAY_APPR_PROCESS_START_BAVG', 'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 
    'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON', 'NAME_TYPE_SUITE_BAVG', 
    'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 
    'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY', 
    'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION', 'NAME_CONTRACT_STATUS_CCAVG' 
]
categorical_feats = categorical_feats + non_obj_categoricals

# Re-separate into train and test
train_df = merged_df[:len_train]
test_df = merged_df[len_train:]
del merged_df, app_test_df, bureau_df, bureau_balance_df, credit_card_df, pos_cash_df, prev_app_df
gc.collect()

""" Train the model """
target = train_df.pop('TARGET')
test_df.drop(columns='TARGET', inplace=True)
#---------------
print('data done')
data_train = train_df
data_test = test_df

data_train.fillna(-1, inplace=True)
data_test.fillna(-1, inplace=True)
cols = data_train.columns

ntrain = data_train.shape[0]
ntest = data_test.shape[0]

print(data_train.shape)
from sklearn.cross_validation import KFold
kf = KFold(data_train.shape[0], n_folds=5, shuffle=True, random_state=7)
NFOLDS = 5
x_train = np.array(data_train)
x_test = np.array(data_test)
y_train = np.array(target)

# from https://www.kaggle.com/mmueller/stacking-starter?scriptVersionId=390867/code
class SklearnWrapper(object):
    def __init__(self, clf, seed=7, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        print("Training..")
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        print("Predicting..")
        return self.clf.predict_proba(x)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        print("Training..")
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        print("Predicting..")
        return self.gbdt.predict(xgb.DMatrix(x))


def get_oof(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)[:,1]  # or [:,0]
        oof_test_skf[i, :] = clf.predict(x_test)[:,1]  # or [:,0]

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
def get_oof_xgb(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)  # or [:,0]
        oof_test_skf[i, :] = clf.predict(x_test)  # or [:,0]

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
    
et_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.2,
    'max_depth': 8,
    'min_samples_leaf': 2,
}

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 7,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'auc',
    'nrounds': 350
}

cb_params = {
'iterations':1000,
'learning_rate':0.1,
'depth':6,
'l2_leaf_reg':40,
'bootstrap_type':'Bernoulli',
'subsample':0.7,
'scale_pos_weight':5,
'eval_metric':'AUC',
'metric_period':50,
'od_type':'Iter',
'od_wait':45,
'allow_writing_files':False    
}

xg = XgbWrapper(seed=SEED, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
cb = SklearnWrapper(clf=CatBoostClassifier, seed=SEED, params=cb_params)

print("xg..")
xg_oof_train, xg_oof_test = get_oof_xgb(xg)
print("et..")
et_oof_train, et_oof_test = get_oof(et)
print("rf..")
rf_oof_train, rf_oof_test = get_oof(rf)
print("cb..")
cb_oof_train, cb_oof_test = get_oof(cb)

x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train, cb_oof_train), axis=1)
x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test, cb_oof_test), axis=1)

np.save('x_train', x_train)
np.save('x_test', x_test)
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'auc',
}

print("xgb cv..")
res = xgb.cv(xgb_params, dtrain, num_boost_round=500, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=25, verbose_eval=10, show_stdv=True)
best_nrounds = res.shape[0] - 1

print("meta xgb train..")
gbdt = xgb.train(xgb_params, dtrain, best_nrounds)
fi = gbdt.predict(dtest)
fi = np.array(fi)
np.save('fi', fi)

subm['TARGET'] = fi
subm.to_csv('stack3_diff_data.csv', index=False)