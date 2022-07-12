"""
This kernel is forked from a LightGBM script I have recently tried. The idea is to test the 
performance of CatBoost on the same feature set. If performance is comparable, the two models
could be stacked to provide an even better one.

As before, there are two supplementary functions designed with room to grow as the kernel develops:
    - feature_engineering: Contains the appending of extra features to the main dataset. There are
      a lot of datasets to go through in this challenge, so this is very much in progress
    - process_dataframe: Takes the engineered dataframe and makes it ready for LightGBM. Currently
      is only label encoding thanks to LightGBMs flexbility with nulls and not needing one-hots
"""
import os
import gc
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

""" Load and process inputs """
input_dir = os.path.join(os.pardir, 'input')
print('Input files:\n{}'.format(os.listdir(input_dir)))
print('Loading data sets...')

sample_size = None
app_train_df = pd.read_csv(os.path.join(input_dir, 'application_train.csv'), nrows=sample_size)
app_test_df = pd.read_csv(os.path.join(input_dir, 'application_test.csv'), nrows=sample_size)
bureau_df = pd.read_csv(os.path.join(input_dir, 'bureau.csv'), nrows=sample_size)
bureau_balance_df = pd.read_csv(os.path.join(input_dir, 'bureau_balance.csv'), nrows=sample_size)
credit_card_df = pd.read_csv(os.path.join(input_dir, 'credit_card_balance.csv'), nrows=sample_size)
pos_cash_df = pd.read_csv(os.path.join(input_dir, 'POS_CASH_balance.csv'), nrows=sample_size)
prev_app_df = pd.read_csv(os.path.join(input_dir, 'previous_application.csv'), nrows=sample_size)
install_df = pd.read_csv(os.path.join(input_dir, 'installments_payments.csv'), nrows=sample_size)
print('Data loaded.\nMain application training data set shape = {}'.format(app_train_df.shape))
print('Main application test data set shape = {}'.format(app_test_df.shape))
print('Positive target proportion = {:.2f}'.format(app_train_df['TARGET'].mean()))

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
    print('Combined train & test input shape before any merging  = {}'.format(app_data.shape))
    agg_funs = {'SK_ID_CURR': 'count', 'AMT_CREDIT': 'sum'}
    prev_apps = prev_app_df.groupby('SK_ID_CURR').agg(agg_funs)
    prev_apps.columns = ['PREV APP COUNT', 'TOTAL PREV LOAN AMT']
    merged_df = app_data.merge(prev_apps, left_on='SK_ID_CURR', right_index=True, how='left')

    # Average the rest of the previous app data
    prev_apps_avg = prev_app_df.groupby('SK_ID_CURR').mean()
    merged_df = merged_df.merge(prev_apps_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_PAVG'])
                                
    # Max and min the previous app data
    prev_apps_max = prev_app_df.groupby('SK_ID_CURR').max()
    merged_df = merged_df.merge(prev_apps_max, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_PMAX'])
    prev_apps_min = prev_app_df.groupby('SK_ID_CURR').min()
    merged_df = merged_df.merge(prev_apps_min, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_PMIN'])
    print('Shape after merging with previous apps num data = {}'.format(merged_df.shape))
    
    # Previous app categorical features
    prev_app_df, cat_feats, _ = process_dataframe(prev_app_df)
    prev_apps_cat_avg = prev_app_df[cat_feats + ['SK_ID_CURR']].groupby('SK_ID_CURR')\
                             .agg({k: lambda x: str(x.mode().iloc[0]) for k in cat_feats})
    merged_df = merged_df.merge(prev_apps_cat_avg, left_on='SK_ID_CURR', right_index=True,
                            how='left', suffixes=['', '_BAVG'])
    print('Shape after merging with previous apps cat data = {}'.format(merged_df.shape))

    # Credit card data - numerical features
    wm = lambda x: np.average(x, weights=-1/credit_card_df.loc[x.index, 'MONTHS_BALANCE'])
    credit_card_avgs = credit_card_df.groupby('SK_ID_CURR').agg(wm)   
    merged_df = merged_df.merge(credit_card_avgs, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_CCAVG'])
    
    # Credit card data - categorical features
    most_recent_index = credit_card_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    cat_feats = credit_card_df.columns[credit_card_df.dtypes == 'object'].tolist()  + ['SK_ID_CURR']
    merged_df = merged_df.merge(credit_card_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR', right_on='SK_ID_CURR',
                       how='left', suffixes=['', '_CCAVG'])
    print('Shape after merging with credit card data = {}'.format(merged_df.shape))

    # Credit bureau data - numerical features
    credit_bureau_avgs = bureau_df.groupby('SK_ID_CURR').mean()
    merged_df = merged_df.merge(credit_bureau_avgs, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_BAVG'])
    print('Shape after merging with credit bureau data = {}'.format(merged_df.shape))
    
    # Bureau balance data
    most_recent_index = bureau_balance_df.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].idxmax()
    bureau_balance_df = bureau_balance_df.loc[most_recent_index, :]
    merged_df = merged_df.merge(bureau_balance_df, left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU',
                            how='left', suffixes=['', '_B_B'])
    print('Shape after merging with bureau balance data = {}'.format(merged_df.shape))

    # Pos cash data - weight values by recency when averaging
    wm = lambda x: np.average(x, weights=-1/pos_cash_df.loc[x.index, 'MONTHS_BALANCE'])
    f = {'CNT_INSTALMENT': wm, 'CNT_INSTALMENT_FUTURE': wm, 'SK_DPD': wm, 'SK_DPD_DEF':wm}
    cash_avg = pos_cash_df.groupby('SK_ID_CURR')['CNT_INSTALMENT','CNT_INSTALMENT_FUTURE',
                                                 'SK_DPD', 'SK_DPD_DEF'].agg(f)
    merged_df = merged_df.merge(cash_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_CAVG'])
    
    # Pos cash data data - categorical features
    most_recent_index = pos_cash_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    cat_feats = pos_cash_df.columns[pos_cash_df.dtypes == 'object'].tolist()  + ['SK_ID_CURR']
    merged_df = merged_df.merge(pos_cash_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR', right_on='SK_ID_CURR',
                       how='left', suffixes=['', '_CAVG'])
    print('Shape after merging with pos cash data = {}'.format(merged_df.shape))

    # Installments data
    ins_avg = install_df.groupby('SK_ID_CURR').mean()
    merged_df = merged_df.merge(ins_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_IAVG'])
    print('Shape after merging with installments data = {}'.format(merged_df.shape))
    
    # Add more value counts
    merged_df = merged_df.merge(pd.DataFrame(bureau_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_CNT_BUREAU'])
    merged_df = merged_df.merge(pd.DataFrame(credit_card_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_CNT_CRED_CARD'])
    merged_df = merged_df.merge(pd.DataFrame(pos_cash_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_CNT_POS_CASH'])
    merged_df = merged_df.merge(pd.DataFrame(install_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_CNT_INSTALL'])
    print('Shape after merging with counts data = {}'.format(merged_df.shape))

    return merged_df

def process_dataframe(input_df, encoder_dict=None):
    """ Process a dataframe into a form useable by LightGBM """

    # Label encode categoricals
    print('Label encoding categorical features...')
    categorical_feats = input_df.columns[input_df.dtypes == 'object']
    for feat in categorical_feats:
        encoder = LabelEncoder()
        input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))
    print('Label encoding complete.')

    return input_df, categorical_feats.tolist(), encoder_dict

# Merge the datasets into a single one for training
len_train = len(app_train_df)
app_both = pd.concat([app_train_df, app_test_df])
merged_df = feature_engineering(app_both, bureau_df, bureau_balance_df, credit_card_df,
                                pos_cash_df, prev_app_df, install_df)

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
# Extract target
target = train_df.pop('TARGET')
test_df.drop(columns='TARGET', inplace=True)

# Define the categorical feats and CatBoost params
cat_feats_idx = np.unique([np.argmax(train_df.columns == feat) for feat in categorical_feats])
cb_params = {
    'loss_function': 'Logloss',
    # 'learning_rate': 0.1,
    'iterations': 500,
    'depth': 8,
    'class_weights': [1, 2],
    'bootstrap_type': 'Bernoulli',
    'random_seed': 9,
    'verbose': False
}

# Estimate LB score through CV
cb_clf = CatBoostClassifier(**cb_params)
cv_score = cross_val_score(estimator=cb_clf, 
                           X=train_df, 
                           y=target, 
                           scoring='roc_auc',
                           cv=5,
                           verbose=1)
print('Cross validation AUC: {:.4f}'.format(np.mean(cv_score)))

# Fit final model
cb_clf.fit(X=train_df, y=target, cat_features=cat_feats_idx)

""" Predict on test set and create submission """
y_pred = cb_clf.predict_proba(test_df)
out_df = pd.DataFrame({'SK_ID_CURR': meta_df['SK_ID_CURR'][len_train:], 'TARGET': y_pred[:, 1]})
out_df.to_csv('submission.csv', index=False)

importances = pd.DataFrame({'Feature': train_df.columns, 'Importance': cb_clf.feature_importances_})
top_importances = importances.sort_values(by='Importance', ascending=False)[:20]
fig, ax = plt.subplots(1, 1, figsize=[5, 7])
sns.barplot(x='Importance', y='Feature', data=top_importances, ax=ax)
plt.tight_layout()
plt.savefig('feature_importance.png')
