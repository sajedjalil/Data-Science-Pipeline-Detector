"""
This kernel provides a simple starter framework for a LightGBM model.

There are two supplementary functions designed with room to grow as the kernel develops:
    - feature_engineering: Contains the appending of extra features to the main dataset. There are
      a lot of datasets to go through in this challenge, so this is very much in progress
    - process_dataframe: Takes the engineered dataframe and makes it ready for LightGBM. Currently
      is only label encoding thanks to LightGBMs flexbility with nulls and not needing one-hots
"""
import os
import gc
import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

""" Load and process inputs """
input_dir = os.path.join(os.pardir, 'input/home-credit-default-risk/')
print('Input files:\n{}'.format(os.listdir(input_dir)))
print('Loading data sets...')

sample_size = None
app_train_df = pd.read_csv(os.path.join(input_dir, 'application_train.csv'), nrows=sample_size)
app_test_df = pd.read_csv(os.path.join(input_dir, 'application_test.csv'), nrows=sample_size)
bureau_df = pd.read_csv(os.path.join(input_dir, 'bureau.csv'), nrows=sample_size)
bureau_balance_df = pd.read_csv(os.path.join(input_dir, 'bureau_balance.csv'), nrows=sample_size)
print('Data loaded.\nMain application training data set shape = {}'.format(app_train_df.shape))
print('Main application test data set shape = {}'.format(app_test_df.shape))
print('Positive target proportion = {:.2f}'.format(app_train_df['TARGET'].mean()))

def agg_and_merge(left_df, right_df, agg_method, right_suffix):
    """ Aggregate a df by 'SK_ID_CURR' and merge it onto another.
    This method allows feature name """
    
    agg_df = right_df.groupby('SK_ID_CURR').agg(agg_method)
    merged_df = left_df.merge(agg_df, left_on='SK_ID_CURR', right_index=True, how='left',
                              suffixes=['', '_' + right_suffix + agg_method.upper()])
    return merged_df

def feature_engineering(app_data, bureau_df, bureau_balance_df):
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
    
    # # Aggregate and merge supplementary datasets

    # Credit bureau data - numerical features
    for agg_method in ['mean', 'max', 'min']:
        merged_df = agg_and_merge(app_data, bureau_df, agg_method, 'B')
    print('Shape after merging with credit bureau data = {}'.format(merged_df.shape))
    
    # Bureau balance data
    most_recent_index = bureau_balance_df.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].idxmax()
    bureau_balance_df = bureau_balance_df.loc[most_recent_index, :]
    merged_df = merged_df.merge(bureau_balance_df, left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU',
                            how='left', suffixes=['', '_B_B'])
    print('Shape after merging with bureau balance data = {}'.format(merged_df.shape))

    # Add more value counts
    merged_df = merged_df.merge(pd.DataFrame(bureau_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_CNT_BUREAU'])
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
app_both = pd.concat([app_train_df, app_test_df],sort=True)
merged_df = feature_engineering(app_both, bureau_df, bureau_balance_df)
merged_df.to_csv('processed_input_bureauonly_csv',index=False)

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
    'ORGANIZATION_TYPE', 'STATUS',
    'WALLSMATERIAL_MODE', 'WEEKDAY_APPR_PROCESS_START']

categorical_feats = categorical_feats + non_obj_categoricals

# Re-separate into train and test
train_df = merged_df[:len_train]
test_df = merged_df[len_train:]
del merged_df, app_test_df, bureau_df, bureau_balance_df
gc.collect()

""" Train the model """
target = train_df.pop('TARGET')
test_df.drop(columns='TARGET', inplace=True)
lgbm_train = lgbm.Dataset(data=train_df,
                          label=target,
                          categorical_feature=categorical_feats,
                          free_raw_data=False)
del app_train_df
gc.collect()

lgbm_params = {
    'boosting': 'dart',
    'application': 'binary',
    'learning_rate': 0.1,
    'min_data_in_leaf': 30,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.5,
    'scale_pos_weight': 2,
    'drop_rate': 0.02
}

cv_results = lgbm.cv(train_set=lgbm_train,
                     params=lgbm_params,
                     nfold=5,
                     num_boost_round=600,
                     early_stopping_rounds=50,
                     verbose_eval=50,
                     metrics=['auc'],stratified=False)

optimum_boost_rounds = np.argmax(cv_results['auc-mean'])
print('Optimum boost rounds = {}'.format(optimum_boost_rounds))
print('Best CV result = {}'.format(np.max(cv_results['auc-mean'])))

clf = lgbm.train(train_set=lgbm_train,
                 params=lgbm_params,
                 num_boost_round=optimum_boost_rounds)

""" Predict on test set and create submission """
y_pred = clf.predict(test_df)
out_df = pd.DataFrame({'SK_ID_CURR': meta_df['SK_ID_CURR'][len_train:], 'TARGET': y_pred})
out_df.to_csv('submission.csv', index=False)

fig, (ax, ax1) = plt.subplots(1, 2, figsize=[11, 7])
lgbm.plot_importance(clf, ax=ax, max_num_features=20, importance_type='split')
lgbm.plot_importance(clf, ax=ax1, max_num_features=20, importance_type='gain')
ax.set_title('Importance by splits')
ax1.set_title('Importance by gain')
plt.tight_layout()
plt.savefig('feature_importance.png')
