import time
start = time.time()

# 1. Define the problem
# The goal of this competition is to predict whether an application to DonorsChoose
# is accepted. Submissions are evaluated on area under the ROC curve between the
# predicted probability and the observed target.

# ScoreMethod = area under the ROC curve

# 1.1 Load Library--------------------------------------------------------------
# data analysis and wrangling
import gc
import pandas as pd
import numpy as np
import scipy as sp
from tqdm import tqdm
from sklearn import metrics

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999



# 1.2 Load data --------------------------------------------------------------
data_dir = '../input/'

# Load Data
dtype = {
    'id': str,
    'teacher_id': str,
    'teacher_prefix': str,
    'school_state': str,
    'project_submitted_datetime': str,
    'project_grade_category': str,
    'project_subject_categories': str,
    'project_subject_subcategories': str,
    'project_title': str,
    'project_essay_1': str,
    'project_essay_2': str,
    'project_essay_3': str,
    'project_essay_4': str,
    'project_resource_summary': str,
    'teacher_number_of_previously_posted_projects': int,
    'project_is_approved': np.uint8}

resources_df = pd.read_csv(data_dir + 'resources.csv')
test_df = pd.read_csv(data_dir + 'test.csv', dtype=dtype, low_memory=True)
train_df = pd.read_csv(data_dir + 'train.csv', dtype=dtype,low_memory=True)
# test_df = pd.read_csv(data_dir + 'test.csv', dtype=dtype, skiprows = range(1, 70000), low_memory=True)
# train_df = pd.read_csv(data_dir + 'train.csv', dtype=dtype, skiprows = range(1, 180000), low_memory=True)
Full_df = pd.concat([test_df,train_df])
del [test_df, train_df]
gc.collect()

# 2. Prepare data --------------------------------------------------------------
# 6C : Checking, Correcting, Completing, Creating, Combining, Converting

# # 2.1 Checking - missing values, variables -------------------------------------
# Full_df.isnull().sum()
# resources_df.isnull().sum()

# 2.2 Correcting ---------------------------------------------------------------
# Converting project_submitted_datetime from object to date time
Full_df['project_submitted_datetime']= pd.to_datetime(Full_df['project_submitted_datetime'])


# 2.3 Completing - NA ----------------------------------------------------------
Full_df['project_essay_full'] = Full_df['project_essay_1'].map(str) + ' ' + Full_df['project_essay_2'].map(str) + ' ' + Full_df['project_essay_3'].map(str) + ' ' + Full_df['project_essay_4'].map(str)
Full_df = Full_df.drop(columns = ['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4'])

# There are a few missing 'teacher_prefix'. Replace with the most common prefix
# Full_df.groupby('teacher_prefix',as_index=False)['id'].count().sort_values('id')
Full_df['teacher_prefix'] = Full_df['teacher_prefix'].fillna('Mrs.')

# fill NA with empty string
resources_df['description'] = resources_df['description'].fillna('')


# 2.4 Creating - Transform and Engineer data -----------------------------------

# a. teacher_id - other prjects' approval rate and of and previously submitted
# approval rate of the teacher's other submission
approved_df = Full_df.groupby('teacher_id',as_index=False)['project_is_approved'].sum()
submitted_df = Full_df.groupby('teacher_id',as_index=False)['id'].count()
approved_df = approved_df.rename(columns = {'project_is_approved': 'teacher_projects_approved_sum'})
submitted_df = submitted_df.rename(columns = {'id': 'teacher_projects_count'})
Full_df = Full_df.merge(approved_df, on = ['teacher_id'])
Full_df = Full_df.merge(submitted_df, on = ['teacher_id'])
# approval rate for the teacher for other projects
Full_df['teacher_projects_approved_rate'] = (Full_df['teacher_projects_approved_sum'] - Full_df['project_is_approved']) / (Full_df['teacher_projects_count'] - 1)
teacher_rate_mean = Full_df['teacher_projects_approved_rate'].mean()
Full_df['teacher_projects_approved_rate'] = Full_df['teacher_projects_approved_rate'].fillna(teacher_rate_mean)
Full_df['teacher_projects_approved_rate'] = Full_df.apply(lambda r: r['teacher_projects_approved_rate'] if (r['teacher_projects_count']> 4) else teacher_rate_mean, axis = 1)
# Full_df['teacher_projects_approved_rate'] = Full_df['teacher_projects_approved_rate'].clip(0.5,1)

# d. project_submitted_datetime - Submission Month of the day
firstDate = Full_df['project_submitted_datetime'].min()
Full_df['day_num'] = (Full_df['project_submitted_datetime'].map(lambda x: (x - firstDate).days))

# f. project_subject_categories - category of the project (e.g., "Music & The Arts")
cateWords = pd.Series(', '.join(Full_df["project_subject_categories"]).split(', ')).value_counts().to_frame().reset_index()
cateWords.columns = ['cate_word', 'Count']
for index, word in tqdm(enumerate(cateWords['cate_word'])):
    Full_df['cate_' + str(index)] = np.where(Full_df['project_subject_categories'].str.contains(word), 1, 0)
    Full_df['cate_' + str(index)] = Full_df['cate_' + str(index)].astype(int)

# g. project_subject_subcategories - sub-category of the project (e.g., "Visual Arts")
subCateWords = pd.Series(', '.join(Full_df["project_subject_subcategories"]).split(', ')).value_counts().to_frame().reset_index()
subCateWords.columns = ['sub_cate_word', 'Count']
for index, word in tqdm(enumerate(subCateWords['sub_cate_word'])):
    Full_df['sub_cate_' + str(index)] = np.where(Full_df['project_subject_subcategories'].str.contains(word), 1, 0)
    Full_df['sub_cate_' + str(index)] = Full_df['sub_cate_' + str(index)].astype(int)

# h. quantity and price of resources
resources_df['resource_price_total'] = resources_df['quantity'] * resources_df['price']
resources_price_total_df = resources_df.groupby(['id'],as_index=False)['resource_price_total'].sum()
resources_quantity_total_df = resources_df.groupby(['id'],as_index=False)['quantity'].sum()
resources_quantity_total_df = resources_quantity_total_df.rename(columns = {'quantity': 'resources_quantity_total'})
resources_count_df = resources_df.groupby(['id'],as_index=False)['quantity'].count()
resources_count_df = resources_count_df.rename(columns = {'quantity': 'resources_variety_count'})
resources_money_df = resources_price_total_df.merge(resources_quantity_total_df)
resources_money_df = resources_money_df.merge(resources_count_df)
resources_money_df['resources_price_ave'] = resources_money_df['resource_price_total'] / resources_money_df['resources_quantity_total']
Full_df = Full_df.merge(resources_money_df)

# Full_df['resource_price_total'] = round(Full_df['resource_price_total'] / 50).clip(0,20).astype(int)
# Full_df['resources_quantity_total'] = round(Full_df['resources_quantity_total'] / 4).clip(0,15).astype(int)
# Full_df['resources_price_ave'] = round(Full_df['resources_price_ave'] / 20).clip(0,16).astype(int)
# Full_df['resources_variety_count'] = round(Full_df['resources_variety_count']).clip(0,25).astype(int)
print('Finish Preprocessing resources.')

# l. description - description of the resource requested
resources_df['all_resources_description'] = resources_df['description'].astype(str)
resources_description_df = resources_df.groupby('id')['all_resources_description'].apply(lambda x: '. '.join(x)).reset_index()
Full_df = Full_df.merge(resources_description_df)
del [resources_df, resources_price_total_df, resources_quantity_total_df, resources_count_df, resources_money_df]
gc.collect()

# i1. text length
textColumnList = ['project_title', 'project_essay_full', 'project_resource_summary', 'all_resources_description']
for textColumn in tqdm(textColumnList):
    Full_df[textColumn+'_len'] = Full_df[textColumn].map(lambda x: len(str(x)))
# Full_df['project_title_len'] = round(Full_df['project_title' + '_len'] / 5).clip(2,15).astype(int)
# Full_df['project_essay_full_len'] = round(Full_df['project_essay_full' + '_len'] / 40).clip(25,55).astype(int)
# Full_df['project_resource_summary_len'] = round(Full_df['project_resource_summary' + '_len'] / 10).clip(5,24).astype(int)
# Full_df['all_resources_description_len'] = round(Full_df['all_resources_description' + '_len'] / 40).clip(1,24).astype(int)

# i2. text sentiments
from textblob import TextBlob

def get_polarity(text):
    textblob = TextBlob(text)
    pol = textblob.sentiment.polarity
    return round(pol,2)

def get_subjectivity(text):
    textblob = TextBlob(text)
    subj = textblob.sentiment.subjectivity
    return round(subj,2)

Full_df['project_essay_full' + '_sent_polarity'] = Full_df['project_essay_full'].apply(get_polarity)
# Full_df['project_essay_full' + '_sent_polarity'] = (round(Full_df['project_essay_full' + '_sent_polarity'],1)).clip(0,0.5)
Full_df['project_essay_full' + '_sent_subjectivity'] = Full_df['project_essay_full'].apply(get_subjectivity)
# Full_df['project_essay_full' + '_sent_subjectivity'] = (round(Full_df['project_essay_full' + '_sent_subjectivity'],1)).clip(0.2,0.8)

from sklearn.feature_extraction.text import TfidfVectorizer
n_features = [400,5000,400,400]

for c_i, textColumn in tqdm(enumerate(textColumnList)):
    tfidf = TfidfVectorizer(max_features=n_features[c_i], min_df=3)

    features_df = pd.DataFrame(tfidf.fit_transform(Full_df[textColumn].astype(str)).toarray())
    features_df.columns = [textColumn + '_' + str(i) for i in range(min(n_features[c_i], features_df.shape[1]))]
    Full_df = pd.concat((Full_df, pd.DataFrame(features_df)), axis=1, ignore_index=False).reset_index(drop=True)

    del tfidf, features_df
    gc.collect()

print('Finish Preprocessing text.')


# 2.5 Converting - format data type --------------------------------------------
colomnsToBeCoded = ['school_state','teacher_prefix', 'project_grade_category']
Full_df = pd.get_dummies(Full_df, columns=colomnsToBeCoded)

print('Finish engineering features')

# # 2.6 Check data----------------------------------------------------------------
varRemoved = [
         'project_resource_summary',
         'project_subject_categories',
         'project_subject_subcategories',
         'project_submitted_datetime',
         'project_title',
         'teacher_id',
         'project_essay_full',
         'teacher_projects_approved_sum',
         'teacher_projects_count',
         'all_resources_description']

Full_df = Full_df.drop(columns = varRemoved)

# Full_df.to_csv('Full_df_ver3_withText_noClip.csv',index=False)

# 4. Building Models -----------------------------------------------------------
# Full_df = pd.read_csv('./Full_df_ver3_noClip.csv', low_memory=True)
print('Start modelling')

SEED = 2018 # for reproducibility
scoringMethod = 'roc_auc'
Predict_df = Full_df[pd.isna(Full_df['project_is_approved'])]
Full_df = Full_df[pd.notna(Full_df['project_is_approved'])]
Full_df['project_is_approved'] = Full_df['project_is_approved'].astype(int)

varCoded = list(Full_df)
varCoded.remove('project_is_approved')
varCoded.remove('id')

Train_df = Full_df
gc.collect()

# from sklearn.model_selection import train_test_split
# Train_df, Test_df = train_test_split(Full_df, test_size=0.3, random_state=SEED)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, RepeatedKFold
import lightgbm as lgb

X = Train_df[varCoded]
y = Train_df['project_is_approved']

# X_test = Test_df[varCoded]
# y_test = Test_df['project_is_approved']
# del Train_df, Test_df

X_test = Predict_df[varCoded]
id_test = Predict_df['id'].values
del Train_df, Predict_df

feature_names = varCoded
gc.collect()

# Build the model
cnt = 0
p_buf = []
n_splits = 5
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=0)

for train_index, valid_index in kf.split(X):
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 16,
        'num_leaves': 31,
        'learning_rate': 0.025,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 1,
        'lambda_l2': 1,
        'min_gain_to_split': 0,
    }


    model = lgb.train(
        params,
        lgb.Dataset(X.loc[train_index], y.loc[train_index], feature_name=feature_names),
        num_boost_round=10000,
        valid_sets=[lgb.Dataset(X.loc[valid_index], y.loc[valid_index])],
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    p = model.predict(X.loc[valid_index], num_iteration=model.best_iteration)

    p = model.predict(X_test, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p)
    else:
        p_buf += np.array(p)

    cnt += 1
    # if cnt > 0: # Comment this to run several folds
    #     break

    del model
    gc.collect

    print(str(cnt) + ' Done')
    print('-' * 50)

print('Training Done!')

preds = p_buf/cnt
print('Pred mean: ' + str(preds.mean()))

# roc_auc_score(y_test, preds)


# Submission -------------------------------------------------------------------
subm = pd.DataFrame()
subm['id'] = id_test
subm['project_is_approved'] = preds
subm.to_csv('submission_lightGBM_3_with_text_no_clipping.csv', index=False)
print('Exportion Done!')

print("Notebook took %0.2f minutes to Run"%((time.time() - start)/60))