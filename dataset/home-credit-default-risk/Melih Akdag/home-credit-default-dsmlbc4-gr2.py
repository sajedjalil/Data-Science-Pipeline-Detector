######################################
# HOME CREDIT DEFAULT RISK COMPETITION
######################################

# Importing essential libraries
import numpy as np
import pandas as pd
import time
import gc
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=FutureWarning)


# Defining timer to track progress
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# Defining one-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Defining Sin-cos transformation for cyclic features
def encode(df, col, max_val):
    df[col + '_SIN'] = np.sin(2 * np.pi * df[col]/max_val)
    df[col + '_COS'] = np.cos(2 * np.pi * df[col]/max_val)
    return df


# Defining dynamic rare encoding for column categories
def dyn_rare_encoder(df, columns, rare_percent):
    for col in columns:
        tmp = df[col].value_counts() / len(df) * 100
        rare_labels = tmp[tmp < rare_percent].index
        df[col] = np.where(df[col].isin(rare_labels), 'Other', df[col])
    return df


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

    

#####################################
# Application Train and Test Data
#####################################
def application_train_test(num_rows=None, nan_as_category=True):
    # Read and merge data
    df = pd.read_csv('../input/home-credit-default-risk/application_train.csv', nrows=num_rows)
    test_df = pd.read_csv('../input/home-credit-default-risk/application_test.csv', nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()

    # Removing 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    # Deleting FLAG_MOBIL because there is only 1 person without mobile phone
    df.drop('FLAG_MOBIL', axis=1, inplace=True)
    df.drop('FLAG_CONT_MOBILE', axis=1, inplace=True)

    # NaN values for DAYS_EMPLOYED: 365243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # Changing rare categories of NAME_INCOME_TYPE with the similar categories
    df.loc[df['NAME_INCOME_TYPE'] == 'Student', 'NAME_INCOME_TYPE'] = 'State servant'
    df.loc[df['NAME_INCOME_TYPE'] == 'Maternity leave', 'NAME_INCOME_TYPE'] = 'Pensioner'
    df.loc[df['NAME_INCOME_TYPE'] == 'Unemployed', 'NAME_INCOME_TYPE'] = 'Pensioner'
    df.loc[df['NAME_INCOME_TYPE'] == 'Businessman', 'NAME_INCOME_TYPE'] = 'Commercial associate'

    # Dynamic rare encoding
    df = dyn_rare_encoder(df, ['ORGANIZATION_TYPE'], rare_percent=1.9)
    df = dyn_rare_encoder(df, ['NAME_TYPE_SUITE'], rare_percent=3.6)
    df = dyn_rare_encoder(df, ['OCCUPATION_TYPE'], rare_percent=1.5)
    df = dyn_rare_encoder(df, ['WALLSMATERIAL_MODE'], rare_percent=20)

    # Rare Encoding NAME_HOUSING_TYPE with 'Other'
    df.loc[(df['NAME_HOUSING_TYPE'] == 'Office apartment') &
           (df['NAME_HOUSING_TYPE'] == 'Co-op apartment'), 'NAME_HOUSING_TYPE'] = 'Other'

    #  Changing unknown family status with the most observed category
    df['NAME_FAMILY_STATUS'].replace('Unknown', 'Married', inplace=True)

    #  Changing HOUSETYPE_MODE not null values with
    df.loc[df['HOUSETYPE_MODE'].notnull(), 'HOUSETYPE_MODE'] = 'house_type_reported'

    # Changing weekdays with integer values
    weekday_dict = {'MONDAY': 1, 'TUESDAY': 2, 'WEDNESDAY': 3, 'THURSDAY': 4, 'FRIDAY': 5, 'SATURDAY': 6, 'SUNDAY': 7}
    df.replace({'WEEKDAY_APPR_PROCESS_START': weekday_dict}, inplace=True)
    # Creating sin-cos transformed features
    df = encode(df, 'WEEKDAY_APPR_PROCESS_START', 7)
    df = encode(df, 'HOUR_APPR_PROCESS_START', 23)
    # Deleting initial WEEKDAY_APPR_PROCESS_START and HOUR_APPR_PROCESS_START features
    df.drop(['WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START'], axis=1, inplace=True)

    # New features (percentages)
    df['NEW_DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['NEW_INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['NEW_ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['NEW_ANNUITY_CREDIT_RATIO'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # Loan to Value Ratio (LVR)
    df['NEW_LVR'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']

    # LVR_RISK assesment feature
    df.loc[df['NEW_LVR'] >= 0.80, 'NEW_LVR_RISK'] = 1
    df.loc[df['NEW_LVR'] < 0.80, 'NEW_LVR_RISK'] = 0

    # Mean of External Sources
    df["NEW_EXT_MEAN"] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)

    # Product of External Sources
    df['NEW_EXT_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

    # Ages of customers
    df['NEW_AGE'] = df['DAYS_BIRTH'] / -365

    # NEW_AGE_SEGMENT segments
    df.loc[df['NEW_AGE'] <= 34, 'NEW_AGE_SEGMENT'] = 'AGE_GRP_1'
    df.loc[(df['NEW_AGE'] > 34) & (df['NEW_AGE'] <= 54), 'NEW_AGE_SEGMENT'] = 'AGE_GRP_2'
    df.loc[df['NEW_AGE'] > 54, 'NEW_AGE_SEGMENT'] = 'AGE_GRP_3'

    # Total documents demonstrated
    df['NEW_TOTAL_DOC_NUM'] = df.loc[:, 'FLAG_DOCUMENT_2':'FLAG_DOCUMENT_21'].sum(axis=1)
    df.drop(df.loc[:, 'FLAG_DOCUMENT_2':'FLAG_DOCUMENT_21'], axis=1, inplace=True)

    # Product-Credit-Salary relation
    df["NEW_PROD_CRED_SALARY"] = (df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]) / df["AMT_INCOME_TOTAL"]

    # NEW_ACCOMPANIED feature
    df.loc[df['NAME_TYPE_SUITE'] == 'Unaccompanied', 'NEW_ACCOMPANIED'] = 0
    df.loc[df['NAME_TYPE_SUITE'] != 'Unaccompanied', 'NEW_ACCOMPANIED'] = 1
    df.loc[df['NAME_TYPE_SUITE'].isnull(), 'NEW_ACCOMPANIED'] = np.nan

    # Social circle with both 30 and 60 days default (binary)
    df.loc[(df['DEF_30_CNT_SOCIAL_CIRCLE'] > 0) & (df['DEF_60_CNT_SOCIAL_CIRCLE'] > 0),
           'NEW_DEF_30&60_SOCIAL_CIRCLE'] = 1
    df.loc[(df['DEF_30_CNT_SOCIAL_CIRCLE'] == 0) & (df['DEF_60_CNT_SOCIAL_CIRCLE'] == 0),
           'NEW_DEF_30&60_SOCIAL_CIRCLE'] = 0

    # Label encoding
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # Dropping feature named index
    df.drop('index', axis=1, inplace=True)

    del test_df
    gc.collect()
    return df


#####################################
# Bureau Data
#####################################
def bb__agg(num_rows = None, nan_as_category = True):
    bb = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv', nrows = num_rows)
    
    # DPD (Days Past Due) 'ye düşmüşmü, düşmemişmi?'
    liste = ['1','2','3','4','5']
    bb['NEW_FLAG'] = bb['STATUS'].apply(lambda x : 1 if (x in liste) else ("X" if x == "X" else 0))
    
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bb.drop("NEW_FLAG_X", inplace=True, axis = 1)
    bb_cat.remove('NEW_FLAG_X')
    
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    
    del bb
    gc.collect()
    return  bb_agg, bb_cat


def bureau_(num_rows = None, nan_as_category = True):
    bu = pd.read_csv('../input/home-credit-default-risk/bureau.csv', nrows = num_rows)
    
    # Kredi Aktive ve Closed toplam Sayılarını ve Oranlarını hesaplamak
    temp_bu = bu[['SK_ID_CURR', 'CREDIT_ACTIVE']]
    temp_bu = pd.get_dummies(temp_bu)
    temp_bu = temp_bu.groupby('SK_ID_CURR').agg({'CREDIT_ACTIVE_Active':'sum','CREDIT_ACTIVE_Closed':'sum' })
    temp_bu.columns = ['CREDIT_ACTIVE_Active_Count','CREDIT_ACTIVE_Closed_Count']
    temp_bu['CREDIT_ACTIVE_Active_ratio'] = temp_bu['CREDIT_ACTIVE_Active_Count'] / (temp_bu['CREDIT_ACTIVE_Active_Count'] + temp_bu['CREDIT_ACTIVE_Closed_Count'])
    temp_bu['CREDIT_ACTIVE_Closed_ratio'] = temp_bu['CREDIT_ACTIVE_Closed_Count'] / (temp_bu['CREDIT_ACTIVE_Active_Count'] + temp_bu['CREDIT_ACTIVE_Closed_Count'])
    bu = bu.merge(temp_bu, on=['SK_ID_CURR'], how='left')
    
    # Kredi DAYS_CREDIT'i SK_ID_CURR bazında sıralayarak NEW_DAYS_DIFF değişkeni üretmek kredi alma frekansı bilgisi verebilir.
    temp_bu = bu[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].groupby(by = ['SK_ID_CURR']).apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending = True)).reset_index(drop = True)
    temp_bu['NEW_DAYS_DIFF'] = temp_bu.groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT'].diff()
    temp_bu = temp_bu[['SK_ID_BUREAU','NEW_DAYS_DIFF']]
    temp_bu['NEW_DAYS_DIFF'] = temp_bu['NEW_DAYS_DIFF'].fillna(0)
    bu = bu.merge(temp_bu, on=['SK_ID_BUREAU'], how='left')
    
    # Active ve Closed Krediler için kredi erken kapanmışmı? 
    bu.loc[(bu['CREDIT_ACTIVE'] == 'Active') & (bu['DAYS_CREDIT_ENDDATE'] < 0), 'NEW_EARLY_ACTİVE'] = 1
    bu.loc[(bu['CREDIT_ACTIVE'] == 'Closed') & (abs(bu['DAYS_CREDIT_ENDDATE']) < abs(bu['DAYS_ENDDATE_FACT']) ), 'NEW_EARLY_CLOSED'] = 1
    
    # Uzatılmış Kredilerin 1 ile değiştirilmesi
    prolong = [1,2,3,4,5,6,7,8,9]
    bu['CNT_CREDIT_PROLONG'].replace(prolong, 1, inplace= True)
    
    # Kişi Kaç farklı kredi tipi almış
    temp_bu = bu[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by=['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(index=str, columns={'CREDIT_TYPE': 'NEW_BUREAU_LOAN_TYPES'})
    bu = bu.merge(temp_bu, on=['SK_ID_CURR'], how='left')
    
    # Borç Oranı
    bu['NEW_DEPT_RATİO'] = bu['AMT_CREDIT_SUM_DEBT'] / (bu['AMT_CREDIT_SUM']+1)
    
    # Kredi Tiplerinin 'others' ile değiştirilmesi
    credit_type = ['Loan for working capital replenishment',
       'Loan for business development', 'Real estate loan',
       'Unknown type of loan', 'Another type of loan',
       'Cash loan (non-earmarked)', 'Loan for the purchase of equipment',
       'Mobile operator loan', 'Interbank credit',
       'Loan for purchase of shares (margin lending)']
       
    bu['CREDIT_TYPE'].replace(credit_type, 'others', inplace= True)
    
    # Aylık Ödeme Oranı
    bu['NEW_AMT_ANNUITY_RATİO'] = bu['AMT_ANNUITY'] / bu['AMT_CREDIT_SUM']
    
    # Kredi güncellenmesi yenimi ?
    bu['NEWS_DAYS_CREDIT_UPDATE'] = bu['DAYS_CREDIT_UPDATE'].apply(lambda x : 'old' if x < -90 else 'new')
    
    # 'CREDIT_CURRENCY' değişkenini düşürmek 
    bu.drop('CREDIT_CURRENCY',inplace=True,axis = 1)
    
    del temp_bu
    gc.collect()
    return bu


def combine(bureau,bb_agg):
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    return bureau


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bb_agg, bb_cat = bb__agg(num_rows,nan_as_category)
    bureau = bureau_(num_rows,nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau,nan_as_category)
    bureau = combine(bureau,bb_agg)
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        "CREDIT_ACTIVE_Active_Count":["mean"],
        "CREDIT_ACTIVE_Closed_Count":["mean"],
        "CREDIT_ACTIVE_Active_ratio":["mean"],
        "NEW_DAYS_DIFF":['max', 'mean'],
        "NEW_EARLY_ACTİVE":['mean'],
        "NEW_EARLY_CLOSED":['mean'],
        "NEW_BUREAU_LOAN_TYPES":['mean'],
        "NEW_DEPT_RATİO":['max', 'mean'],
        "NEW_AMT_ANNUITY_RATİO":['max', 'mean']
        }
    
    for col in bb_cat:
        num_aggregations[col + "_MEAN"] = ['mean']
    
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    
    del active, active_agg
    gc.collect()
    
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    
    del closed, closed_agg, bureau,bb_agg
    gc.collect()
    return bureau_agg


#####################################
# Previous Application Data
#####################################
def previous_app(num_rows = None, nan_as_category=True):
    df_prev = pd.read_csv('../input/home-credit-default-risk/previous_application.csv', nrows = num_rows)
    cat_cols = [col for col in df_prev.columns if df_prev[col].dtypes == 'O']
    num_cols = [col for col in df_prev.columns if df_prev[col].dtypes != 'O']

    # days 365243 values to nan
    df_prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    df_prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    df_prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    df_prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    df_prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    # XNA, XAP to nan for cat_cols.
    na = ['XNA', 'XAP']
    for col in cat_cols:
        for n in na:
            df_prev.loc[df_prev[col] == n, col] = np.nan

    # delete columns columns that do not contain information or missing values over 80 percent of the entire data
    del_cols = ['RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'DAYS_FIRST_DRAWING',
                'NAME_CASH_LOAN_PURPOSE', 'CODE_REJECT_REASON', 'FLAG_LAST_APPL_PER_CONTRACT',
                'NFLAG_LAST_APPL_IN_DAY', 'SELLERPLACE_AREA']
    df_prev.drop(del_cols, axis=1, inplace=True)

    # Feature Engineering
    # X-sell approved & Walk-in Approved
    df_prev['NEW_X_SELL_APPROVED'] = 0
    df_prev.loc[(df_prev['NAME_PRODUCT_TYPE'] == 'x-sell') &
                (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_X_SELL_APPROVED'] = 1
    df_prev['NEW_WALK_IN_APPROVED'] = 0
    df_prev.loc[(df_prev['NAME_PRODUCT_TYPE'] == 'walk-in') &
                (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_WALK_IN_APPROVED'] = 1

    # Customer status approved
    df_prev['NEW_REPEATER_APPROVED'] = 0
    df_prev.loc[(df_prev['NAME_CLIENT_TYPE'] == 'Repeater') &
                (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_REPEATER_APPROVED'] = 1
    df_prev['NEW_NEWCUST_APPROVED'] = 0
    df_prev.loc[(df_prev['NAME_CLIENT_TYPE'] == 'New') &
                (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_NEWCUST_APPROVED'] = 1
    df_prev['NEW_REFRESHED_APPROVED'] = 0
    df_prev.loc[(df_prev['NAME_CLIENT_TYPE'] == 'Refreshed') &
                (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_REFRESHED_APPROVED'] = 1

    # Purpose of application approved
    df_prev['NEW_CARDS_APPROVED'] = 0
    df_prev.loc[(df_prev['NAME_PORTFOLIO'] == 'Cards') &
                (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_CARDS_APPROVED'] = 1
    df_prev['NEW_CASH_APPROVED'] = 0
    df_prev.loc[(df_prev['NAME_PORTFOLIO'] == 'Cash') &
                (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_CASH_APPROVED'] = 1
    df_prev['NEW_POS_APPROVED'] = 0
    df_prev.loc[(df_prev['NAME_PORTFOLIO'] == 'POS') &
                (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_POS_APPROVED'] = 1

    # Interest approved
    df_prev['NEW_HIGH_APPROVED'] = 0
    df_prev.loc[(df_prev['NAME_YIELD_GROUP'] == 'high') &
                (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_HIGH_APPROVED'] = 1
    df_prev['NEW_MIDDLE_APPROVED'] = 0
    df_prev.loc[(df_prev['NAME_YIELD_GROUP'] == 'middle') &
                (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_MIDDLE_APPROVED'] = 1
    df_prev['NEW_LOWACTION_APPROVED'] = 0
    df_prev.loc[(df_prev['NAME_YIELD_GROUP'] == 'low_action') &
                (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_LOWACTION_APPROVED'] = 1
    df_prev['NEW_LOWNORMAL_APPROVED'] = 0
    df_prev.loc[(df_prev['NAME_YIELD_GROUP'] == 'low_normal') &
                (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_LOWNORMAL_APPROVED'] = 1

    # Application hour convert to categorical
    df_prev.loc[(df_prev['HOUR_APPR_PROCESS_START'] >= 0) &
                (df_prev['HOUR_APPR_PROCESS_START'] <= 6), 'NEW_APP_DAY_TIME'] = 'night'
    df_prev.loc[(df_prev['HOUR_APPR_PROCESS_START'] > 6) &
                (df_prev['HOUR_APPR_PROCESS_START'] <= 12), 'NEW_APP_DAY_TIME'] = 'morning'
    df_prev.loc[(df_prev['HOUR_APPR_PROCESS_START'] > 12) &
                (df_prev['HOUR_APPR_PROCESS_START'] <= 18), 'NEW_APP_DAY_TIME'] = 'afternoon'
    df_prev.loc[(df_prev['HOUR_APPR_PROCESS_START'] > 18) &
                (df_prev['HOUR_APPR_PROCESS_START'] < 24), 'NEW_APP_DAY_TIME'] = 'evening'
    df_prev.drop('HOUR_APPR_PROCESS_START', axis=1, inplace=True)

    # Client apply with someone
    df_prev.loc[df_prev['NAME_TYPE_SUITE'] == 'Unaccompanied', 'NEW_ACCOMPANIED'] = 0
    df_prev.loc[df_prev['NAME_TYPE_SUITE'] != 'Unaccompanied', 'NEW_ACCOMPANIED'] = 1
    df_prev.loc[df_prev['NAME_TYPE_SUITE'].isnull(), 'NEW_ACCOMPANIED'] = np.nan
    df_prev.drop('NAME_TYPE_SUITE', axis=1, inplace=True)

    # credit requested / credit given ratio
    df_prev['NEW_APP_CREDIT_RATIO'] = df_prev['AMT_APPLICATION'].div(df_prev['AMT_CREDIT']).replace(np.inf, 0)
    # loan installment / credit amount ratio
    df_prev['NEW_ANNUITY_CREDIT_RATIO'] = df_prev['AMT_ANNUITY'] / df_prev['AMT_CREDIT']
    # credit amount / goods price ratio
    df_prev['NEW_CREDIT_GOODS_RATIO'] = df_prev['AMT_CREDIT'].div(df_prev['AMT_GOODS_PRICE']).replace(np.inf, 0)
    # interest amount
    df_prev['NEW_AMT_INTEREST'] = df_prev['CNT_PAYMENT'] * df_prev['AMT_ANNUITY'] - df_prev['AMT_CREDIT']
    # interest ratio
    df_prev['NEW_INTEREST_RATIO'] = df_prev['NEW_AMT_INTEREST'] / df_prev['AMT_CREDIT']
    # needed amount / credit amount (belki silinir)
    df_prev['NEW_AMT_NEEDED_CREDIT_RATIO'] = (df_prev['AMT_GOODS_PRICE'] - df_prev['AMT_DOWN_PAYMENT']) / \
                                             df_prev['AMT_CREDIT']

    # risk assessment via NEW_CREDIT_GOODS_RATIO
    df_prev.loc[df_prev['NEW_CREDIT_GOODS_RATIO'] >= 0.80, 'NEW_CREDIT_GOODS_RISK'] = 1
    df_prev.loc[df_prev['NEW_CREDIT_GOODS_RATIO'] < 0.80, 'NEW_CREDIT_GOODS_RISK'] = 0

    # risk to approved
    df_prev['NEW_RISK_APPROVED'] = 0
    df_prev.loc[(df_prev['NEW_CREDIT_GOODS_RISK'] == 1) &
                (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_RISK_APPROVED'] = 1

    # non risk to approved
    df_prev['NEW_NONRISK_APPROVED'] = 0
    df_prev.loc[(df_prev['NEW_CREDIT_GOODS_RISK'] == 0) &
                (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_NONRISK_APPROVED'] = 1

    # Application weekdays cycle encoding
    df_prev['WEEKDAY_APPR_PROCESS_START'] = df_prev['WEEKDAY_APPR_PROCESS_START'].map({
        'MONDAY': 1, 'TUESDAY': 2, 'WEDNESDAY': 3, 'THURSDAY': 4, 'FRIDAY': 5, 'SATURDAY': 6, 'SUNDAY': 7})
    df_prev['NEW_WEEKDAY_SIN'] = np.sin(2 * np.pi * df_prev['WEEKDAY_APPR_PROCESS_START'] / 7)
    df_prev['NEW_WEEKDAY_COS'] = np.cos(2 * np.pi * df_prev['WEEKDAY_APPR_PROCESS_START'] / 7)
    df_prev.drop('WEEKDAY_APPR_PROCESS_START', axis=1, inplace=True)

    # Rare encoding
    a = ['Auto Accessories', 'Jewelry', 'Homewares', 'Medical Supplies', 'Vehicles', 'Sport and Leisure',
         'Gardening', 'Other', 'Office Appliances', 'Tourism', 'Medicine', 'Direct Sales', 'Fitness',
         'Additional Service', 'Education', 'Weapon', 'Insurance', 'House Construction', 'Animals']
    df_prev["NAME_GOODS_CATEGORY"] = df_prev["NAME_GOODS_CATEGORY"].replace(a, 'others')

    b = ['Channel of corporate sales', 'Car dealer']
    df_prev["CHANNEL_TYPE"] = df_prev["CHANNEL_TYPE"].replace(b, 'Other_Channel')

    c = ['Auto technology', 'Jewelry', 'MLM partners', 'Tourism']
    df_prev["NAME_SELLER_INDUSTRY"] = df_prev["NAME_SELLER_INDUSTRY"].replace(c, 'Others')

    d = ['Non-cash from your account', 'Cashless from the account of the employer']
    df_prev["NAME_PAYMENT_TYPE"] = df_prev["NAME_SELLER_INDUSTRY"].replace(d, 'Others')

    # One hot encoder
    new_df_prev, new_cat_cols = one_hot_encoder(df_prev, nan_as_category)

    # Getting to all the cat cols
    origin_bin_cols = [col for col in df_prev.columns if (df_prev[col].dtypes != 'O') & (df_prev[col].nunique() == 2)]
    all_cat_cols = new_cat_cols + origin_bin_cols

    # Getting to the num cols
    # x_cols = ['SK_ID_PREV','SK_ID_CURR', 'DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE','DAYS_TERMINATION']
    # new_num_cols = [col for col in new_df.columns if (col not in all_binary_cols) and (col not in x_cols)]
    # num_aggregations = {}
    # for num in new_num_cols:
    # num_aggregations[num] = ['min', 'max', 'mean', 'median']

    # Previous app num features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean', 'median'],
        'AMT_APPLICATION': ['min', 'max', 'mean', 'median'],
        'AMT_CREDIT': ['min', 'max', 'mean', 'median'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean', 'median'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean', 'median'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean', 'median'],
        'DAYS_DECISION': ['min', 'max', 'mean', 'median'],
        'CNT_PAYMENT': ['min', 'max', 'mean', 'median'],
        'NEW_APP_CREDIT_RATIO': ['min', 'max', 'mean', 'median'],
        'NEW_ANNUITY_CREDIT_RATIO': ['min', 'max', 'mean', 'median'],
        'NEW_CREDIT_GOODS_RATIO': ['min', 'max', 'mean', 'median'],
        'NEW_AMT_INTEREST': ['min', 'max', 'mean', 'median'],
        'NEW_INTEREST_RATIO': ['min', 'max', 'mean', 'median'],
        'NEW_AMT_NEEDED_CREDIT_RATIO': ['min', 'max', 'mean', 'median'],
        'NEW_WEEKDAY_SIN': ['min', 'max', 'mean', 'median'],
        'NEW_WEEKDAY_COS': ['min', 'max', 'mean', 'median']}

    # Previous app cat features
    cat_aggregations = {}
    for cat in all_cat_cols:
        cat_aggregations[cat] = ['mean']

    final_prev_df = new_df_prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    final_prev_df.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in final_prev_df.columns.tolist()])

    # Approved App - only num features
    approved = new_df_prev[new_df_prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(
        ['PREV_APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    final_prev_df = final_prev_df.join(approved_agg, how='left', on='SK_ID_CURR')

    # refused App - only numerical features
    refused = new_df_prev[new_df_prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['PREV_REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    final_prev_df = final_prev_df.join(refused_agg, how='left', on='SK_ID_CURR')

    del refused, refused_agg, approved, approved_agg, new_df_prev
    gc.collect()
    return final_prev_df



#####################################
# POS_CASH_balance Data
#####################################
def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv', nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    
    del pos
    gc.collect()
    return pos_agg



#####################################
# Installments_payments Data
#####################################
def installments_payments(num_rows=None, nan_as_category=True):
    ins = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv', nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']}
    
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    
    del ins
    gc.collect()
    return ins_agg



#####################################
# Credit_card_balance Data
#####################################
def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv', nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    
    del cc
    gc.collect()
    return cc_agg



#####################################
# LightGBM GBDT with KFold or Stratified KFold
#####################################
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified=False, debug=False):
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    
    del df
    gc.collect()
    
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
        
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1)

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=200, early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index=False)
    display_importances(feature_importance_df)
    return feature_importance_df



#####################################
# Main Function
#####################################
def main(debug = False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_app(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        # Exporting combined_df to investigate features
        df.to_csv('combined_df.csv')
        del cc
        gc.collect()
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds= 10, stratified= False, debug= debug)

        
if __name__ == "__main__":
    submission_file_name = "submission_DSMLBC4_Grp2.csv"
    with timer("Full model run"):
        main(debug = True)
