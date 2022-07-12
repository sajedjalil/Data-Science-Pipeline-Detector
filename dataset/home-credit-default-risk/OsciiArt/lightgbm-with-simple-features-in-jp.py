# this is japanese-translated folk of [Updated 0.792 LB] LightGBM with Simple Features
# additionally, I publish japanese-translated version of HomeCredit_columns_description.csv (under construction)
# https://docs.google.com/spreadsheets/d/1EipnrrXDGkESEH-_53T0TOHAtNwFZXjzefmamToxoiY/edit?usp=sharing
# HOME CREDIT DEFAULT RISK COMPETITION
# Most features are created by applying min, max, mean, sum and var functions to grouped tables. 
# Little feature selection is done and overfitting might be a problem since many features are related.
# The following key ideas were used:
# - Divide or subtract important features to get rates (like annuity and income)
# - In Bureau Data: create specific features for Active credits and Closed credits
# - In Previous Applications: create specific features for Approved and Refused applications
# - Modularity: one function for each table (except bureau_balance and application_test)
# - One-hot encoding for categorical features
# All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).
# You can use LightGBM with KFold or Stratified KFold. Please upvote if you find usefull, thanks!

# Update 16/06/2018:
# - Added Payment Rate feature
# - Removed index from features
# - Set early stopping to 200 rounds
# - Use standard KFold CV (not stratified)
# Public LB increased to 0.792

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
コンテキストマネージャ
ジェネレータを @contextlib.contextmanager デコレータで修飾すると、それがコンテキストマネージャになる.
with timer
    関数f()
で実行するとyieldの部分で関数fが実行される.
ここではwith timer で関数を実行すると所要時間が出力される.
"""
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    """
    データフレームの内，型がobjectの列をone hot 化.
    引数
    df: データフレーム
    出力
    df: カテゴリ変数をone hot 化ｓたデータフレーム
    new_columns: 新しく作成した列の名前
    """
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    # appのtrainとtestを読み込んで結合
    df = pd.read_csv('../input/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('../input/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    # カテゴリ変数 (性別, 車所有, 不動産所有) を01に変換
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    # 勤続日数 (DAYS_EMPLOYED) の365243をnanに置換. 元々nanがこの値で埋められている.
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'] # 勤続日数/年齢日数
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT'] # 総収入/借入額
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'] # 総収入/家族人数
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'] # 月々の返済額/総収入
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'] # 月々の返済額/借入額
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('../input/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('../input/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    """
    df.grouby('列名c'): 
        データをグループ分けする. 
        列名cが同じ行をグループにしてデータを集計して、それぞれの平均、最小値、最大値、合計などの統計量を算出することが可能.
    df.groupby('列名C').agg(['min', 'max']:
        グループに対して統計量を算出
    """
    # bbの各列のグループに対しどの統計量を計算するかの指定を行う.
    # 月返済額 (MONTHS_BALANCE): min, max, size(グループのサンプル数)
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat: # one hot化したカテゴリ変数全て: mean
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations) # CB申請ID (CBに対する一度の借金申請ごとにIDが振られている)ごとでグループ化
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()]) # 新規列名の命名
    # df1.join(df2, how='left', on='列名c'): df1にdf2を結合. 列名cが一致する行を結合する.
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU') # bbの集計した統計量をCB申請IDでひも付けてbureauに結合
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True) # CB申請IDを列から削除
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    # 数値特徴量に対して計算する統計量を指定
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'], # CBクレジットが現在申請 (application_{train|test}.csv) の何日前に申請されたか
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'], # CBクレジットの返済までの残り日数 (現在申請日時点)
        'DAYS_CREDIT_UPDATE': ['mean'], # CBクレジットの情報の最終更新日 (現在申請日の何日前か)
        'CREDIT_DAY_OVERDUE': ['max', 'mean'], # CBクレジットの延滞日数 (現在申請日時点)
        'AMT_CREDIT_MAX_OVERDUE': ['mean'], # CBクレジットのこれまでで最大の延滞日数 (現在申請日時点)
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'], # CBクレジットの現在の借入額
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'], # CBクレジットの現在の債務額. credit amount と debt の違いが分からない
        'AMT_CREDIT_SUM_OVERDUE': ['mean'], # CBクレジットに対する現在の返済滞納額
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'], # CBクレジットに報告されているクレジットカードの現在の限度額
        'AMT_ANNUITY': ['max', 'mean'], # CBクレジットの月々の返済額 (利息を含む)
        'CNT_CREDIT_PROLONG': ['sum'], # CBクレジットが何回延長されたか
        'MONTHS_BALANCE_MIN': ['min'], # 月々の返済の最少額 (bbから集計された統計量)
        'MONTHS_BALANCE_MAX': ['max'], # 月々の返済の最大額 (bbから集計された統計量) 
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'] #  月々の返済のサンプル数 (bbから集計された統計量)
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {} # one hot化したカテゴリ変数全て: mean
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations}) # ローンID (applicationの債務者ID)ごとでグループ化
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()]) # 新規列名の命名
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1] # active: 返済が終了していない借金の情報だけ抜き出す
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations) # activeについて数値統計量のみ集計
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR') # activeをbureauに結合
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1] # closed: 返済が終了している借金の情報だけ抜き出す
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR') # closedをbureauに結合
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('../input/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage金額
    # AMT_APPLICATIONは顧客が申請した借金額であるが、AMT_CREDITは実際に支払われた借金額である。したがって審査の過程で金額が申請額から変わっている場合がある。
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT'] # 申請した金額/実際に貸した借金額
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'], # 月々の返済額 (利息を含む)
        'AMT_APPLICATION': ['min', 'max', 'mean'], # 顧客の申請した借金額
        'AMT_CREDIT': ['min', 'max', 'mean'], # 実際に貸した借金額
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'], # 申請した金額/実際に貸した借金額
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'], # 前払金
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'], # ローンが組まれた対象商品の価格 (消費者ローンの場合)
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'], # 顧客が借入申請を何時にしたか (四捨五入済)
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'], # 前払金の割合 (正規化済み)
        'DAYS_DECISION': ['min', 'max', 'mean'], # 申請日に対して決定がなされた日 (現在申請日の何日前か)
        'CNT_PAYMENT': ['mean', 'sum'], # 返済期間 (返済にかかる月数)?
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations}) # ローンID (applicationの債務者ID)ごとでグループ化
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1] # approved: 受理された申請のみ抜き出す
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1] # refused: 却下された申請のみ抜き出す
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('../input/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'], # 月々の支払額
        'SK_DPD': ['max', 'mean'], # 月ごとの支払い遅延日数
        'SK_DPD_DEF': ['max', 'mean'] # 月ごとの支払い遅延日数の許容限度 (少額のローンでは無視される)?
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size() # ローンIDごとのPOS CASHのサンプルサイズ
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
# 分割支払いデータの前処理
# 過去借入の返済の一度の分割支払いごとの情報
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('../input/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT'] # 実際の分割支払額/計画上の分割支払額
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT'] #  実際の分割支払額-計画上の分割支払額
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT'] # 支払遅延日数 = 実際の分割支払日 - 予定上の分割支払日
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT'] # 早期支払日数 = 予定上の分割支払日 - 実際の分割支払日 
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0) # マイナスになってる方は0に置換
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'], # 分割支払いカレンダーのバージョン (0ならクレジットカード).  返済スケジュールの変更回数を表す?: nunique (ユニークな値の数), 返済スケジュールの変更回数を表す?
        'DPD': ['max', 'mean', 'sum'], # 支払遅延日数
        'DBD': ['max', 'mean', 'sum'], # 早期支払日数
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'], # 実際の分割支払額/計画上の分割支払額
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'], #  実際の分割支払額-計画上の分割支払額
        'AMT_INSTALMENT': ['max', 'mean', 'sum'], # 計画上の分割支払額
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'], # 実際の分割支払額
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'] #  実際の分割支払日 
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size() # 分割支払いのサンプルサイズ=分割支払回数
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('../input/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        # LGBMのパラメータはベイズ最適化により決定
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
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

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
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


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
        prev = previous_applications(num_rows)
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
        del cc
        gc.collect()
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds= 5, stratified= False, debug= debug)

if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()