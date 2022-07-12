# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import k_means
from xgboost import XGBRegressor, XGBClassifier

import datetime as dt
import math as math

CN_ECOLOGY_EXCELLENT = 5
CN_ECOLOGY_GOOD      = 4
CN_ECOLOGY_SATISFY   = 3
CN_ECOLOGY_POOR      = 2
CN_PL_DIV = 40

VN_MIN_PRICE = 500000
VN_MAX_PRICE = 27000000

vn_quant = 40

GN_BC_MIN_POW = -1
GN_BC_MAX_POW = 1
GN_BC_STEP = 0.5
GN_BC_MIN_CORR = 0.4
GN_BC_NICE = 0.7
GN_BC_MAX_ITEMS = 120

VN_MACRO_DAYS = 30

VB_MACRO_ALL = True
VB_CLUSTER = False
VB_CENTER = True
VB_USE_CC = True
VB_USE_MAX = True
VB_ITER_TRAIN = False
VB_SELF_EXTRACT = False
VB_SELF_BY_TAIL = True
lb_get_best_corr_seg = True
VB_ONLY_UNIQUE_PAIRS = False
VB_DROP_WEAK_PAIRS = False
VB_DOUBLE_AUTOCORR = False
VB_USE_CHEAP_TRAIN = False
VB_USE_CHEAP_CC = False
VB_LEARN_ERROR = False
VB_USE_CA = False
VN_CHEAP_INDEX = 0.6
VC_ERR_CTRL = 'rmsle' #'mea', 'rmsle', 'over'
VC_AUTOCORR_COLUMN = 'a_m1_p1' #'price_doc' # 
VC_XGB_EVAL_METRIC = 'rmse' #'logloss' 'rmse' 

VN_FULL_SQ_M = 1
VN_PLP_M = 1

VN_PL_CONST = 3.5

VN_LEARN_ERROR_RATE = 0.2

GN_FILL_NA_STATE = 2
GN_FILL_4_STATE = 4

VL_MAIN_CC = [ 'full_sq','sadovoe_km','metro_km_walk','park_km'
#,'university_top_20_raion'
,'a_big_roads'
       ,'school_education_centers_raion','a_child_km'
       #,'a_education'
       #,'ecology'
       ,'sport_objects_raion','university_km'
       ,'indust_part','a_bad_factors_km'
       #,'a_2000'
       ,'office_raion','shopping_centers_raion'
       #,'a_16_29'
       ,'floor','max_floor'
       ,'housing_fund_sqm','oil_urals','cpi','rts','usdrub','rent_price_1room_eco'
       #,'apartment_fund_sqm'
       ,'ppi','mortgage_rate'
       #,'salary','childbirth'
       ,'a_material','a_mmp','a_cr','a_pl_p'
       #,'area_id'
       ]

VL_MAIN_CC = list(pd.Series (VL_MAIN_CC).drop_duplicates())
print ('columns: ', len (VL_MAIN_CC))

VL_EXCLUDE_CC = ['id','timestamp','life_sq','price_doc','sub_area'
,'product_type','a_dt','a_day','#','a_sq','a_q','a_mm','a_kr','a_m1_p1','a_m1_p1rl'
,'a_cheap','a_expensive','apartment_fund_sqm'
#,'a_cr'
,'salary','childbirth','a_16_29','ecology'
,'a_pl','area_id','a_2000','university_top_20_raion','a_education'
,'child_on_acc_pre_school','modern_education_share','old_education_build_share'
#,'floor','max_floor','fitness_km','hospice_morgue_km','balance_trade','net_capital_export'
]

#V_CC_CHEAP = ['floor', 'max_floor', 'material', 'a_cr', 'state', 'build_year', 'a_qm', 'a_sq','sadovoe_km','mkad_km','a_pl_p','a_m']
V_CC_CHEAP = []

V_MACRO = [
'usdrub', 'oil_urals', 'cpi'
,'eurrub','housing_fund_sqm','rent_price_2room_bus'
,'ppi','mortgage_rate','salary','childbirth'
,'brent','rts','apartment_fund_sqm','rent_price_1room_eco'
#14.06:
,'balance_trade','net_capital_export','construction_value','salary_growth','deposits_rate'
]

VL_SUB_A1 = [['Arbat',432,0.7,0]
,['Jakimanka',365,0.2,1] 
,['Hamovniki',341,0.7,0]
#,['Tverskoe',411] - gives incorrect results
,['Meshhanskoe',287,0.2,1]
,['Gagarinskoe',269,0.2,1],['Ramenki',269,0.2,1]
,['Lomonosovskoe',269,0.2,1],['Presnenskoe',266,0.2,1]
,['Dorogomilovo',262,0.2,1]
,['Taganskoe',253,0.2,1]
,['Donskoe',235,0.2,1]
,['Severnoe Butovo',125,0.2,1]
,['Zapadnoe Degunino',128,0.2,1]
#,['Birjulevo Zapadnoe',108,0.05],['Nagatinskij Zaton',162,0.05,1]
]

vl_sc, vl_sn, vl_sk, vl_sb, vl_pb = ['#'], [1], [7], [True], [False]

VCC_LIST = ['a_day','a_m1_p1','state','a_pt','a_pl_p','a_cheap','full_sq']

vd_start_tr  = dt.date (2011, 8, 20)
vd_start_mac = dt.date (2010, 1, 1)
vd_end_tr    = dt.date (2010, 1, 1)
vn_day0      = (vd_start_tr - vd_start_mac).days
vn_days      = 0

#m = KNeighborsRegressor (n_eighbors=4,algorithm='ball_tree',weights='distance')
m = XGBRegressor (max_depth=7,learning_rate=0.05,n_estimators=280,subsample=1,colsample_bytree=1)
m_c = XGBClassifier(max_depth=5,learning_rate=0.075,n_estimators=180)
m_e = XGBRegressor (max_depth=5,learning_rate=0.075,n_estimators=180)

def fix_by_area (p_df, vl_sub_a1, p_print=False):
    if len(vl_sub_a1)>0:
        ln_fixed = 0
        if p_print:
            print ('Fix prices by VL_SUB_A1:')
        for a in vl_sub_a1:
            for index, row in p_df.iterrows():
                if row['sub_area']==a[0] and round (row['price_doc']/row['full_sq']) != round(1000*a[1]) and (a[3]==0 or a[3] == row['a_pt']):
                    p_df.set_value (index, 'price_doc', a[2]*1000*a[1]*row['full_sq'] + (1-a[2])*row['price_doc'])
                    ln_fixed += 1
            if p_print:        
                print_avg_area_price (tst_df, a[0])
        if p_print:        
            print ('flats fixed:', ln_fixed)

def print_feature_importances (X_trn, m, vl_cols=[]):
    df = pd.DataFrame ({'feature':X_trn.columns, 'importance':m.feature_importances_})
    df.sort_values (by='importance',ascending=False,inplace=True)
    print ('Feature importances:')
    ln_zero = 0
    if len(vl_cols)>0:
        for index, row in df.iterrows():
            if row['feature'] in vl_cols:
                print (row['feature'], row['importance'])
            if row['importance']==0:
                ln_zero += 1
        print ('with 0 importance: ', ln_zero)        
    else:
        print (df)
    return df['feature']

def print_avg_area_price (p_df, p_value):
    vb = p_df['sub_area']==p_value
    df = p_df.loc[vb, ['full_sq','price_doc']].copy()
    df['p1m'] = 0.001*df['price_doc']/df['full_sq']
    if VN_FULL_SQ_M != 1:
        df['p2m'] = df['p1m']*VN_FULL_SQ_M
        print ('avg price in sub-area', p_value, round (df['p1m'].mean()), round (df['p2m'].mean()))
    else:
        print ('avg price in sub-area', p_value, round (df['p1m'].mean()))
    del df

def g_list_cross (p_list_fact, p_lists):
    ln = len (p_lists)
    ln1, ln2 = 0, 0
    if ln>0:
        vl = p_lists[0]
        ln1 = sum (vl)
        for i in range (1,ln-1):
            vl = vl * p_lists[i]
            ln1 = sum (vl)
        ln2 = sum (vl*p_list_fact)
    vt_cross = (ln1, ln2)
    return vt_cross

def g_list_mean (p_lists):
    vl = []
    ln = len (p_lists)
    print ('g_list_mean: count = ', ln)
    if ln>0:
        vl = list(p_lists[0])
        for i in range (1,ln):
            print ('g_list_mean: add array with index = ', i)
            for j in range(len(vl)):
                vl[j] += p_lists[i][j]
    for i in range(len(vl)):
        vl[i] /= ln
    return vl

def g_rmsle (p_df, p_c1, p_c2):
    vn_s = 0
    ln_res = 0
    if len(p_df)>0:
        for index, row in p_df.iterrows():
            vn = math.log (row[p_c1]+1) - math.log(row[p_c2]+1)
            vn_s += vn*vn
        ln_res = math.sqrt (vn_s/len(p_df))     
    return ln_res

def ecology_code (p_industrial_km):
    if p_industrial_km >= 1 : vn_code = CN_ECOLOGY_EXCELLENT
    elif p_industrial_km >= 0.8 : vn_code = CN_ECOLOGY_GOOD
    elif p_industrial_km >= 0.6 : vn_code = CN_ECOLOGY_SATISFY
    else: vn_code = CN_ECOLOGY_POOR
    return vn_code

def list_columns_with_nan (p_df, p_drop_nan): 
    for x in p_df.columns.values: 
        if np.any (np.isnan(p_df[x])): 
            print ('has NaN: ', x)
            if p_drop_nan:
                p_df.drop (x,axis=1,inplace=True)
    return        

def list_columns_with_inf (p_df):    
    for x in p_df.columns.values: 
        if np.any (np.isinf(p_df[x])): 
            print ('has inf: ', x)
    return        

def all_float (p_df): 
    for x in p_df.columns.values: 
        p_df [x].astype (float)
    return        

def save_submission (p_df, p_pred):
    submission = pd.DataFrame({'id': p_df['id'], 'price_doc': p_pred})
    submission.to_csv('submission.csv', index=False)    
    return

def print_cc (p_s, p_df, p_cols, p_drop): 
    for x in p_cols: 
        vn = p_s.corr (p_df [x])
        print (x + ':', vn)
        if ((abs(vn)<0.1) & p_drop): p_df.drop([x],axis=1,inplace=True)
    return

def sort_cc (p_s, p_df, p_cols):
    vn_corr = []
    for x in p_cols: 
        #print ('do ', x)
        vn_corr.append (abs(p_s.corr (p_df [x])))
    cc_df = pd.DataFrame({'col': p_cols, 'corr': vn_corr})
    cc_df.sort_values (by='corr', ascending=False, inplace=True)
    print (cc_df)
    return list(cc_df['col'])

def center (p_df, p_cols): 
    for x in p_cols: 
        vn = p_df[x].mean()
        #print ('    center ', x)
        if vn > 0: p_df[x] = p_df[x]/vn
    return

def replace_yn (p_df, p_cols): 
    for x in p_cols:
        p_df[x].replace ('yes', 1, inplace=True)
        p_df[x].replace ('no', 0, inplace=True)
        p_df[x].fillna (0, inplace=True)
    return

def replace_0 (p_df, p_cols): 
    for x in p_cols:
        p_df[x].fillna (0, inplace=True)
    return

def use_model (p_m, p_df, p_cols, p_center):
    X_tst = p_df [p_cols].copy()
    if p_center:
        center (X_tst, X_tst.columns)
    Y_pred = p_m.predict (X_tst)
    return Y_pred

def g_errors (p_df, p_pred, p_col_trn, p_cr, p_print): 
    sub_df = pd.DataFrame({'id': p_df['id'], 'pred': p_pred, 'fact': p_df[p_col_trn]})
    vl_add = (0, 0, 0, 0)
    vn_over_err = 0
    if p_cr:
        sub_df['a_err'] = abs (sub_df ['pred']-sub_df ['fact'])
        ln_cnt_1 = sum (sub_df['fact'])
        ln_cnt_2 = sum (sub_df['pred'])
        ln_cnt_3 = sum (sub_df['fact']*sub_df['pred']) 
        vl_add = (ln_cnt_1, ln_cnt_2, ln_cnt_3, 0)
    else:
        sub_df ['a_err'] = 1.0 - sub_df ['pred']/sub_df ['fact']
        ln_len = len (sub_df)
        if ln_len > 0:
            ln_err_1 = sum (sub_df.loc [sub_df ['a_err'] > 0, 'a_err'])/len (sub_df)
            ln_err_sum = abs(sum (sub_df.loc [sub_df ['a_err'] < 0, 'a_err']))
            ln_err_2 = ln_err_sum/len (sub_df)
            vn_over_err = ln_err_2
            ln_err_3 = len (sub_df.loc [sub_df ['a_err'] == 0])/len (sub_df)
            ln_err_4 = 0
            if ln_err_sum > 0:
                ln_err_4 = ln_err_sum/len((sub_df.loc [sub_df ['a_err'] < 0]))
            vl_add = (ln_err_1, ln_err_2, ln_err_3, ln_err_4)
    sub_df ['a_erra'] = abs (sub_df ['a_err'])
    sub_df.sort_values (by='a_err',ascending=False,inplace=True)
    if p_print:
        sub_df.sort_values (by='a_erra',ascending=False,inplace=True)
        print (sub_df.head(10))
    return g_rmsle (sub_df,'pred','fact'), sub_df ['a_erra'].mean(), vl_add, sub_df, vn_over_err

def test_model (p_m, p_df_self, p_col_trn, p_cols, p_center, p_cr, p_print):
    Y_pred = use_model (p_m, p_df_self, p_cols, p_center)
    vn_rmsle, vn_err, vl_add, sub_df, vn_over_err = g_errors (p_df_self, Y_pred, p_col_trn, p_cr, p_print)
    return vn_rmsle, vn_err, vl_add, sub_df, vn_over_err

def self_train_model (p_m, p_df_trn, p_df_self, p_col_trn, p_cols, p_center, p_cr, p_add_cmd, p_y_trn=[]):
    vn_rmsle, vn_rmsle_best, vn_err, vn_err_best, vn_over_err, vn_over_err_best = 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
    vl_cols_best, vl_pc, vl_cols_bad = [], [], []
    vl_cols = list (p_cols)
    vl_add = (0, 0, 0)
    if len (p_y_trn)>0:
        print ('self_train_model: use given training array, len ', len (p_y_trn))
        Y_trn = list (p_y_trn)
    else:    
        Y_trn = p_df_trn[p_col_trn].copy()
    if VB_ITER_TRAIN:    
        for i in range(1, len(vl_cols)):
            vl_pc = vl_cols[:i]
            X = p_df_trn [vl_pc].copy()
            X.drop (vl_cols_bad, axis=1, inplace=True)
            if p_center: center (X, X.columns)
            if p_m.__class__.__name__ == 'XGBRegressor':
                p_m.fit (X, Y_trn, eval_metric=VC_XGB_EVAL_METRIC)
            else:    
                p_m.fit (X, Y_trn)
            vn_rmsle, vn_err, vl_add, sub_df, vn_over_err = test_model (p_m, p_df_self, p_col_trn, X.columns, p_center, p_cr, p_print=False)
            if (VC_ERR_CTRL == 'mea' and vn_err < vn_err_best) or (VC_ERR_CTRL == 'rmsle' and vn_rmsle < vn_rmsle_best) or (VC_ERR_CTRL == 'over' and vn_over_err < vn_over_err_best):
                vn_rmsle_best = vn_rmsle
                vn_err_best, vn_over_err_best = vn_err, vn_over_err
                vl_cols_best = list(X.columns)
            else:
                vl_cols_bad.append(vl_cols[i-1])
    else:
        vl_cols_best = list(vl_cols)
    X = p_df_trn [vl_cols_best].copy()
    if p_center: 
        center (X, X.columns)
    if p_m.__class__.__name__ == 'XGBRegressor':
        p_m.fit (X, Y_trn, eval_metric=VC_XGB_EVAL_METRIC)
        print_feature_importances (X, p_m, X.columns)
    else:    
        p_m.fit (X, Y_trn)
    if not VB_ITER_TRAIN:
        vn_rmsle_best, vn_err_best, vl_add, sub_df, vn_over_err_best = test_model (p_m, p_df_self, p_col_trn, X.columns, p_center, p_cr, p_print=False)
    return vn_rmsle_best, vn_err_best, vl_cols_best, vl_add, p_m.score (X, Y_trn), sub_df, vn_over_err_best

def find_best_models (p_m, p_m_e, p_m_c, p_df_trn, p_df_self, p_cols, p_center):
    vn_err, vn_err_best, vn_rmsle, vn_rmsle_best, vn_mean_cp = 0.0, 0.0, 0.0, 0.0, 0.0
    vn_errc, vn_err_bestc, vn_rmslec, vn_rmsle_bestc, vn_score, vn_over_err, vn_over_errc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    vl_cols_best, vl_cols_bestc = [], []
    vn_cp_k, vn_cp_k_test, vn_cp_k_step, vn_cp_k_steps = 1.0, 0.2, 0.25, 8
    vl_add = (0, 0, 0)
    vb_use_c = False
    
    vn_rmsle_best, vn_err_best, vl_cols_best, vl_add, vn_score, sub_df, vn_over_err = self_train_model (p_m, p_df_trn, p_df_self, 'price_doc', p_cols, p_center, False, '')
    print ('main model, add values:', vl_add)
    if VC_ERR_CTRL == 'over':
        print ('best overestimate =', vn_over_err)
    if VB_USE_CHEAP_TRAIN:
        vl_cols_c = list(V_CC_CHEAP)
        vn_mean_cp = p_df_trn.loc[p_df_trn['a_cheap']==1,'price_doc'].mean()
        for c in p_cols:
            if c not in vl_cols_c:
                vl_cols_c.append (c)
        vn_rmsle_bestc, vn_err_bestc, vl_cols_bestc, vl_add, vn_score, sub_df_c, vn_over_errc = self_train_model (p_m_c, p_df_trn, p_df_self, 'a_cheap', vl_cols_c, p_center, True, '')
        print ('best cheap train error:', vn_err_bestc, vn_rmsle_bestc)
        print ('cheap model columns: ', vl_cols_bestc)
        print ('add values:', vl_add)
        Y_pred = use_model (p_m, p_df_self, vl_cols_best, p_center)
        Y_predc = use_model (p_m_c, p_df_self, vl_cols_bestc, p_center)
        vn_cp_k_err, vn_cp_k_rmsle = 0.0, 10000.0
        Y_pred0 = list (Y_pred)
        for s in range(vn_cp_k_steps):
            Y_pred = list (Y_pred0)                    
            for i in range(len(Y_pred)):
                if Y_predc[i]==1:
                    Y_pred[i] = vn_cp_k_test*vn_mean_cp
            vn_rmsle, vn_err, vl_add, sub_df_c = g_errors (p_df_self, Y_pred, 'price_doc', False, True)
            if (VC_ERR_CTRL == 'mea' and vn_err < vn_err_best) or (VC_ERR_CTRL == 'rmsle' and vn_rmsle < vn_rmsle_best):
                print ('Nice cp_k = ', vn_cp_k_test)
                print ('May use cheap train:', vn_err, vn_err_best)
                vn_rmsle_best, vn_err_best = vn_rmsle, vn_err
                vn_cp_k = vn_cp_k_test
                vb_use_c = True
            else:
                print ('do not use cheap with k, err', vn_cp_k_test, vn_err)
            vn_cp_k_test += vn_cp_k_step
            
        if not vb_use_c:    
            print ('do not use cheap train:', vn_err, vn_err_best)
    if VB_LEARN_ERROR:
        sub_df['a_merr'] = sub_df['fact']-sub_df['pred']
        p_m_e.fit (p_df_self[vl_cols_best], list(sub_df['a_merr']))
    print ('cheap rate =', len(p_df_trn.loc[p_df_trn['a_cheap']==1])/len(p_df_trn))    
    print ('self len   =', len(p_df_self)) 
    print ('main errors:')
    print (sub_df.head(12))
    print (sub_df.tail(5))
    return vn_rmsle_best, vn_err_best, vl_cols_best, vl_cols_bestc, vb_use_c, vn_cp_k*vn_mean_cp

def set_mac_col (mac_df, c, trn_df2, tst_df, vl_i1, vl_i2, vl_d1, vl_d2, min_day1, min_day2):
    ln_df1_ind, ln_df2_ind, ln_df1_day, ln_df2_day = 0, 0, 0, 0
    ln_df1_day = min_day1
    print ('min day in trn_df2 = ', ln_df1_day, len (trn_df2))
    ln_df2_day = min_day2
    print ('min day in tst_df = ', ln_df2_day, len (tst_df))
                
    for index, row in mac_df.iterrows():
        while True:
            if ln_df1_day - VN_MACRO_DAYS == row['a_day']:
                #print (ln_df1_ind, vl_i1[ln_df1_ind])
                trn_df2.set_value (vl_i1[ln_df1_ind], c, row[c])
                ln_df1_ind += 1
                if ln_df1_ind < len (vl_i1):
                    ln_df1_day = vl_d1[ln_df1_ind]
            else:
                if ln_df1_day - VN_MACRO_DAYS < row['a_day']:
                    ln_df1_day += 1
                else:
                    break
            if ln_df1_ind == len (vl_i1):
                break
        while True:
            if ln_df2_day - VN_MACRO_DAYS == row['a_day']:
                tst_df.set_value (vl_i2[ln_df2_ind], c, row[c])
                ln_df2_ind += 1
                if ln_df2_ind < len (vl_i2):
                    ln_df2_day = vl_d2[ln_df2_ind]
            else: 
                if ln_df2_day - VN_MACRO_DAYS < row['a_day']:
                    ln_df2_day += 1
                else:
                    break
            if ln_df2_ind == len (vl_i2):
                break

def get_cols_in (p_df_cols, p_cols):
    v_in = []
    for x in p_cols:
        if x in p_df_cols:
            v_in.append(x)
    return v_in

def mytrain (m, m_e, trn_df2, tst_df, self_df, p_col, p_val, v_cc, p_center, p_seg, p_find_best_corr):
    print ('mytrain on col/val', p_col, p_val)
    trn_df3 = trn_df2.loc[trn_df2[p_col] == p_val].copy()
    Y_trn = trn_df3['price_doc'].copy()
    vn_err, vn_rmsle, vn_mean_cp = 0, 0, 0
    Y_pred, vl_cols_c = [], []
    if len(Y_trn)>0:
        self_df2 = self_df.loc[self_df[p_col] == p_val].copy()
        tst_df3 = tst_df.loc[tst_df[p_col] == p_val].copy()
        if p_find_best_corr and not VB_USE_CC:
            for c in trn_df3.columns:
                if c.find('*')>0 and c.find('^')>0:
                    for df in [trn_df3, self_df2, tst_df3]:
                        df.drop([c],axis=1,inplace=True)
                    #print ('column removed:', c)
            vl_cc = get_cols_in (trn_df3.columns, v_cc)
            print ('vl_cc len = ', len (vl_cc))
            vl_cols = get_best_corr (trn_df3, self_df2, tst_df3, Y_trn, vl_cc, GN_BC_MIN_POW, GN_BC_MAX_POW, GN_BC_STEP, GN_BC_MIN_CORR, GN_BC_NICE, True)
            print ('best segment corr:')
            print (vl_cols)
        else:
            vl_cols = v_cc
        vn_rmsle, vn_err, v_cols_nice, vl_cols_c, vb_use_c, vn_mean_cp = find_best_models (m, m_e, m_c, trn_df3, self_df2, vl_cols, p_center)
        tst_df2 = tst_df3['id'].copy()
        if VB_ITER_TRAIN:
            X_trn = trn_df3[v_cols_nice].copy()
            if p_center: center (X_trn, X_trn.columns)
            if m.__class__.__name__ == 'XGBRegressor':
                m.fit (X_trn, Y_trn, eval_metric=VC_XGB_EVAL_METRIC)
            else:    
                m.fit (X_trn, Y_trn)
        X_tst = tst_df3[v_cols_nice].copy()
        if len(X_tst)>0:
            if p_center: center (X_tst, X_tst.columns)
            Y_pred = m.predict (X_tst)
            if vb_use_c:
                X_tst_c = tst_df3[vl_cols_c].copy()
                Y_pred_c = m_c.predict (X_tst_c)
                for i in range(len(X_tst_c)):
                    if Y_pred_c[i]==1:
                        Y_pred[i] = vn_mean_cp
            if VB_LEARN_ERROR:
                Y_err = m_e.predict (X_tst)
                i = 0
                for index, row in tst_df3.iterrows():
                    #if row['build_year']!=0 and Y_err[i]<0:
                    Y_pred[i] += VN_LEARN_ERROR_RATE*Y_err[i]
                    i += 1 
    print ('  mytrain len/err/RMSLE = ', len(Y_trn), round (vn_err, 3), round (vn_rmsle, 3))
    print ('    Y_pred len    = ', len(Y_pred))
    sub_df = pd.DataFrame({})
    if len(Y_pred)>0: sub_df = pd.DataFrame({'id': tst_df2, 'price_pred': Y_pred})
    return sub_df, vn_err, len(trn_df3), vn_rmsle    

print ('run')
trn_df0 = pd.read_csv ('../input/train.csv')
print ('train len =', len(trn_df0))
trn_df0 ['full_sq'].astype (float)
vb_min_full_sq = trn_df0 ['full_sq'] > 20
vb_max_full_sq = trn_df0 ['full_sq'] < 300
vb_pt          = True

trn_df = trn_df0.loc [vb_min_full_sq & vb_max_full_sq & vb_pt].copy()
del trn_df0
print ('train len with adequate square = ', len (trn_df))
tst_df = pd.read_csv ('../input/test.csv')

for df in [trn_df, tst_df]:
    df['#']=1
    for x in ['a_cheap','a_expensive','a_pt','a_kr']:
        df[x]=0

trn_df ['a_m1_p1'] = 0.001 * trn_df ['price_doc'] / trn_df ['full_sq']
trn_df ['a_m1_p1rl'] = round (trn_df ['a_m1_p1']/CN_PL_DIV)

# price level by sub_area
print ('Price level by area:')
v_pl_by_area = trn_df[['a_m1_p1rl', 'sub_area']].groupby (['sub_area'], as_index = False).mean().sort_values (by = 'a_m1_p1rl', ascending = False)
print (v_pl_by_area)

print ('prepare data for analysis and training')    
for df in [trn_df, tst_df]:
    df['a_sq'] = [math.trunc(x/20) for x in df.full_sq]
    df['product_type'].fillna  ('Investment', inplace=True)
    df.loc[df['product_type'] == 'OwnerOccupier', 'a_pt'] = 1
    df.loc[df['product_type'] == 'Investment', 'a_pt'] = 2
    df ['state'].fillna (GN_FILL_NA_STATE, inplace = True)
    df ['state'].replace (33, 3, inplace = True)
    df ['state'].replace (4, GN_FILL_4_STATE, inplace = True)
    df ['material'].fillna (7, inplace = True)
    df ['material'].replace (3, 4, inplace = True)
    replace_yn (df, ['thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion', 'radiation_raion', 'detention_facility_raion'
    , 'nuclear_reactor_raion', 'culture_objects_top_25','railroad_terminal_raion','big_market_raion','water_1line','big_road1_1line','railroad_1line'])
    df ['a_bad_factors'] = df ['incineration_raion'] + df ['thermal_power_plant_raion'] + df ['oil_chemistry_raion'] + df ['radiation_raion'] + df['detention_facility_raion'] + df['nuclear_reactor_raion']
    replace_0 (df, ['incineration_km','thermal_power_plant_km','oil_chemistry_km','radiation_km','detention_facility_km','nuclear_reactor_km'
    ,'cafe_count_5000_price_1500','cafe_count_5000_price_2500','cafe_count_5000_price_4000','cafe_count_5000_price_high','leisure_count_5000','sport_count_5000','market_count_5000'
    ,'cafe_count_2000_price_1500','cafe_count_2000_price_2500','cafe_count_2000_price_4000','cafe_count_2000_price_high','leisure_count_2000','sport_count_2000','market_count_2000'
    ,'cafe_count_1000_price_2500','cafe_count_1000_price_4000','cafe_count_1000_price_high'
    ,'cafe_count_500_price_2500','cafe_count_500_price_4000','cafe_count_500_price_high'
    ,'green_part_500','prom_part_500','office_sqm_500','trc_sqm_500','preschool_km'
    ,'kindergarten_km','school_km'
    ,'green_part_1000','prom_part_1000','office_sqm_1000','trc_sqm_1000','school_education_centers_raion','preschool_education_centers_raion'
    ,'floor','max_floor','build_year','kremlin_km'
    ])
    df ['a_5000'] = df['cafe_count_5000_price_1500']+df['cafe_count_5000_price_2500']+df['cafe_count_5000_price_4000']+df['cafe_count_5000_price_high']+df['leisure_count_5000']+df['sport_count_5000']+df['market_count_5000']
    df ['a_2000e'] = df['leisure_count_2000']+df['sport_count_2000']
    df ['a_2000'] = df['cafe_count_2000_price_1500']+df['cafe_count_2000_price_2500']+df['cafe_count_2000_price_4000']+df['cafe_count_2000_price_high']+df['a_2000e']+df['market_count_2000']
    df ['a_bad_factors_km'] = df ['incineration_km'] + df ['thermal_power_plant_km'] + df ['oil_chemistry_km'] + df ['radiation_km'] + df['detention_facility_km'] + df['nuclear_reactor_km'] + df['power_transmission_line_km']
    for c in ['0_17','16_29','ekder']:
        df ['a_'+c] = df[c+'_all']/df['full_all']
    df ['a_big_roads'] = df ['big_road1_km'] + df ['big_road2_km']
    df ['a_education'] = df ['preschool_education_centers_raion'] + df ['school_education_centers_raion'] 
    df ['a_p500'] = df['green_part_500']+df['office_sqm_500']+df['trc_sqm_500']
    df ['a_child_km'] = df ['kindergarten_km'] + df ['school_km'] + df['preschool_km']
    df ['a_pl_by_area_p'] = df ['sub_area'] + df ['product_type']
    df ['ecology'] = df ['ecology'].replace ('excellent',CN_ECOLOGY_EXCELLENT)
    df ['ecology'] = df ['ecology'].replace ('good',CN_ECOLOGY_GOOD)
    df ['ecology'] = df ['ecology'].replace ('satisfactory',CN_ECOLOGY_SATISFY)
    df ['ecology'] = df ['ecology'].replace ('poor',CN_ECOLOGY_POOR)
    df ['a_pl_by_area'] = [v_pl_by_area.loc [v_pl_by_area ['sub_area'] == x, 'a_m1_p1rl'] for x in df.sub_area]
    df ['a_pl0'] = [x.tolist() for x in df.a_pl_by_area]
    df ['a_pl0'] = [np.append(x, [VN_PL_CONST]) for x in df.a_pl0]
    df ['a_pl'] = [round(x[0],1) for x in df.a_pl0]
    dfm = df.loc [df ['metro_km_walk'].isnull(), ['metro_km_walk', 'metro_km_avto']].copy()
    df.metro_km_walk.replace(dfm.set_index('metro_km_walk')['metro_km_avto'], inplace=True)
    df.drop (['a_pl_by_area', 'a_pl0'], axis = 1, inplace=True)

vb_not_cheap     = trn_df ['price_doc'] >= VN_MIN_PRICE
vb_not_expensive = trn_df ['price_doc'] <= VN_MAX_PRICE
trn_df.loc[trn_df['a_m1_p1rl'] < VN_CHEAP_INDEX*trn_df['a_pl'], 'a_cheap'] = 1
trn_df.loc[trn_df['a_m1_p1rl'] > 2*trn_df['a_pl'], 'a_expensive'] = 1

v_cheap_by_area = trn_df[['sub_area','#','a_cheap']].groupby (['sub_area'],as_index=False).sum().sort_values (by=['a_cheap'],ascending=False)
v_cheap_by_area['rate'] = v_cheap_by_area['a_cheap']/v_cheap_by_area['#']

vb_acheap = trn_df['a_cheap']==1
vn_acheap = len(trn_df.loc[vb_acheap])
print ('All cheap:', vn_acheap)
vb_1m = trn_df['price_doc']<=1000000
vb_2m = trn_df['price_doc']<=2000000
vb_3m = trn_df['price_doc']<=3000000

print ('All <=1 mln:', len(trn_df.loc[vb_1m]), len(trn_df.loc[vb_acheap & vb_1m]))
print ('All <=2 mln:', len(trn_df.loc[vb_2m]), len(trn_df.loc[vb_acheap & vb_2m]))
print ('All <=3 mln:', len(trn_df.loc[vb_3m]), len(trn_df.loc[vb_acheap & vb_3m]))

if 'area_id' not in VL_EXCLUDE_CC:
    print ('put sub_area id')
    vl_area = []
    for index, row in trn_df.iterrows():
        lc_area = row['sub_area']
        if lc_area not in vl_area:
            vl_area.append (lc_area)
        trn_df.set_value (index, 'area_id', vl_area.index (lc_area)+1)
    for index, row in tst_df.iterrows():
        lc_area = row['sub_area']
        if lc_area not in vl_area:
            vl_area.append (lc_area)
        tst_df.set_value (index, 'area_id', vl_area.index (lc_area)+1)

if 'a_cr' in VL_MAIN_CC:
    for df in [trn_df, tst_df]:
        df ['a_c_by_area'] = [v_cheap_by_area.loc [v_cheap_by_area ['sub_area'] == x, 'rate'] for x in df.sub_area]
        df ['a_cr0'] = [x.tolist() for x in df.a_c_by_area]
        df ['a_cr0'] = [np.append(x, [0.0]) for x in df.a_cr0]
        df ['a_cr'] = [x[0] for x in df.a_cr0]
        df.drop (['a_c_by_area', 'a_cr0'], axis = 1, inplace=True)

vb1 = trn_df['a_cheap']==0
vb3 = trn_df['a_expensive']==0
vb_pt = vb1 
trn_df2 = trn_df.loc [vb_not_cheap & vb_not_expensive & vb_pt & vb3].copy()
print ('trn_df2 len=',len(trn_df2))
del trn_df

print ('part 1 completed')
v_pl_by_area_p = trn_df2[['a_m1_p1rl', 'a_pl_by_area_p']].groupby (['a_pl_by_area_p'], as_index = False).mean().sort_values (by = 'a_pl_by_area_p', ascending = True)

if VN_FULL_SQ_M != 1:
    print ('Full square multiplication:')
    print (tst_df['full_sq'].head())
    tst_df['full_sq'] = VN_FULL_SQ_M * tst_df['full_sq']
    print (tst_df['full_sq'].head())

for df in [trn_df2, tst_df]:
    df ['a_pl_by_area_p2'] = [v_pl_by_area_p.loc [v_pl_by_area_p ['a_pl_by_area_p'] == x, 'a_m1_p1rl'] for x in df.a_pl_by_area_p]
    df ['a_pl0p'] = [x.tolist() for x in df.a_pl_by_area_p2]
    df ['a_pl0p'] = [np.append(x, df ['a_pl']) for x in df.a_pl0p]
    df ['a_pl_p'] = [x[0] for x in df.a_pl0p]
    eco_no_data_df2 = df.loc [df ['ecology'] == 'no data', ['ecology','industrial_km']]      
    eco_no_data_df2 ['a_ecology'] = [ecology_code (x) for x in eco_no_data_df2.industrial_km]
    df.ecology.replace(eco_no_data_df2.set_index('ecology')['a_ecology'], inplace=True)
    df ['ecology'] = df ['ecology'].astype (int)
    df ['a_dt'] = [dt.datetime.strptime(x, "%Y-%m-%d").date() for x in df.timestamp]
    df ['a_day'] = [(x - vd_start_mac).days for x in df.a_dt]
    df ['a_q'] = [math.trunc (x/91) for x in df.a_day]
    df ['a_mm'] = [math.trunc (x/31) for x in df.a_day]
    df ['a_m'] = [math.trunc (x/30)%12 for x in df.a_day]
    df.drop (['a_pl_by_area_p2', 'a_pl0p','a_pl_by_area_p'], axis = 1, inplace=True)

if VB_USE_MAX:
    for c in trn_df2.columns: 
        if c not in VL_EXCLUDE_CC and c not in VL_MAIN_CC:
            VL_MAIN_CC.append (c)
            print (c)
            trn_df2[c].astype (float)
            tst_df[c].astype (float)

if VN_PLP_M != 1:
    print ('Pl_p multiplication:')
    print (tst_df['a_pl_p'].head())
    tst_df['a_pl_p'] = VN_PLP_M * tst_df['a_pl_p']
    print (tst_df['a_pl_p'].head())

if VB_MACRO_ALL or len(V_MACRO)>0:
    print ('fill by macro...')
    print ('sort trn_df2')
    trn_df2.sort_values (by='id',ascending=True,inplace=True)
    print ('sort tst_df')
    tst_df.sort_values (by='id',ascending=True,inplace=True)
    ln_min_day1 = trn_df2['a_day'].min()
    ln_min_day2 = tst_df['a_day'].min()
    
    print ('read macro.csv')
    mac_df = pd.read_csv ('../input/macro.csv')
    mac_df ['a_dt'] = [dt.datetime.strptime(x, "%Y-%m-%d").date() for x in mac_df.timestamp]
    mac_df ['a_day'] = [(x - vd_start_mac).days for x in mac_df.a_dt]
    vl_i1, vl_i2, vl_d1, vl_d2 = [], [], [], []
    ln_col = 0
    for index, row in trn_df2.iterrows():
        vl_i1.append (index)
        vl_d1.append (row['a_day'])
    for index, row in tst_df.iterrows():
        vl_i2.append (index)
        vl_d2.append (row['a_day'])
    if VB_MACRO_ALL:
        print ('mac columns len = ', len (mac_df.columns))
        for c in mac_df.columns:
            if c not in VL_EXCLUDE_CC:
                print ('ffill ', c)
                mac_df [c].fillna (method='ffill', inplace=True)
        for c in mac_df.columns:
            if c not in VL_EXCLUDE_CC:
                ln_col += 1
                if c not in VL_MAIN_CC:
                    VL_MAIN_CC.append (c)
                print ('set ', c, ln_col)
                set_mac_col (mac_df, c, trn_df2, tst_df, vl_i1, vl_i2, vl_d1, vl_d2, ln_min_day1, ln_min_day2)
    else:                
        for c in V_MACRO:
            if c in VL_MAIN_CC:
                print ('ffill ', c)
                mac_df [c].fillna (method='ffill', inplace=True)
        for c in V_MACRO:
            if c in VL_MAIN_CC:
                ln_col += 1
                print ('set ', c, ln_col)
                set_mac_col (mac_df, c, trn_df2, tst_df, vl_i1, vl_i2, vl_d1, vl_d2, ln_min_day1, ln_min_day2)
    del mac_df

vd_end_tr = trn_df2['a_dt'].max()
print ('max date=', vd_end_tr)
vn_days = (vd_end_tr-vd_start_tr).days

v_p1_by_material = trn_df2[['material','a_m1_p1']].groupby (['material'],as_index=False).mean().sort_values (by=['a_m1_p1'],ascending=False)
print ('p1 by material:')
print (v_p1_by_material)

if 'a_material' in VL_MAIN_CC:
    for df in [trn_df2, tst_df]:
        df ['a_material0'] = [v_p1_by_material.loc [v_p1_by_material ['material'] == x, 'a_m1_p1'] for x in df.material]
        df ['a_m0'] = [x.tolist() for x in df.a_material0]
        df ['a_m0'] = [np.append(x, [138.0]) for x in df.a_m0]
        df ['a_material'] = [x[0] for x in df.a_m0]
        df.drop (['a_material0', 'a_m0'], axis = 1, inplace=True)

v_p1_by_mm = trn_df2[['a_mm','a_m1_p1']].groupby (['a_mm'],as_index=False).mean().sort_values (by=['a_m1_p1'],ascending=False)

if 'a_mmp' in VL_MAIN_CC:
    for df in [trn_df2, tst_df]:
        df ['a_mmp0'] = [v_p1_by_mm.loc [v_p1_by_mm ['a_mm'] == x, 'a_m1_p1'] for x in df.a_mm]
        df ['a_mmp00'] = [x.tolist() for x in df.a_mmp0]
        df ['a_mmp00'] = [np.append(x, [0]) for x in df.a_mmp00]
        df ['a_mmp'] = [x[0] for x in df.a_mmp00]
        df.drop (['a_mmp00', 'a_mmp0'], axis = 1, inplace=True)

v_cc = list(VL_MAIN_CC)

#v_cc = sort_cc (trn_df2[VC_AUTOCORR_COLUMN],trn_df2,v_cc)

vl_cols_global = []
vl_cols = v_cc

vn_m1, vn_m2, vn_m1t, vn_m2t = 0, 0, 0, 0
sum_df = []

print ('fix full_sq in tst_df:')
ln_sq_fixed = 0
for index, row in tst_df.iterrows():
    if row['full_sq']<15:
        print (row['id'], row['full_sq'])
        if row['life_sq']>10:
            tst_df.set_value(index, 'full_sq', row['life_sq'])
        else:
            if row['num_room']>0:
                tst_df.set_value (index, 'full_sq', 20+20*row['num_room'])
            else:
                tst_df.set_value (index, 'full_sq', 45)
        ln_sq_fixed += 1
print ('fixed: ', ln_sq_fixed)

print ('Get self-test selection with quant = 1 /', vn_quant)
v_ucols = []
if lb_get_best_corr_seg and len(vl_cols_global)==0:
    v_cc2 = list(v_cc)
    v_ucols = list(v_cc)
else:
    if len(vl_cols_global)>0:
        v_cc2 = vl_cols_global.copy()
        v_ucols = vl_cols_global.copy()
    else:    
        v_cc2 = list(v_cc)
        v_ucols = list(v_cc)

# cluster
if VB_CLUSTER:
    trn_k = trn_df2[v_ucols].copy()
    center (trn_k, trn_k.columns)
    tst_k = tst_df[v_ucols].copy()
    center (tst_k, tst_k.columns)
    trn_k['k#'] = 1
    tst_k['k#'] = 2
    u_df = pd.concat ([trn_k, tst_k])
    list_columns_with_inf (u_df)
    list_columns_with_nan (u_df,True)
    k_s = u_df['k#'].copy()
    u_df.drop (['k#'],axis=1,inplace=True)
    print ('clustering')
    v_kc, v_kl, v_inertia = k_means (u_df, ln_clusters)
    print ('clustered:', v_inertia, len(v_kc), len(v_kl))
    print (v_kc)
    vs_l = pd.Series (v_kl, index=u_df.index)
    u_df['a_k'] = vs_l
    u_df['k#'] = k_s
    trn_df2['a_k'] = u_df.loc[u_df['k#']==1,'a_k']
    tst_df['a_k'] = u_df.loc[u_df['k#']==2,'a_k']
    del trn_k
    del tst_k
    del k_s

for x in pd.Series (vl_sc).drop_duplicates():
    if not x in v_cc2: v_cc2.append (x)

for x in pd.Series (VCC_LIST).drop_duplicates():
    if not x in v_cc2: v_cc2.append (x)

for x in pd.Series (V_CC_CHEAP).drop_duplicates():
    if not x in v_cc2: v_cc2.append (x)

v_cc2.append('price_doc')
v_cc2.append('id')
self_df = trn_df2.tail (round(len(trn_df2)/vn_quant))[v_cc2].copy()
print ('self        len = ', len (self_df))
print ('trn_df2 new len = ', len (trn_df2))

vn_segments = len(vl_sc)
for i in range(vn_segments):
    print ('######## DO SEGMENT',(i+1),'########################')
    if vl_sb[i]:
        vl_ucols = list (vl_cols_global)
    else:
        vl_ucols = list (v_cc)
    print ('vl_ucols len = ', len(vl_ucols))
    vl_cols_in = get_cols_in (trn_df2.columns, v_ucols)
    print ('vl_cols_in len = ', len(vl_cols_in))
    lc_cur_col, ln_cur_val = vl_sc[i], vl_sn[i]
    vb_segm_cols = (not vl_sb[i]) and lb_get_best_corr_seg
    if vl_sk[i] and m.__class__.__name__ == 'KNeighborsRegressor':
        m.set_params(n_neighbors=vl_sk[i])
    if vl_pb[i]:
        vl_cols_in = g_unique_pairs (pd.DataFrame ({'col' : vl_cols_in}))
        print ('only unique pairs:', len(vl_cols_in))
    #if len (vl_cols_in)>GN_BC_MAX_ITEMS:
    #    vl_cols_in = trunc_corr_list (vl_cols_in, GN_BC_MAX_ITEMS) 
    sub_df_, ln_err, ln_seg, ln_rmsle = mytrain (m, m_e, trn_df2, tst_df, self_df, lc_cur_col, ln_cur_val, vl_cols_in, VB_CENTER, i+1, vb_segm_cols)
    if i < vn_segments-1:
        for df in [trn_df2, tst_df, self_df]:
            for j in range(vn_segments-1-i):
                if vl_sc[j+i+1] != lc_cur_col:
                    df.loc[df[lc_cur_col] == ln_cur_val, vl_sc[j+i+1]] = -1
    if ln_seg>0: 
        if ln_err < 0 or ln_err > 0:
            vn_m1 += ln_err * ln_seg
        vn_m2 += ln_seg
    ln_sub_df = len(sub_df_)
    if ln_sub_df > 0:
        if ln_err < 0 or ln_err > 0:
            vn_m1t += ln_err * ln_sub_df
        vn_m2t += ln_sub_df
        sum_df.append (sub_df_)
    print ('END SEGMENT, RMSLE=',round(ln_rmsle,5))
vn_mid_err = vn_m1/vn_m2
vn_mid_errt = vn_m1t/vn_m2t
print ('mid err/test =', round (vn_mid_err, 3), round (vn_mid_errt, 3))
#if round (vn_mid_err,2) <= VN_MAX_ERROR:
sub_df = pd.concat (sum_df)
sub_df.sort_values (by = 'id', ascending = True, inplace = True)
print (sub_df.head(5))
print (sub_df.tail(5))
print ('selected objects:')
print ('1-ka Marjino, 2015:', sub_df.loc[sub_df['id']==32741])
print ('2-ka Marjino, 2016:', sub_df.loc[sub_df['id']==38125])
print ('3-ka Marjino, 2016:', sub_df.loc[sub_df['id']==37467])
    
print ('avg price=', sub_df['price_pred'].mean())
tst_df['price_doc'] = sub_df['price_pred']
tst_df['m1'] = tst_df['price_doc']/tst_df['full_sq']
print ('avg price per m2 = ', tst_df['m1'].mean())

for r in ['Birjulevo Zapadnoe','Severnoe Butovo','Arbat','Hamovniki','Tverskoe','Dorogomilovo','Nagatinskij Zaton',
'Jakimanka','Meshhanskoe','Gagarinskoe','Presnenskoe','Donskoe','Ramenki','Taganskoe']:
    print_avg_area_price (tst_df, r)

fix_by_area (tst_df, VL_SUB_A1, True)

sub_df['price_pred'] = tst_df['price_doc']            
print ('avg price (2)        =', sub_df['price_pred'].mean())
tst_df['m1_'] = tst_df['price_doc']/tst_df['full_sq']
print ('avg price per m2 (2) = ', tst_df['m1_'].mean())

price_pred = list(sub_df['price_pred'])
price_pred = list (map (lambda x: round(x/10000)*10000, price_pred))
print ('data len =', len(price_pred))
save_submission (sub_df, price_pred) 
    
