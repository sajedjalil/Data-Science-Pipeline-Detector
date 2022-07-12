# -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
# Helper Script for M5 pipeline stage 2 .|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\-
# -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import lightgbm as lgbm
from sklearn.decomposition import PCA
from colorama import Fore

# -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
# Loading data --/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-

def load_stage2_data(verbose=True):
    """
    Loads competition files, as well as files with pre-generated model features.
    """
    
    files = {}
    
    not_load = ["sales_train_validation.csv", "sales_train_evaluation.csv", "sell_prices.csv"]
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            if filename.endswith('.csv') and (filename not in not_load):
                files[filename[:-4]] = pd.read_csv(os.path.join(dirname, filename))
                
    return files

# -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
# Time-cycle features --\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-

def grp_trig_time_feat(grp, grp_var, n=None):
    """
    Generates sin/cos time-cycle features within a given period.
    """
    grp.loc[:,"idx_adj"] = grp["idx"] - grp.idx.min()
    if grp_var == "year" and grp.idx.min() == 0:
        grp.loc[:,"idx_adj"] = grp["idx_adj"] + 28
    else:
        pass
    if n is None:
        n = grp.shape[0]
    else:
        pass
    grp.loc[:,grp_var+"_sin"] = np.sin(2*np.pi*grp.idx_adj/n)
    grp.loc[:,grp_var+"_cos"] = np.cos(2*np.pi*grp.idx_adj/n)
    return grp.drop("idx_adj", 1)

def gen_trig_time_feat(cal_df, grp_var, n=None):
    """
    Generates sin/cos time-cycle features across multiple periods. 
    """
    out_df = cal_df.copy()
    grp_df = cal_df.groupby(grp_var)[["idx"]].apply(lambda x: grp_trig_time_feat(x, grp_var, n))
    return out_df.merge(grp_df, on=["idx"], how="left")

# -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
# Event-based features -\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-

def gen_spec_day_feat(cal_df, spec_day_dict={}, na_fill=0):
    """
    Generates feature vectors from event_name columns in calendar data.
    """
    out_df = cal_df.copy()
    
    for spec_day, n in spec_day_dict.items():
        event_days = out_df[(out_df.event_name_1 == spec_day)|
                            (out_df.event_name_2 == spec_day)][["idx"]]
        if spec_day == "IndependenceDay":
            event_days = event_days.append(pd.DataFrame({"idx":[1983]}))
        else:
            pass
        # pre-event period
        if n[0] > 0:
            pre_event = [pd.DataFrame({"pre_"+spec_day+"_idx": list(range(row.idx-n[0], row.idx+1)),
                                       "idx": row.idx}) for i, row in event_days.iterrows()]
            pre_event = pd.concat(pre_event, ignore_index=True)
            pre_event.loc[:, "pre_"+spec_day] = 1 / (1 + pre_event["idx"] - 
                                                     pre_event["pre_"+spec_day+"_idx"])
            out_df = out_df.merge(pre_event.drop("idx",1), right_on=["pre_"+spec_day+"_idx"], 
                                  left_on=['idx'], how='left').drop("pre_"+spec_day+"_idx", 1)
            out_df.loc[:, "pre_"+spec_day] = out_df["pre_"+spec_day].fillna(na_fill)
        else:
            pass
        # post-event period
        if n[1] > 0:
            post_event = [pd.DataFrame({"post_"+spec_day+"_idx": list(range(row.idx, row.idx+n[1]+1)),
                                        "idx": row.idx}) for i, row in event_days.iterrows()]
            post_event = pd.concat(post_event, ignore_index=True)
            post_event.loc[:, "post_"+spec_day] = 1 / (1 + post_event["post_"+spec_day+"_idx"] - 
                                                       post_event["idx"])
            out_df = out_df.merge(post_event.drop("idx",1), right_on=["post_"+spec_day+"_idx"], 
                                  left_on=['idx'], how='left').drop("post_"+spec_day+"_idx", 1)
            out_df.loc[:, "post_"+spec_day] = out_df["post_"+spec_day].fillna(na_fill)
        else:
            pass
            
    out_df = out_df.drop(["event_name_1", "event_name_2",
                          "event_type_1", "event_type_2"], 1)    
    
    return out_df

# -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
# All calendar features \|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-

def gen_all_cal_feat(cal_df, spec_day_dict={
        "Easter": (24, 14),
        "Mother's day": (15, 0),
        "Father's day": (15, 0),
        "Thanksgiving": (15, 10)
    }, na_fill=0):
    """
    Generates calendar features.
    """
    out_df = cal_df.copy()
    out_df = out_df.reset_index().rename(columns={"index":"idx"})
    
    # sine/cosine time cycle features
    out_df = gen_trig_time_feat(out_df, "year", n=365)
    out_df.loc[:,"wday_sin"] =  np.sin(2*np.pi*out_df["wday"]/7)
    out_df.loc[:,"wday_cos"] =  np.cos(2*np.pi*out_df["wday"]/7)
    
    # event-based features
    out_df = gen_spec_day_feat(out_df, spec_day_dict, na_fill)
    
    # add y-m index
    ym_df = out_df.copy()[['year','month']].drop_duplicates().reset_index(drop=True
                 ).reset_index().rename(columns={'index':'ymidx'})
    out_df = out_df.merge(ym_df, on=['year','month'], how='left')
    out_df = out_df.drop(["date", "year", "weekday"], 1)
    
    return out_df

# -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
# Group feature extraction |/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
    
def gen_grp_dict():
    """
    Generates dictionary that maps group levels to item ID token indices.
    """
    
    grp_dict = {
        'item': [0, 1, 2],
        'dept': [0, 1],
        'cat': [0],
        'store': [3, 4],
        'state': [3]
    }
    
    pairs = [['dept', 'state'], ['cat', 'state'], ['item', 'state'], 
             ['dept', 'store'], ['cat','store']]
    
    for pair in pairs:
        k = "_".join(pair)
        v = []
        for p in pair:
            v += grp_dict[p]
        grp_dict[k] = v
    
    return grp_dict
                    
def get_item_grps(item_id, prc_grps):
    """
    Gets an item's relevant group IDs based on its own ID.
    """
    
    grp_dict = gen_grp_dict()
    item_tokens = item_id.split("_")
    item_grps = []
    
    for k, v in grp_dict.items():
        grp_id = []
        for i in v:
            grp_id.append(item_tokens[i])
        item_grps.append("_".join(grp_id))

    item_grps.append(prc_grps.query("id == @item_id").values[0][1])
    
    return item_grps

def get_grp_prc_features(item_grps, grp_prices):
    """
    Gets the price features for a set of item groups.
    """
    grps_df = grp_prices.copy().query("grp_id in @item_grps")
    
    grps_list = []

    for grp_id in grps_df["grp_id"].unique():  
        temp_df = grps_df.query("grp_id == @grp_id").drop('grp_id', 1).set_index("wm_yr_wk")
        temp_df.columns = ["_".join([c, grp_id]) for c in temp_df.columns]
        grps_list.append(temp_df)

    return pd.concat(grps_list, axis=1)

# -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
# Compile item features \|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-

def gen_item_df(files, item_id, cal_df, wk_lags=[1], 
                max_corr=0.99, pca_min_var=None, verbose=False):
    """
    Generates item-level training data.
    """
    # get item-level sale data
    item_df = files["item_sales"].query("id == @item_id")
    item_df = item_df.set_index("id").transpose().rename(columns={item_id:"TARGET"})

    # merge calendar features
    item_state = item_id.split("_")[3]
    item_df = cal_df.join(item_df, on="d", how="left").drop("d", 1)
    for col in [c for c in item_df.columns if c.startswith("snap_") 
                and not c.endswith(item_state)]:
        item_df = item_df.drop(col, 1) # remove non-relevant snaps

    # merge item prices | drop na-prices as item not being sold
    item_prc = files["item_prices"].query("id == @item_id").drop("id",1)
    item_df = item_df.merge(item_prc, on=["wm_yr_wk"], 
                            how="left")

    # merge item's group-level prices
    item_grps = get_item_grps(item_id, files["price_grps"])
    item_grp_price = get_grp_prc_features(item_grps, files["grp_prices"])
    prc_col_keep = [c for c in item_grp_price.columns if 
                    (not c.startswith("prc_idx_")) or
                    (c in ["prc_idx_"+item_grps[i] for i in [0, 7, -1]])]
    item_df = item_df.merge(item_grp_price[prc_col_keep], on=["wm_yr_wk"], 
                            how="left")

    # make group price indexes relative
    for col in [c for c in item_df.columns if c.startswith("prc_idx_")]:
        item_df.loc[:, col] = item_df[col] / item_df["prc_idx"]

    # create wk_idx
    wk_idx = item_df[["wm_yr_wk"]].drop_duplicates().reset_index(drop=True
                                 ).reset_index().rename(columns={"index":"wk_idx"})
    item_df = item_df.merge(wk_idx, on="wm_yr_wk").drop("wm_yr_wk", 1)
    item_df.loc[item_df["wday"].isin([1,2]),"wk_idx"
                ] = item_df.loc[item_df["wday"].isin([1,2]),"wk_idx"] - 1
    item_df = item_df.copy().query("wk_idx >= 0")

    # create lags
    for lag in wk_lags:
        item_df.loc[:,"sales_lag_"+str(lag)] = item_df["TARGET"].shift(lag*7)
        item_df.loc[:,"mean_sales_lag_"+str(lag)
                    ] = item_df.groupby("wk_idx")["sales_lag_"+str(lag)].transform("mean")
        item_df.loc[:,"std_sales_lag_"+str(lag)
                    ] = item_df.groupby("wk_idx")["sales_lag_"+str(lag)].transform("std")
        item_df.loc[:,"n0_sales_lag_"+str(lag)
                    ] = item_df.groupby("wk_idx")["sales_lag_"+str(lag)].transform(lambda x: sum(x==0))
        item_df.loc[:,"max_sales_lag_"+str(lag)
                    ] = item_df.groupby("wk_idx")["sales_lag_"+str(lag)].transform("max")
        
    # remove na rows
    item_df = item_df.dropna(subset=["prc_idx"])
    
    # remove constant columns
    col_std = item_df.std()
    to_drop = col_std[col_std == 0].index.tolist()
    item_df = item_df.drop(to_drop, 1)
    if verbose:
        print(Fore.CYAN+"Dropping %d columns with 0 variance: %s"%(len(to_drop),
                                                                   ", ".join(to_drop)))
        
    # remove highly correlated columns
    corr_df = item_df.copy().drop("TARGET", 1).corr()
    to_drop = []
    for col in corr_df.columns:
        temp = corr_df.loc[col, col:].drop(col)
        hi_corr = temp[temp > max_corr]
        if len(hi_corr) > 0:
            to_drop += hi_corr.index.tolist()
    to_drop = [c for c in list(set(to_drop)) 
               if c not in ['idx', 'ymidx', 'wk_idx', 'TARGET']]
    if verbose:
        print(Fore.CYAN+"Dropping %d almost-identical columns: %s"%(len(to_drop),
                                                                    ", ".join(to_drop)))
        
    # apply PCA to price change columns to condense price features  
    item_df = item_df.drop(to_drop, 1)
    df = item_df.copy()
    if pca_min_var is not None:
        pca = PCA(n_components=pca_min_var)
        for i in [0,1]:
            oth_price = [c for c in df if c.startswith("pct_chg_"+str(i)+"_") 
                         and not c.endswith(item_grps[-1])]
            pca_vec = pca.fit_transform(df[oth_price])
            for n in range(pca_vec.shape[1]):
                df.loc[:,"pct_chg_"+str(i)+"_pca"+str(n)] = pca_vec[:,n]
            df = df.drop(oth_price,1)
        if verbose:
            print(Fore.CYAN+"Reducing %d columns to %d using PCA"%(item_df.shape[1],
                                                                   df.shape[1])) 
    return df

# -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
# Model training /|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-

def staggered_pred(model, X_test, round_preds):
    """
    Staggered predictions forward-filling AR terms.
    """
    preds = []
    for i in range(4):
        tmp = X_test.copy().iloc[i*7:(i+1)*7,:]
        if i > 0:
            if "sales_lag_1" in tmp.columns:
                tmp.loc[:,"sales_lag_1"] = tmp_pred
            if "mean_sales_lag_1" in tmp.columns:    
                tmp.loc[:,"mean_sales_lag_1"] = np.mean(tmp_pred)
            if "std_sales_lag_1" in tmp.columns:
                tmp.loc[:,"std_sales_lag_1"] = np.std(tmp_pred)
            if "n0_sales_lag_1" in tmp.columns:
                tmp.loc[:,"n0_sales_lag_1"] = sum(tmp_pred < 0.5)
            if "max_sales_lag_1" in tmp.columns:
                tmp.loc[:,"max_sales_lag_1"] = max(tmp_pred)            
        else:
            pass
        tmp_pred = model.predict(tmp)
        if round_preds:
            tmp_pred = np.round(np.where(tmp_pred<0, 0, tmp_pred))
        else:
            pass
        preds += list(tmp_pred)
    return preds

def train_lgbm(item_df, params, X_cat=["wday","month"], X_drop=[], 
               valid_split=1912, test_split=1940, lastNdays=90,
               num_boost_round=150, early_stopping_rounds=30,
               show_plots=False, verbose=False, keep_score=None,
               return_model=False, round_preds=False):
    """
    Trains LightGBM and produces both validation and evaluation predictions.
    """
    # separate target and features
    y_col = "TARGET"
    X_col = [c for c in item_df.columns if (c != y_col) and (c not in X_drop)]

    # split the data
    train_df = item_df.copy().query("idx <= @valid_split")
    valid_df = item_df.copy().query("idx > @valid_split and idx <= @test_split")
    final_df = item_df.copy().query("idx <= @test_split")
    test_df = item_df.copy().query("idx > @test_split")

    # add last90days columns
    if lastNdays is not None:
        X_col.append('last_'+str(lastNdays))
        train_df.loc[:,'last_'+str(lastNdays)] = np.where(train_df['idx']>(test_split-lastNdays), 1, 0)
        valid_df.loc[:,'last_'+str(lastNdays)] = 1
        final_df.loc[:,'last_'+str(lastNdays)] = np.where(final_df['idx']>(test_split-lastNdays+28), 1, 0)
        test_df.loc[:,'last_'+str(lastNdays)] = 1

    # split features and target
    X_train, y_train = train_df[X_col], train_df[y_col]
    X_valid, y_valid = valid_df[X_col], valid_df[y_col]
    X_final, y_final = final_df[X_col], final_df[y_col]
    X_test = test_df[X_col]

    # convert to LGBM format
    lgbm_train = lgbm.Dataset(X_train, label=y_train, categorical_feature=X_cat)
    lgbm_valid = lgbm.Dataset(X_valid, label=y_valid, categorical_feature=X_cat)
    lgbm_final = lgbm.Dataset(X_final, label=y_final, categorical_feature=X_cat)
    
    # validation training 
    if params['boosting'] == 'dart':
        early_stopping_rounds = None
    else:
        pass
    train_model = lgbm.train(params, lgbm_train, valid_sets=lgbm_valid,
                             num_boost_round=num_boost_round, 
                             early_stopping_rounds=early_stopping_rounds,
                             categorical_feature=X_cat, verbose_eval=verbose)
    y_valid_pred = train_model.predict(X_valid)
    
    # final training and predictions
    n_rounds = train_model.best_iteration
    if n_rounds == 0:
        n_rounds = num_boost_round
    else:
        pass
    final_model = lgbm.train(params, lgbm_final, num_boost_round=n_rounds, 
                             categorical_feature=X_cat)
    if "sales_lag_1" in X_col:
        y_valid_pred_stag = staggered_pred(train_model, X_valid, round_preds)
        y_valid_check = y_valid_pred_stag
        y_eval_pred = staggered_pred(final_model, X_test, round_preds)
    else:
        y_eval_pred = final_model.predict(X_test)
        y_valid_check = y_valid_pred
        
    if round_preds:
        y_valid_check, y_eval_pred = np.array(y_valid_check), np.array(y_eval_pred)
        y_valid_check = np.round(np.where(y_valid_check<0, 0, y_valid_check))
        y_eval_pred = np.round(np.where(y_eval_pred<0, 0, y_eval_pred))
    else:
        pass
        
    sq_err = np.mean((y_valid_check - y_valid.values) ** 2)
    rmse = np.sqrt(sq_err)
    rmsse = np.sqrt(sq_err / (y_train.diff().dropna() ** 2).mean())
    if keep_score is None:
        if verbose:
            print(Fore.CYAN+"RMSSE: %.3f"%rmsse)
            print(Fore.CYAN+"RMSE: %.3f"%rmse)
            print(Fore.CYAN+"RMSE_train: %.3f"%train_model.best_score['valid_0']['rmse'])
            print(Fore.CYAN+"Number of Rounds: %d"%train_model.best_iteration)
        else:
            pass
    else:
        keep_score["RMSSE"].append(rmsse)
        keep_score["RMSE"].append(rmse)
        keep_score["RMSE_train"].append(train_model.best_score['valid_0']['rmse'])
        keep_score["Num_Rounds"].append(train_model.best_iteration)
    
    # plots
    if show_plots:
        fig, ax = plt.subplots(2, 1, facecolor='w', figsize=(12,12))
        ax[0].plot(y_valid_pred, 'b', label='predicted sales')
        if "sales_lag_1" in X_col:
            ax[0].plot(y_valid_check, ":r", label='predicted sales (staggered)')
        ax[0].plot(y_eval_pred, ":k", label='final predictions')
        ax[0].plot(y_valid.values, "g",  label='actual sales')
        ax[0].legend()
        lgbm.plot_importance(final_model, ax=ax[1], height=0.5, 
                             max_num_features=20, title=None)
        plt.show()
    else:
        pass
    
    if return_model:
        return final_model
    else:
        return y_valid_check, y_eval_pred

# -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
# Model training with binary feature -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\-
    
def staggered_pred0(model, model0, X_test):
    """
    Staggered predictions forward-filling AR terms.
    """
    preds = []
    for i in range(4):
        tmp = X_test.copy().iloc[i*7:(i+1)*7,:]        
        if i > 0:
            if "sales_lag_1" in tmp.columns:
                tmp.loc[:,"sales_lag_1"] = tmp_pred
            if "mean_sales_lag_1" in tmp.columns:    
                tmp.loc[:,"mean_sales_lag_1"] = np.mean(tmp_pred)
            if "std_sales_lag_1" in tmp.columns:
                tmp.loc[:,"std_sales_lag_1"] = np.std(tmp_pred)
            if "n0_sales_lag_1" in tmp.columns:
                tmp.loc[:,"n0_sales_lag_1"] = sum(tmp_pred < 0.5)
            if "max_sales_lag_1" in tmp.columns:
                tmp.loc[:,"max_sales_lag_1"] = max(tmp_pred)            
        else:
            pass
        tmp.loc[:, "pr0"] = model0.predict(tmp)
        tmp_pred = model.predict(tmp)
        preds += list(tmp_pred)
    return preds

def train_lgbm0(item_df, params, X_cat=["wday","month"], X_drop=[], 
                test_split=1940, lastNdays=90, num_boost_round=25, 
                verbose=False):
    """
    Trains LightGBM and produces both validation and evaluation predictions.
    """
    # separate target and features
    y_col = "TARGET"
    X_col = [c for c in item_df.columns if (c != y_col) and (c not in X_drop)]

    # split the data
    final_df = item_df.copy().query("idx <= @test_split")
    test_df = item_df.copy().query("idx > @test_split")

    # add last90days columns
    if lastNdays is not None:
        X_col.append('last_'+str(lastNdays))
        final_df.loc[:,'last_'+str(lastNdays)] = np.where(final_df['idx']>(test_split-lastNdays+28), 1, 0)
        test_df.loc[:,'last_'+str(lastNdays)] = 1

    # split features and target
    X_final, y_final = final_df.copy()[X_col], final_df.copy()[y_col]
    X_test = test_df.copy()[X_col]
    
    # pr0 estimates
    lgbm_final_0 = lgbm.Dataset(X_final, label=y_final>0, categorical_feature=X_cat)
    params0 = params.copy()
    params0["objective"] = "binary"
    train_model_0 = lgbm.train(params0, lgbm_final_0, num_boost_round=num_boost_round,
                               categorical_feature=X_cat, verbose_eval=verbose)
    X_final.loc[:,"pr0"] = train_model_0.predict(X_final)

    
    # convert to LGBM format
    lgbm_final = lgbm.Dataset(X_final, label=y_final, categorical_feature=X_cat)        
        
    # final training and predictions
    final_model = lgbm.train(params, lgbm_final, num_boost_round=num_boost_round, 
                             categorical_feature=X_cat)

    y_eval_pred = staggered_pred0(final_model, train_model_0, X_test)

    return y_eval_pred