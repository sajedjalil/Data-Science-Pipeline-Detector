# adjusted copy of https://www.kaggle.com/ceshine/a-simple-post-processing-trick-lb-0237-0204/code
import gc

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from builtins import list
from tqdm import tqdm

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

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

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

if __name__ == '__main__':

    print("Loading data from input files")
    
    print("Parts of the prediction")
    # Any results you write to the current directory are saved as output.
    #df_sub = pd.read_csv("../input/random-forest-regressor/rfr_submission_2_score_0.029005332340924082.csv")
    df_sub = pd.read_csv("../input/pubg-lgbm-regressor-fork/lgbm_submission_1.csv") # ../input/pubg-lgbm-regressor/   pubg-lgbm-regressor-fork
    df_sub2 = pd.read_csv("../input/pubg-lgbm-regressor-fork/lgbm_submission_2.csv")
    print("Merge 1")
    df_sub = df_sub.append(df_sub2, sort=False, ignore_index=True)
    del df_sub2
    df_sub2 = None
    df_sub2 = pd.read_csv("../input/pubg-lgbm-regressor-fork/lgbm_submission_3.csv")
    print("Merge 2")
    df_sub = df_sub.append(df_sub2, sort=False, ignore_index=True)
    del df_sub2
    df_sub2 = None
    
    df_sub.to_csv("lgbm_submission_4_raw.csv", index=False)
    
    # df_sub = reduce_mem_usage(df_sub)
    
    print("Test data...")
    df_test = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv") # ../input/pubg-finish-placement-prediction/
    # df_test = reduce_mem_usage(df_test)
    
    # Restore some columns
    df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "killPlace","maxPlace", "killPoints", "rankPoints", "winPoints", "numGroups", "kills"]], on="Id", how="left")
    df_sub["killPlace_2"] = -df_sub["killPlace"]
    df_sub["points_sum"] = df_sub["killPoints"] + df_sub["rankPoints"] + df_sub["winPoints"]
    df_sub["missing_groups_perc"] = (df_sub["maxPlace"] - df_sub["numGroups"]) / df_sub["maxPlace"]
    # df_sub.drop(columns="killPlace", inplace=True)
    df_sub.drop(columns="killPoints", inplace=True)
    df_sub.drop(columns="rankPoints", inplace=True)
    df_sub.drop(columns="winPoints", inplace=True)
    # df_sub = reduce_mem_usage(df_sub)
    
    del df_test
    df_test = None
    gc.collect()
    
    # unify team results, just in case
    # df_sub['tean_avg_winPlacePerc'] = df_sub.groupby(["matchId", "groupId"])['winPlacePerc'].transform('mean')
    # df_sub["winPlacePerc"] = df_sub["tean_avg_winPlacePerc"]
    # df_sub.drop(labels="tean_avg_winPlacePerc", axis=1, inplace=True)
    
    # Deal with edge cases
    df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0
    df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1
    # Edge case
    df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0
    
    # just in case
    df_sub.loc[df_sub.winPlacePerc < 0, "winPlacePerc"] = 0
    df_sub.loc[df_sub.winPlacePerc > 1, "winPlacePerc"] = 1
    
    # check for same place for different teams
    print("Checking for anomalies in the winPlacePerc - every group should have different score")
    df_sub["group_size"] = df_sub.groupby(["matchId", "groupId"])['Id'].transform('count')
    df_sub['winPlacePerc_size'] = df_sub.groupby(["matchId", 'winPlacePerc'])['Id'].transform('count')
    df_sub['winPlacePerc_size'] = df_sub['winPlacePerc_size'] / df_sub["group_size"]
    
    df_sub_2 = df_sub[df_sub["winPlacePerc_size"] != 1.0]
    
    print("Size of anomalies: " + str(len(df_sub_2)))
    
    if len(df_sub_2) > 0:
        df_sub_2.sort_values(ascending=False, by=["matchId", "winPlacePerc", "killPlace_2", "maxPlace", "points_sum", "groupId"], inplace=True)
    
        prev_match = "match_XXXX"
            
        for i in tqdm(range(len(df_sub_2)), desc="Correcting equal winPlacePerc", mininterval=2):
            row =  df_sub_2.iloc[i]
        #     print('matchId: ' + row["matchId"] + ", groupId: " + row["groupId"])
            if prev_match != row["matchId"]:
        #         print("New match!!!")
                mods = dict()
                next_penalty = 0.0
                group_to_skip = None
            prev_match = row["matchId"]
            
            # if "solo" in row["matchType"]:
            #     continue
            
            if row["groupId"] in mods.keys():
                penalty = mods[row["groupId"]]
        #         print("Penalty from cache: " + str(penalty))
            else:
                if group_to_skip == None:
                    group_to_skip = row["groupId"]
                elif group_to_skip != row["groupId"]:
                    next_penalty = next_penalty + 0.00001
                    penalty = next_penalty
                    mods[row["groupId"]] = penalty
        #             print("New penalty: " + str(penalty))
            if group_to_skip != row["groupId"]: 
                # df_sub.loc[(df_sub["Id"] == row["Id"]) & (df_sub["matchId"] == row["matchId"]) & (df_sub["groupId"] == row["groupId"]), "winPlacePerc"] = df_sub_2.winPlacePerc.iloc[i] - penalty
                df_sub_2.winPlacePerc.iloc[i] = df_sub_2.winPlacePerc.iloc[i] - penalty
                
        print("Updating winPlacePerc")
        df_sub.loc[df_sub["winPlacePerc_size"] != 1.0, "winPlacePerc"] = df_sub_2["winPlacePerc"]
        del df_sub_2
        df_sub_2 = None
        
        gc.collect()
            
        
        print("Finished correcting winPlacePerc")
    
    df_sub_2 = None
    
    # df_sub.drop(labels=["group_size"], axis=1, inplace=True)
    df_sub.drop(labels=["winPlacePerc_size"], axis=1, inplace=True)
    
    gc.collect()
    
    
    # based on observation from https://www.kaggle.com/plasticgrammer/pubg-finish-placement-prediction-playground
    print()
    print("Checking for anomalies in the winPlacePerc - players with same number of kills should have scores in order of killPlace")
    
    do_correct = True
    iteration_number = 1
    
    while do_correct & (iteration_number <= 1000):
        df_sub.sort_values(ascending=False, by=["matchId", "kills", "killPlace", "winPlacePerc", "groupId"], inplace=True)
        df_sub["winPlacePerc_diff"] = df_sub["winPlacePerc"].diff()
        df_sub["kills_diff"] = df_sub["kills"].diff()
        df_sub["prev_matchId"] = df_sub["matchId"].shift(1)
        df_sub["prev_groupId"] = df_sub["groupId"].shift(1)
        df_sub["prev_winPlacePerc"] = df_sub["winPlacePerc"].shift(1)
        
        df_sub2 = df_sub[(df_sub["winPlacePerc_diff"] < 0) & (df_sub["kills_diff"] == 0) & (df_sub["matchId"] == df_sub["prev_matchId"])]
        anomalies_count = len(df_sub2)
        
        print("Iteration " + str(iteration_number) + " Anomalies count: " + str(anomalies_count))
        
        changed_groups = list()
        
        if anomalies_count > 0:
            print()
            print("Looking for pairs to change...")
            
            df_sub2["new_winPlacePerc"] = df_sub2["winPlacePerc"] 
            
            df_sub3 = pd.DataFrame()
            
            for i in tqdm(range(1, min(15001, max(anomalies_count, 2))), desc="Identifying unique groups", mininterval=10):
                row = df_sub2.iloc[i - 1]
                    
                id_prev = str(row["prev_matchId"]) + "!" + str(row["prev_groupId"])
                id_cur = str(row["matchId"]) + "!" + str(row["groupId"])
                
                if (not id_prev in changed_groups) & (not id_cur in changed_groups):
                    changed_groups.append(id_prev)
                    changed_groups.append(id_cur)
                    
                    df_sub3 = df_sub3.append({"matchId" : row["matchId"], "groupId" : row["prev_groupId"], "new_winPlacePerc" : row["winPlacePerc"]}, sort=False, ignore_index=True)
                    df_sub3 = df_sub3.append({"matchId" : row["matchId"], "groupId" : row["groupId"], "new_winPlacePerc" : row["prev_winPlacePerc"]}, sort=False, ignore_index=True)
            
            df_sub3.drop_duplicates(inplace=True)
            df_sub = df_sub.merge(df_sub3, on=["matchId", "groupId"], how="left")
            df_sub.loc[df_sub["new_winPlacePerc"].notna(), "winPlacePerc"] = df_sub.loc[df_sub["new_winPlacePerc"].notna()]["new_winPlacePerc"]       
            df_sub.drop(labels="new_winPlacePerc", axis=1, inplace=True)
            del df_sub2
            del df_sub3
            df_sub2 = None
            df_sub3 = None
            gc.collect()
        else:
            do_correct = False
        
        iteration_number = iteration_number + 1
        
    if do_correct:
        print("Limit of iterations reached...")
    
    print("Finished correcting winPlacePerc")
    
    # print()
    # print()
    # print("Adjusting winPlacePerc with its rank and numGroups and then with maxPlace where missing groups percentage > 0.6")
    # # Sort, rank, and assign adjusted ratio
    # df_sub_group = df_sub.loc[df_sub["missing_groups_perc"] < 0.6].groupby(["matchId", "groupId"]).first().reset_index()
    # df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()
    # df_sub_group = df_sub_group.merge(
    #     df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 
    #     on="matchId", how="left")
    # df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)
    
    # # Align with maxPlace
    # # Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
    # subset = df_sub_group.loc[df_sub_group.maxPlace > 1]
    # gap = 1.0 / (subset.maxPlace.values - 1)
    # new_perc = np.around(subset.winPlacePerc.values / gap) * gap
    # df_sub_group.loc[df_sub_group.maxPlace > 1, "winPlacePerc"] = new_perc
    
    # df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
    # df_sub.loc[df_sub["missing_groups_perc"] < 0.6, "winPlacePerc"] = df_sub.loc[df_sub["missing_groups_perc"] < 0.6]["adjusted_perc"]
    
    # del df_sub_group
    # df_sub_group = None
    # gc.collect()
    
    # print()
    # print()
    # print("Fixing placing where missing groups percentage > 0.6")
    
    # print()
    # print("Aligning against maxPlace")
    
    # df_sub.loc[df_sub["missing_groups_perc"] > 0.6, "winPlacePerc"] = 1.0 - df_sub.loc[df_sub["missing_groups_perc"] > 0.6]["winPlacePerc"]
    # subset = df_sub.loc[df_sub["missing_groups_perc"] > 0.6]
    
    # subset["gap"] = 1.0 / (subset["maxPlace"] - 1)
    # subset["aligned_winPlacePerc"] = np.around(subset["winPlacePerc"] / subset["gap"])# * subset["gap"]
    # subset["aligned_winPlacePerc"] = subset["aligned_winPlacePerc"].astype(int)
    
    # print("Phase 1 - pushing down")
    
    # do_correct = True
    # iteration_number = 1
    
    # while do_correct & (iteration_number <= 1000):
    #     subset.sort_values(ascending=False, by=["matchId", "aligned_winPlacePerc", "winPlacePerc", "groupId"], inplace=True)
        
    #     subset["aligned_winPlacePerc_diff"] = subset["aligned_winPlacePerc"].diff()
    #     subset["prev_matchId"] = subset["matchId"].shift(1)
    #     subset["prev_groupId"] = subset["groupId"].shift(1)
    #     subset["prev_aligned_winPlacePerc"] = subset["aligned_winPlacePerc"].shift(1)
        
    #     df_sub2 = subset[(subset["aligned_winPlacePerc_diff"] == 0) & (subset["groupId"] != subset["prev_groupId"]) & (subset["matchId"] == subset["prev_matchId"])]
    #     anomalies_count = len(df_sub2)
        
    #     print("Iteration " + str(iteration_number) + " Anomalies count: " + str(anomalies_count))
        
    #     if anomalies_count > 0:
            
    #         df_sub2["new_aligned_winPlacePerc"] = df_sub2["aligned_winPlacePerc"] - 1
            
    #         df_sub2 = df_sub2[["new_aligned_winPlacePerc", "matchId", "groupId"]]
    #         df_sub2.drop_duplicates(inplace=True)
            
    #         subset = subset.merge(df_sub2, on=["matchId", "groupId"], how="left")
    #         subset.loc[subset["new_aligned_winPlacePerc"].notna(), "aligned_winPlacePerc"] = subset.loc[subset["new_aligned_winPlacePerc"].notna()]["new_aligned_winPlacePerc"]
    #         subset.drop(labels="new_aligned_winPlacePerc", axis=1, inplace=True)
    #         del df_sub2
    #         df_sub2 = None
    #         gc.collect()
    #     else:
    #         do_correct = False
        
    #     iteration_number = iteration_number + 1
        
    # if do_correct:
    #     print("Limit of iterations reached...")
    
    
    # print("Phase 2 - pushing up where < 0")
    
    # subset.loc[subset["aligned_winPlacePerc"] < 0, "aligned_winPlacePerc"] = 0
    
    
    # print("Phase 3 - pushing up")
    
    # do_correct = True
    # iteration_number = 1
    
    # while do_correct & (iteration_number <= 1000):
    #     subset.sort_values(ascending=True, by=["matchId", "aligned_winPlacePerc", "winPlacePerc", "groupId"], inplace=True)
        
    #     subset["aligned_winPlacePerc_diff"] = subset["aligned_winPlacePerc"].diff()
    #     subset["prev_matchId"] = subset["matchId"].shift(1)
    #     subset["prev_groupId"] = subset["groupId"].shift(1)
    #     subset["prev_aligned_winPlacePerc"] = subset["aligned_winPlacePerc"].shift(1)
        
    #     df_sub2 = subset[(subset["aligned_winPlacePerc_diff"] == 0) & (subset["groupId"] != subset["prev_groupId"]) & (subset["matchId"] == subset["prev_matchId"])]
    #     anomalies_count = len(df_sub2)
        
    #     print("Iteration " + str(iteration_number) + " Anomalies count: " + str(anomalies_count))
        
    #     if anomalies_count > 0:
            
    #         df_sub2["new_aligned_winPlacePerc"] = df_sub2["aligned_winPlacePerc"] + 1
            
    #         df_sub2 = df_sub2[["new_aligned_winPlacePerc", "matchId", "groupId"]]
    #         df_sub2.drop_duplicates(inplace=True)
            
    #         subset = subset.merge(df_sub2, on=["matchId", "groupId"], how="left")
    #         subset.loc[subset["new_aligned_winPlacePerc"].notna(), "aligned_winPlacePerc"] = subset.loc[subset["new_aligned_winPlacePerc"].notna()]["new_aligned_winPlacePerc"]
    #         subset.drop(labels="new_aligned_winPlacePerc", axis=1, inplace=True)
    #         del df_sub2
    #         df_sub2 = None
    #         gc.collect()
    #     else:
    #         do_correct = False
        
    #     iteration_number = iteration_number + 1
        
    # if do_correct:
    #     print("Limit of iterations reached...")
    
    # subset["aligned_winPlacePerc"] = subset["aligned_winPlacePerc"] * subset["gap"]
    # print("Number of negative scores: " + str(len(subset.loc[subset["aligned_winPlacePerc"] < 0])))
    # print("Number of scores greater then 1: " + str(len(subset.loc[subset["aligned_winPlacePerc"] > 1])))
    # subset = subset[["aligned_winPlacePerc", "matchId", "groupId"]]
    # subset.drop_duplicates(inplace=True)
    
    # df_sub = df_sub.merge(subset, on=["matchId", "groupId"], how="left")
    # df_sub.loc[df_sub["aligned_winPlacePerc"].notna(), "winPlacePerc"] = df_sub.loc[df_sub["aligned_winPlacePerc"].notna()]["aligned_winPlacePerc"]
    
    # print("Finished fixing placing where numGroups < maxPlace")
    
    # original alinging code
    print("Adjusting winPlacePerc with its rank and numGroups")
    # Sort, rank, and assign adjusted ratio
    df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()
    df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()
    df_sub_group = df_sub_group.merge(
        df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 
        on="matchId", how="left")
    df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)
    
    df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
    df_sub["winPlacePerc"] = df_sub["adjusted_perc"]
    
    print("Aligning with maxPlace")
    # Align with maxPlace
    # Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
    subset = df_sub.loc[df_sub.maxPlace > 1]
    gap = 1.0 / (subset.maxPlace.values - 1)
    new_perc = np.around(subset.winPlacePerc.values / gap) * gap
    df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc
    
    # Edge case
    df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0
    df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1
    df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0
    
    #another constant cases
    subset = df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 2)]
    subset["group_size_rank"] = subset.groupby(["matchId", "groupId"])["group_size"].rank(ascending=True, pct=False)
    subset = subset.merge(
        subset.groupby("matchId")["group_size_rank"].max().to_frame("group_size_max_rank").reset_index(), 
        on="matchId", how="left")
    subset.loc[subset["group_size_rank"] != subset["group_size_max_rank"], "winPlacePerc"] = 1
    subset.loc[subset["group_size_rank"] == subset["group_size_max_rank"], "winPlacePerc"] = 0
    subset["new_winPlacePerc2"] = subset["winPlacePerc"]
    df_sub = df_sub.merge(subset[["Id", "matchId", "groupId", "new_winPlacePerc2"]], on=["Id", "matchId", "groupId"], how="left")
    df_sub.loc[df_sub["new_winPlacePerc2"].notna(), "winPlacePerc"] = df_sub.loc[df_sub["new_winPlacePerc2"].notna()]["new_winPlacePerc2"]
    
    assert df_sub["winPlacePerc"].isnull().sum() == 0
    # df_sub["winPlacePerc"] = np.around(df_sub["winPlacePerc"], decimals=4)
    
    print("Storing final submission to file...")
    df_sub[["Id", "winPlacePerc"]].to_csv("lgbm_submission_4_adjusted.csv", index=False)
    
    print("Done.")