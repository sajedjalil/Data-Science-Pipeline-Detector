# -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
# Generate predictions for each store \|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\-
# -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-

import os
import numpy as np 
import pandas as pd 
from m5_helper_s2 import load_stage2_data, gen_all_cal_feat, gen_item_df, train_lgbm

files = load_stage2_data()
cal_df = gen_all_cal_feat(files["calendar"])

store = "CA_1"
all_id = files["price_grps"].copy()[["id"]]
all_id.loc[:,"store"] = all_id["id"].apply(lambda x: "_".join(x.split("_")[3:5]))
item_list = all_id.query("store == @store")["id"].tolist()
item_list_val = [i.replace("evaluation","validation") for i in item_list]
this_sub = files["sample_submission"].copy().query("id in @item_list | id in @item_list_val")

params = {
    "objective": "rmse",
    "boosting": "gbdt",
    "eta": 0.0289,
    "bagging_freq": 1,
    "bagging_fraction": 0.88,
    "feature_fraction": 0.55,
    "min_data_in_leaf": 5
}
keep_score = {"id":[], "RMSSE":[], "RMSE":[], "RMSE_train":[], "Num_Rounds":[]}

for item_id, item_id_val in zip(item_list, item_list_val):
    item_df = gen_item_df(files, item_id, cal_df, verbose=False)
    y_valid_pred, y_eval_pred = train_lgbm(item_df, params, keep_score=keep_score, X_drop=['idx', 'wk_idx'])
    this_sub.loc[this_sub['id']==item_id_val, "F1":"F28"] = y_valid_pred
    this_sub.loc[this_sub['id']==item_id, "F1":"F28"] = y_eval_pred
    keep_score["id"].append(item_id)

scores = pd.DataFrame(keep_score)

scores.to_csv("Scores_"+store+".csv", index=False)
this_sub.to_csv("Submit_"+store+".csv", index=False)