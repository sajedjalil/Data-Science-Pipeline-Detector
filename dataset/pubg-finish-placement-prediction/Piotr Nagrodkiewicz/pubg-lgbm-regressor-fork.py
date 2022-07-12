import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.metrics import mean_absolute_error
import gc
#from random import shuffle
import lightgbm as lgb
import os

# parameters
TRAIN_FILE_NAME = '../input/train_V2.csv' #../input/
TEST_FILE_NAME = '../input/test_V2.csv'#../input/

GRAPHS_FOLDER = "graphs/"

# functions

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
 
# def convert_match_types_to_numbers(match):
#     """
#     Converts given match to numerical representation.
#     :param match: Match to be converted.
#     :return: Converted representation
#     """
#     if "solo" == match:
#         return 0.3
#     
#     if "duo" == match:
#         return 0.6
#     
#     return 1.0
# 
# def merge_match_types(match):
#     """
#     Mapper used to convert original match types to unified types (solo, duo & mutli[player]).
#     :param match: Original match type.
#     :return: Converted value.
#     """
#     if "solo" in match:
#         return "solo"
# 
#     if "duo" == match:
#         return "duo"
#     
#     return "multi"

# def do_data_analysis(data):
#     """
#     Does data analysis
#     :param data: Data to be analysed
#     :return: nothing
#     """
#     print()
#     print("Generating heatmap...")
#     print()
#     corrmat = data.corr()
#     plt.subplots(figsize=(16,12))
#     sns.set(font_scale=0.7)
#     hm = sns.heatmap(corrmat, annot=True)
#     hm.get_figure().savefig(GRAPHS_FOLDER + "heatmap.png")
#     plt.clf()
#     print()

#     for column in data.columns:
        
#         print()
#         print("Column: " + column)
#         print()
#         print(str(data[column].describe()))
        
#         # histogram
#         if column == "matchType":
#             continue
        
#         print()
#         print("Generating graphs...")
#         print()

#         plt.subplots(figsize=(16,12))
#         axis = sns.distplot(data[column], kde=True)
#         axis.get_figure().savefig(GRAPHS_FOLDER + column + "_histogram.png")
#         plt.clf()
        
#         if column == "winPlacePerc":
#             continue
        
#         plt.subplots(figsize=(16,12))
#         #data2 = pd.concat([data['winPlacePerc'], data[column]], axis=1)
#         #axis = data2.plot.scatter(x=column, y='winPlacePerc', ylim=(0, 1))
#         #axis.get_figure().savefig(GRAPHS_FOLDER + column + "_vs_winPlacePerc_scatter.png")        
#         data2 = pd.concat([data['winPlacePerc'], data[column], data['matchType']], axis=1)
#         plot = sns.relplot(x=column, y="winPlacePerc", hue="matchType", data=data2)
#         plot.savefig(GRAPHS_FOLDER + column + "_vs_winPlacePerc_scatter.png")
#         plt.clf()
    
#     return

# v15
# UNWANTED_FEATURES = ("healsPerWalkDistance_team_match_max", "rankPoints_team_match_median", "heals_team_to_match_ratio_max", "picker_match_median", "sniper_team_to_match_ratio_mean", "longestKill_team_match_min", "kill_to_team_kills_ratio_match_sum", "longestKill_team_match_std", "roadKills_team_match_mean_rank", "healsAndBoostsPerWalkDistance_team_match_min", "sniper_team_match_mean_rank", "boostsPerWalkDistance_team_match_median", "rideDistance_team_to_match_ratio_mean", "winPoints_team_match_max", "headshot_rate_team_to_match_ratio_mean", "winPoints_match_median", "swimDistance_team_match_sum_rank", "killPoints_team_match_median", "vehicleDestroys_team_match_median_rank", "DBNOs_team_to_match_ratio_max", "killPoints_team_match_mean", "multi_killer_match_sum", "weaponsAcquired_team_match_mean", "healsPerWalkDistance_team_match_min", "heals_team_to_match_ratio_median", "kills_team_to_match_ratio_max", "headshot_rate_team_match_max_rank", "rideDistance_team_match_std", "boostsPerWalkDistance_team_match_min", "killPoints_team_match_sum", "rideDistance_team_to_match_ratio_max", "sniper_team_match_sum_rank", "health_items_match_median", "winPoints_team_match_sum", "revives_team_to_match_ratio_sum", "non_leathal_input_team_to_match_ratio_max", "DBNOs_match_max", "kills_and_assists_team_to_match_ratio_max", "headshot_rate_team_match_sum_rank", "killStreaks_team_match_var", "skill_team_match_sum_rank", "assists_team_to_match_ratio_sum", "revives_team_to_match_ratio_median", "headshotKills_match_max", "headshot_rate_team_match_var", "distance_over_weapons_team_match_std", "rideDistance_team_match_sum", "headshotKills_team_match_max_rank", "winPoints_team_match_mean", "rideDistance_team_to_match_ratio_median", "sniper_team_match_var", "sniper_team_match_median_rank", "boosts_team_to_match_ratio_max", "headshotKills_team_to_match_ratio_mean", "rideDistance_team_match_max", "headshot_rate_team_match_min_rank", "teamKills_team_match_sum_rank", "weaponsAcquired_match_median", "assists_team_to_match_ratio_median", "winPoints_team_match_median", "rideDistance_team_to_match_ratio_sum", "skill_match_max", "picker_team_match_sum", "rideDistance_team_match_mean", "kill_to_team_kills_ratio_team_match_min_rank", "swimDistance_team_to_match_ratio_mean", "assists_match_max", "killStreakrate_team_match_mean", "skill_team_match_max_rank", "teamKills_match_sum", "kill_to_team_kills_ratio_match_max", "sniper_team_to_match_ratio_median", "roadKills_team_match_median_rank", "killStreaks_match_max", "rideDistance_team_match_min", "headshotKills_team_match_sum_rank", "kills_and_assists_match_median", "assists_team_match_var", "weaponsAcquired_team_match_sum", "teamKills_team_to_match_ratio_mean", "rideDistance_team_match_var", "killStreakrate_team_match_sum", "revives_team_match_var", "headshotKills_team_to_match_ratio_sum", "headshot_rate_team_to_match_ratio_max", "headshot_rate_team_to_match_ratio_median", "health_items_team_match_mean", "skill_team_to_match_ratio_mean", "rideDistance_team_match_median", "swimDistance_team_to_match_ratio_sum", "revives_match_max", "health_items_team_match_sum", "kill_to_team_kills_ratio_team_match_sum_rank", "heals_team_match_mean", "rankPoints_team_match_std", "skill_team_to_match_ratio_sum", "headshotKills_team_match_var", "killPlace_team_match_std", "swimDistance_team_match_var", "vehicleDestroys_team_match_min_rank", "swimDistance_team_to_match_ratio_median", "swimDistance_team_to_match_ratio_max", "roadKills_team_match_min_rank", "boosts_match_median", "heals_team_match_sum", "picker_team_match_std", "non_leathal_input_team_match_mean", "heals_match_median", "picker_team_match_max", "boosts_team_match_mean", "swimDistance_team_match_max_rank", "DBNOs_match_median", "headshotKills_team_to_match_ratio_median", "weaponsAcquired_team_match_std", "skill_team_to_match_ratio_median", "killPoints_team_match_std", "DBNOs_team_match_mean", "picker_team_match_median", "sniper_team_match_mean", "sniper_team_match_sum", "swimDistance_team_match_max", "winPoints_team_match_std", "sniper_team_match_min_rank", "headshot_rate_team_match_mean", "swimDistance_team_match_mean", "picker_team_match_min", "health_items_team_match_std", "vehicleDestroys_team_match_sum_rank", "killStreaks_team_to_match_ratio_max", "weaponsAcquired_team_match_median", "multi_killer_team_match_min_rank", "kills_match_median", "swimDistance_team_match_std", "assists_team_to_match_ratio_max", "swimDistance_team_match_sum", "teamKills_team_match_max_rank", "health_items_team_match_max", "healsAndBoostsPerWalkDistance_team_match_std", "weaponsAcquired_team_match_max", "kill_to_team_kills_ratio_team_to_match_ratio_mean", "revives_team_match_mean", "killStreakrate_team_match_median", "non_leathal_input_match_median", "killsPerWalkDistance_team_match_std", "swimDistance_team_match_min", "multi_killer_team_match_sum_rank", "assists_team_match_mean", "heals_team_match_median", "teamKills_match_max", "distance_over_weapons_team_to_match_ratio_min", "headshot_rate_team_match_sum", "multi_killer_match_max", "swimDistance_team_match_median", "health_items_team_match_median", "boosts_team_match_sum", "boosts_team_match_max", "skill_team_match_var", "healsPerWalkDistance_team_match_std", "boostsPerWalkDistance_team_match_std", "kills_team_match_mean", "heals_team_match_max", "killStreakrate_team_match_std", "weaponsAcquired_team_match_min", "roadKills_team_match_sum_rank", "headshotKills_team_to_match_ratio_max", "revives_team_to_match_ratio_max", "heals_team_match_std", "vehicleDestroys_team_to_match_ratio_mean", "non_leathal_input_team_match_std", "health_items_team_match_min", "non_leathal_input_team_match_sum", "kills_without_moving_match_mean", "boosts_team_match_std", "kills_team_match_std", "killStreaks_team_match_mean", "kills_and_assists_team_match_std", "walkDistance_team_to_match_ratio_sum", "teamKills_team_to_match_ratio_median", "DBNOs_team_match_sum", "picker_team_to_match_ratio_min", "DBNOs_team_match_std", "teamKills_team_to_match_ratio_sum", "teamKills_team_match_var", "headshot_rate_team_match_max", "kills_and_assists_team_match_median", "multi_killer_team_to_match_ratio_mean", "distance_over_weapons_match_min", "sniper_team_match_median", "weaponsAcquired_team_to_match_ratio_min", "kills_and_assists_team_match_sum", "kills_team_match_max", "kills_without_moving_team_match_median_rank", "vehicleDestroys_match_sum", "roadKills_team_to_match_ratio_mean", "boosts_team_match_median", "headshot_rate_team_match_median", "kills_team_match_min", "heals_team_match_min", "kills_without_moving_team_match_min_rank", "headshot_rate_team_match_std", "skill_team_to_match_ratio_max", "headshotKills_team_match_mean", "kills_team_match_median", "killStreaks_match_median", "sniper_team_match_std", "non_leathal_input_team_match_max", "kills_team_match_sum", "assists_team_match_sum", "DBNOs_team_match_max", "sniper_team_match_max", "assists_team_match_std", "non_leathal_input_team_match_median", "killStreaks_team_match_std", "kills_and_assists_team_match_max", "teamKills_team_match_mean", "kill_to_team_kills_ratio_team_match_var", "total_distance_team_to_match_ratio_sum", "kill_to_team_kills_ratio_team_match_max_rank", "vehicleDestroys_team_match_max_rank", "roadKills_match_sum", "DBNOs_team_match_median", "killPlace_match_max", "sniper_team_to_match_ratio_max", "kill_to_team_kills_ratio_team_to_match_ratio_sum", "revives_team_match_std", "boosts_team_match_min", "kill_to_team_kills_ratio_team_match_mean", "killStreaks_team_match_sum", "revives_team_match_sum", "roadKills_match_max", "killStreakrate_team_match_max", "kill_to_team_kills_ratio_team_to_match_ratio_median", "non_leathal_input_team_match_min", "skill_team_match_mean", "kills_without_moving_team_match_sum_rank", "walkDistance_match_sum", "kills_and_assists_team_match_min", "headshotKills_team_match_sum", "multi_killer_team_match_max_rank", "sniper_team_match_min", "vehicleDestroys_match_max", "multi_killer_team_match_var", "DBNOs_team_match_min", "headshot_rate_team_match_min", "kills_without_moving_team_match_max_rank", "roadKills_team_match_max_rank", "assists_team_match_max", "vehicleDestroys_team_to_match_ratio_median", "revives_team_match_max", "skill_team_match_sum", "killStreaks_team_match_max", "headshotKills_team_match_max", "killStreakrate_team_to_match_ratio_max", "sniper_match_max", "revives_team_match_median", "skill_team_match_std", "kill_to_team_kills_ratio_team_match_sum", "vehicleDestroys_team_match_var", "headshotKills_team_match_std", "multi_killer_team_to_match_ratio_sum", "assists_team_match_median", "multi_killer_team_to_match_ratio_median", "kill_to_team_kills_ratio_team_to_match_ratio_max", "damageDealt_team_to_match_ratio_min", "headshotKills_team_match_median", "longestKill_team_to_match_ratio_min", "multi_killer_team_match_mean", "assists_team_match_min", "teamKills_team_to_match_ratio_max", "killStreaks_team_match_median", "skill_team_match_max", "roadKills_team_to_match_ratio_median", "non_leathal_input_team_to_match_ratio_min", "vehicleDestroys_team_to_match_ratio_sum", "teamKills_team_match_std", "roadKills_team_match_var", "multi_killer_team_to_match_ratio_max", "killStreakrate_team_to_match_ratio_min", "multi_killer_team_match_sum", "roadKills_team_to_match_ratio_sum", "kill_to_team_kills_ratio_team_match_std", "roadKills_team_match_mean", "vehicleDestroys_team_match_mean", "teamKills_team_match_sum", "kills_and_assists_team_to_match_ratio_min", "sniper_match_median", "kill_to_team_kills_ratio_team_match_max", "weaponsAcquired_match_min", "total_distance_match_sum", "kills_without_moving_match_sum", "killsPerWalkDistance_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_match_median", "killStreaks_team_match_min", "revives_team_match_min", "kills_without_moving_team_match_var", "skill_team_match_median", "picker_match_min", "kills_team_to_match_ratio_min", "headshotKills_team_match_min", "assists_match_median", "killsPerWalkDistance_match_min", "multi_killer_team_match_median", "multi_killer_team_match_std", "skill_team_match_min", "longestKill_match_min", "matchDuration", "headshotKills_match_median", "teamKills_team_match_max", "roadKills_team_match_std", "vehicleDestroys_team_match_sum", "killStreaks_team_to_match_ratio_min", "kills_without_moving_team_match_mean", "vehicleDestroys_team_match_max", "damageDealt_match_min", "roadKills_team_to_match_ratio_max", "killStreakrate_match_min", "assists_team_to_match_ratio_min", "teamKills_team_match_median", "vehicleDestroys_team_match_std", "multi_killer_team_match_max", "skill_match_median", "roadKills_team_match_sum", "kills_without_moving_team_match_sum", "headshot_rate_match_median", "roadKills_team_match_max", "DBNOs_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_match_min", "teamKills_team_match_min", "kills_without_moving_team_match_std", "killStreakrate_match_max", "healsAndBoostsPerWalkDistance_match_min", "kills_match_min", "health_items_team_to_match_ratio_min", "healsAndBoostsPerWalkDistance_team_to_match_ratio_min", "revives_match_median", "rideDistance_match_min", "rideDistance_team_to_match_ratio_min", "multi_killer_team_match_min", "vehicleDestroys_team_to_match_ratio_max", "vehicleDestroys_team_match_median", "health_items_match_min", "killStreaks_match_min", "non_leathal_input_match_min", "roadKills_team_match_median", "swimDistance_match_median", "healsPerWalkDistance_team_to_match_ratio_min", "headshot_rate_match_min", "kills_without_moving_team_match_median", "roadKills_match_median", "teamKills_match_median", "vehicleDestroys_match_median", "kills_without_moving_match_median", "kill_to_team_kills_ratio_match_median", "multi_killer_match_median", "roadKills_team_match_min", "vehicleDestroys_team_match_min", "kills_without_moving_team_match_min", "assists_match_min", "boosts_match_min", "DBNOs_match_min", "headshotKills_match_min", "heals_match_min", "killPlace_match_min", "revives_match_min", "roadKills_match_min", "swimDistance_match_min", "teamKills_match_min", "vehicleDestroys_match_min", "kills_and_assists_match_min", "kills_without_moving_match_min", "boostsPerWalkDistance_match_min", "healsPerWalkDistance_match_min", "skill_match_min", "kill_to_team_kills_ratio_match_min", "multi_killer_match_min", "sniper_match_min", "boosts_team_to_match_ratio_min", "headshotKills_team_to_match_ratio_min", "heals_team_to_match_ratio_min", "revives_team_to_match_ratio_min", "roadKills_team_to_match_ratio_min", "swimDistance_team_to_match_ratio_min", "teamKills_team_to_match_ratio_min", "vehicleDestroys_team_to_match_ratio_min", "boostsPerWalkDistance_team_to_match_ratio_min", "headshot_rate_team_to_match_ratio_min", "skill_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_to_match_ratio_min", "multi_killer_team_to_match_ratio_min", "sniper_team_to_match_ratio_min", "kills_without_moving_team_match_max", "kills_without_moving_match_max", "kill_to_team_kills_ratio_team_match_median_rank", "healsAndBoostsPerWalkDistance_team_to_match_ratio_max", "kills_and_assists_team_to_match_ratio_mean", "longestKill_team_match_max", "health_items_team_match_var", "killPoints_match_median", "healsAndBoostsPerWalkDistance_team_match_min_rank", "DBNOs_team_match_var", "headshot_rate_team_match_median_rank", "boostsPerWalkDistance_team_match_min_rank", "healsPerWalkDistance_team_match_median_rank", "killPoints_team_match_max", "assists_team_match_max_rank", "headshot_rate_team_to_match_ratio_sum", "walkDistance_match_min", "DBNOs_team_to_match_ratio_median", "rideDistance_team_match_max_rank", "non_leathal_input_match_max", "health_items_team_match_max_rank", "winPoints_match_mean", "heals_team_match_sum_rank", "total_distance_match_min", "revives_team_to_match_ratio_mean", "healsAndBoostsPerWalkDistance_team_to_match_ratio_sum", "heals_team_match_max_rank", "damageDealt_team_match_min", "healsPerWalkDistance_team_match_var", "assists_team_to_match_ratio_mean", "healsPerWalkDistance_team_to_match_ratio_max", "boostsPerWalkDistance_team_to_match_ratio_mean", "teamKills_team_match_min_rank", "boostsPerWalkDistance_team_match_var", "boostsPerWalkDistance_team_match_sum", "killStreakrate_team_match_var", "winPoints_team_match_min", "killPoints_team_match_min", "healsAndBoostsPerWalkDistance_team_match_sum_rank", "health_items_team_to_match_ratio_sum", "rideDistance_match_median", "revives_team_match_max_rank", "boostsPerWalkDistance_team_match_max", "killPoints_match_min", "boostsPerWalkDistance_team_match_mean", "boostsPerWalkDistance_team_match_sum_rank", "killStreaks_team_to_match_ratio_median", "killStreaks_team_to_match_ratio_sum", "healsAndBoostsPerWalkDistance_team_match_max", "healsPerWalkDistance_team_to_match_ratio_sum", "killsPerWalkDistance_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_max_rank", "heals_team_to_match_ratio_mean", "heals_team_match_var", "kills_and_assists_team_match_var", "boostsPerWalkDistance_team_match_max_rank", "non_leathal_input_team_to_match_ratio_median", "boosts_team_match_var", "healsPerWalkDistance_team_match_sum", "kills_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_mean", "heals_team_to_match_ratio_sum", "healsAndBoostsPerWalkDistance_team_to_match_ratio_mean", "healsAndBoostsPerWalkDistance_team_match_sum", "multi_killer_team_match_median_rank", "healsPerWalkDistance_team_to_match_ratio_mean", "healsPerWalkDistance_team_match_sum_rank", "healsPerWalkDistance_team_match_mean", "healsPerWalkDistance_team_match_max_rank", "health_items_team_to_match_ratio_max", "boostsPerWalkDistance_team_to_match_ratio_median", "longestKill_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_to_match_ratio_median", "kills_and_assists_team_to_match_ratio_median", "longestKill_team_match_median", "boosts_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_median", "killsPerWalkDistance_team_match_median", "health_items_team_to_match_ratio_median", "healsPerWalkDistance_team_to_match_ratio_median", "healsPerWalkDistance_team_match_median", "killPlace_over_maxPlace_team_match_sum_rank", "kills_and_assists_team_match_mean", "killPoints_team_to_match_ratio_mean", "rankPoints_team_match_sum", "health_items_team_match_sum_rank", "damageDealt_team_match_std", "rankPoints_match_median", "killPoints_match_sum", "weaponsAcquired_match_max", "vehicleDestroys_team_match_mean_rank", "damageDealt_team_to_match_ratio_median", "boostsPerWalkDistance_team_to_match_ratio_sum", "boostsPerWalkDistance_team_match_median_rank", "damageDealt_team_match_median", "assists_team_match_sum_rank", "longestKill_team_to_match_ratio_max", "headshot_rate_team_match_mean_rank", "winPoints_team_match_median_rank", "non_leathal_input_match_sum", "killPoints_team_match_sum_rank", "killStreakrate_team_match_median_rank", "winPoints_team_match_sum_rank", "killPoints_team_match_median_rank", "winPoints_team_to_match_ratio_median", "DBNOs_team_to_match_ratio_sum", "killStreaks_team_match_sum_rank", "assists_match_sum", "killPoints_team_match_mean_rank", "healsPerWalkDistance_team_match_min_rank", "boostsPerWalkDistance_team_to_match_ratio_max", "revives_team_match_sum_rank", "roadKills_match_mean", "DBNOs_match_sum", "killPoints_match_mean", "winPoints_team_match_mean_rank", "headshotKills_match_sum", "longestKill_team_match_mean", "killStreaks_team_match_median_rank", "skill_match_sum", "killsPerWalkDistance_team_match_mean", "multi_killer_team_match_mean_rank", "boosts_match_max", "longestKill_team_to_match_ratio_mean", "rankPoints_team_match_max", "longestKill_team_match_sum_rank", "longestKill_team_match_max_rank", "killsPerWalkDistance_team_match_sum", "killsPerWalkDistance_team_match_var", "non_leathal_input_team_match_var", "sniper_team_match_max_rank", "rankPoints_team_match_min", "non_leathal_input_team_to_match_ratio_sum", "revives_match_sum", "sniper_team_to_match_ratio_sum", "kills_and_assists_team_to_match_ratio_sum", "rankPoints_team_match_mean", "killPlace_over_maxPlace_team_match_std", "group_size", "killPlace_over_maxPlace_team_to_match_ratio_sum", "killPlace_over_maxPlace_team_match_min_rank", "killPlace_over_maxPlace_team_match_mean_rank", "kills_without_moving_team_match_mean_rank", "killPlace_team_to_match_ratio_min", "killPlace_match_sum", "killPlace_over_maxPlace_match_min", "heals_match_max", "distance_over_weapons_team_match_sum", "damageDealt_team_match_mean", "boosts_team_match_mean_rank", "revives_team_match_min_rank", "damageDealt_team_to_match_ratio_mean", "weaponsAcquired_team_match_mean_rank", "damageDealt_team_to_match_ratio_sum", "killPoints_team_match_min_rank", "picker_team_to_match_ratio_median", "killPoints_team_to_match_ratio_sum", "total_distance_team_match_min_rank", "non_leathal_input_team_match_max_rank", "kills_match_sum", "rankPoints_team_match_median_rank", "kills_and_assists_team_match_mean_rank", "kill_to_team_kills_ratio_team_match_mean_rank", "distance_over_weapons_team_match_median_rank", "headshotKills_team_match_min_rank", "rankPoints_team_match_mean_rank", "picker_team_match_median_rank", "DBNOs_team_match_max_rank", "damageDealt_team_match_max", "distance_over_weapons_team_match_mean", "distance_over_weapons_team_match_median", "healsAndBoostsPerWalkDistance_team_match_median_rank", "total_distance_team_match_sum", "winPoints_team_to_match_ratio_sum", "non_leathal_input_team_match_mean_rank", "DBNOs_team_match_sum_rank", "non_leathal_input_team_match_sum_rank", "distance_over_weapons_team_match_mean_rank", "total_distance_team_match_max", "kills_and_assists_team_match_sum_rank", "DBNOs_team_match_mean_rank", "picker_team_to_match_ratio_max", "total_distance_team_match_std", "walkDistance_team_to_match_ratio_min", "total_distance_team_match_min", "walkDistance_team_match_std", "teamKills_team_match_median_rank", "non_leathal_input_team_to_match_ratio_mean", "damageDealt_team_match_sum", "killStreakrate_team_to_match_ratio_median", "killStreaks_team_match_mean_rank", "rideDistance_team_match_sum_rank", "killPoints_match_max", "rankPoints_match_mean", "total_distance_team_match_mean", "weaponsAcquired_team_to_match_ratio_max", "winPoints_match_sum", "kills_and_assists_team_match_max_rank", "killPoints_team_match_var", "killPlace_team_match_sum", "killStreakrate_match_median", "rideDistance_match_sum", "winPoints_match_max", "boosts_team_to_match_ratio_sum", "boosts_team_to_match_ratio_mean", "total_distance_team_to_match_ratio_min", "headshot_rate_match_max", "skill_team_match_min_rank", "winPoints_team_match_var", "picker_match_max", "winPoints_match_min", "kills_team_match_sum_rank", "kills_match_max", "kills_team_match_var", "kills_and_assists_match_max"
#                     #,"killPlace_over_maxPlace_team_match_max", "distance_over_weapons_team_match_max", "killsPerWalkDistance_team_match_max", "damageDealt_team_to_match_ratio_max", "killPlace_team_to_match_ratio_max", "killPoints_team_to_match_ratio_max", "rankPoints_team_to_match_ratio_max", "walkDistance_team_to_match_ratio_max", "winPoints_team_to_match_ratio_max", "total_distance_team_to_match_ratio_max", "killPlace_over_maxPlace_team_to_match_ratio_max", "distance_over_weapons_team_to_match_ratio_max", "killsPerWalkDistance_team_to_match_ratio_max"
#                     )
# v29
#UNWANTED_FEATURES = ("healsPerWalkDistance_team_match_max", "rankPoints_team_match_median", "heals_team_to_match_ratio_max", "picker_match_median", "sniper_team_to_match_ratio_mean", "longestKill_team_match_min", "kill_to_team_kills_ratio_match_sum", "longestKill_team_match_std", "roadKills_team_match_mean_rank", "healsAndBoostsPerWalkDistance_team_match_min", "sniper_team_match_mean_rank", "boostsPerWalkDistance_team_match_median", "rideDistance_team_to_match_ratio_mean", "winPoints_team_match_max", "headshot_rate_team_to_match_ratio_mean", "winPoints_match_median", "swimDistance_team_match_sum_rank", "killPoints_team_match_median", "vehicleDestroys_team_match_median_rank", "DBNOs_team_to_match_ratio_max", "killPoints_team_match_mean", "multi_killer_match_sum", "weaponsAcquired_team_match_mean", "healsPerWalkDistance_team_match_min", "heals_team_to_match_ratio_median", "kills_team_to_match_ratio_max", "headshot_rate_team_match_max_rank", "rideDistance_team_match_std", "boostsPerWalkDistance_team_match_min", "killPoints_team_match_sum", "rideDistance_team_to_match_ratio_max", "sniper_team_match_sum_rank", "health_items_match_median", "winPoints_team_match_sum", "revives_team_to_match_ratio_sum", "non_leathal_input_team_to_match_ratio_max", "DBNOs_match_max", "kills_and_assists_team_to_match_ratio_max", "headshot_rate_team_match_sum_rank", "killStreaks_team_match_var", "skill_team_match_sum_rank", "assists_team_to_match_ratio_sum", "revives_team_to_match_ratio_median", "headshotKills_match_max", "headshot_rate_team_match_var", "distance_over_weapons_team_match_std", "rideDistance_team_match_sum", "headshotKills_team_match_max_rank", "winPoints_team_match_mean", "rideDistance_team_to_match_ratio_median", "sniper_team_match_var", "sniper_team_match_median_rank", "boosts_team_to_match_ratio_max", "headshotKills_team_to_match_ratio_mean", "rideDistance_team_match_max", "headshot_rate_team_match_min_rank", "teamKills_team_match_sum_rank", "weaponsAcquired_match_median", "assists_team_to_match_ratio_median", "winPoints_team_match_median", "rideDistance_team_to_match_ratio_sum", "skill_match_max", "picker_team_match_sum", "rideDistance_team_match_mean", "kill_to_team_kills_ratio_team_match_min_rank", "swimDistance_team_to_match_ratio_mean", "assists_match_max", "killStreakrate_team_match_mean", "skill_team_match_max_rank", "teamKills_match_sum", "kill_to_team_kills_ratio_match_max", "sniper_team_to_match_ratio_median", "roadKills_team_match_median_rank", "killStreaks_match_max", "rideDistance_team_match_min", "headshotKills_team_match_sum_rank", "kills_and_assists_match_median", "assists_team_match_var", "weaponsAcquired_team_match_sum", "teamKills_team_to_match_ratio_mean", "rideDistance_team_match_var", "killStreakrate_team_match_sum", "revives_team_match_var", "headshotKills_team_to_match_ratio_sum", "headshot_rate_team_to_match_ratio_max", "headshot_rate_team_to_match_ratio_median", "health_items_team_match_mean", "skill_team_to_match_ratio_mean", "rideDistance_team_match_median", "swimDistance_team_to_match_ratio_sum", "revives_match_max", "health_items_team_match_sum", "kill_to_team_kills_ratio_team_match_sum_rank", "heals_team_match_mean", "rankPoints_team_match_std", "skill_team_to_match_ratio_sum", "headshotKills_team_match_var", "killPlace_team_match_std", "swimDistance_team_match_var", "vehicleDestroys_team_match_min_rank", "swimDistance_team_to_match_ratio_median", "swimDistance_team_to_match_ratio_max", "roadKills_team_match_min_rank", "boosts_match_median", "heals_team_match_sum", "picker_team_match_std", "non_leathal_input_team_match_mean", "heals_match_median", "picker_team_match_max", "boosts_team_match_mean", "swimDistance_team_match_max_rank", "DBNOs_match_median", "headshotKills_team_to_match_ratio_median", "weaponsAcquired_team_match_std", "skill_team_to_match_ratio_median", "killPoints_team_match_std", "DBNOs_team_match_mean", "picker_team_match_median", "sniper_team_match_mean", "sniper_team_match_sum", "swimDistance_team_match_max", "winPoints_team_match_std", "sniper_team_match_min_rank", "headshot_rate_team_match_mean", "swimDistance_team_match_mean", "picker_team_match_min", "health_items_team_match_std", "vehicleDestroys_team_match_sum_rank", "killStreaks_team_to_match_ratio_max", "weaponsAcquired_team_match_median", "multi_killer_team_match_min_rank", "kills_match_median", "swimDistance_team_match_std", "assists_team_to_match_ratio_max", "swimDistance_team_match_sum", "teamKills_team_match_max_rank", "health_items_team_match_max", "healsAndBoostsPerWalkDistance_team_match_std", "weaponsAcquired_team_match_max", "kill_to_team_kills_ratio_team_to_match_ratio_mean", "revives_team_match_mean", "killStreakrate_team_match_median", "non_leathal_input_match_median", "killsPerWalkDistance_team_match_std", "swimDistance_team_match_min", "multi_killer_team_match_sum_rank", "assists_team_match_mean", "heals_team_match_median", "teamKills_match_max", "distance_over_weapons_team_to_match_ratio_min", "headshot_rate_team_match_sum", "multi_killer_match_max", "swimDistance_team_match_median", "health_items_team_match_median", "boosts_team_match_sum", "boosts_team_match_max", "skill_team_match_var", "healsPerWalkDistance_team_match_std", "boostsPerWalkDistance_team_match_std", "kills_team_match_mean", "heals_team_match_max", "killStreakrate_team_match_std", "weaponsAcquired_team_match_min", "roadKills_team_match_sum_rank", "headshotKills_team_to_match_ratio_max", "revives_team_to_match_ratio_max", "heals_team_match_std", "vehicleDestroys_team_to_match_ratio_mean", "non_leathal_input_team_match_std", "health_items_team_match_min", "non_leathal_input_team_match_sum", "kills_without_moving_match_mean", "boosts_team_match_std", "kills_team_match_std", "killStreaks_team_match_mean", "kills_and_assists_team_match_std", "walkDistance_team_to_match_ratio_sum", "teamKills_team_to_match_ratio_median", "DBNOs_team_match_sum", "picker_team_to_match_ratio_min", "DBNOs_team_match_std", "teamKills_team_to_match_ratio_sum", "teamKills_team_match_var", "headshot_rate_team_match_max", "kills_and_assists_team_match_median", "multi_killer_team_to_match_ratio_mean", "distance_over_weapons_match_min", "sniper_team_match_median", "weaponsAcquired_team_to_match_ratio_min", "kills_and_assists_team_match_sum", "kills_team_match_max", "kills_without_moving_team_match_median_rank", "vehicleDestroys_match_sum", "roadKills_team_to_match_ratio_mean", "boosts_team_match_median", "headshot_rate_team_match_median", "kills_team_match_min", "heals_team_match_min", "kills_without_moving_team_match_min_rank", "headshot_rate_team_match_std", "skill_team_to_match_ratio_max", "headshotKills_team_match_mean", "kills_team_match_median", "killStreaks_match_median", "sniper_team_match_std", "non_leathal_input_team_match_max", "kills_team_match_sum", "assists_team_match_sum", "DBNOs_team_match_max", "sniper_team_match_max", "assists_team_match_std", "non_leathal_input_team_match_median", "killStreaks_team_match_std", "kills_and_assists_team_match_max", "teamKills_team_match_mean", "kill_to_team_kills_ratio_team_match_var", "total_distance_team_to_match_ratio_sum", "kill_to_team_kills_ratio_team_match_max_rank", "vehicleDestroys_team_match_max_rank", "roadKills_match_sum", "DBNOs_team_match_median", "killPlace_match_max", "sniper_team_to_match_ratio_max", "kill_to_team_kills_ratio_team_to_match_ratio_sum", "revives_team_match_std", "boosts_team_match_min", "kill_to_team_kills_ratio_team_match_mean", "killStreaks_team_match_sum", "revives_team_match_sum", "roadKills_match_max", "killStreakrate_team_match_max", "kill_to_team_kills_ratio_team_to_match_ratio_median", "non_leathal_input_team_match_min", "skill_team_match_mean", "kills_without_moving_team_match_sum_rank", "walkDistance_match_sum", "kills_and_assists_team_match_min", "headshotKills_team_match_sum", "multi_killer_team_match_max_rank", "sniper_team_match_min", "vehicleDestroys_match_max", "multi_killer_team_match_var", "DBNOs_team_match_min", "headshot_rate_team_match_min", "kills_without_moving_team_match_max_rank", "roadKills_team_match_max_rank", "assists_team_match_max", "vehicleDestroys_team_to_match_ratio_median", "revives_team_match_max", "skill_team_match_sum", "killStreaks_team_match_max", "headshotKills_team_match_max", "killStreakrate_team_to_match_ratio_max", "sniper_match_max", "revives_team_match_median", "skill_team_match_std", "kill_to_team_kills_ratio_team_match_sum", "vehicleDestroys_team_match_var", "headshotKills_team_match_std", "multi_killer_team_to_match_ratio_sum", "assists_team_match_median", "multi_killer_team_to_match_ratio_median", "kill_to_team_kills_ratio_team_to_match_ratio_max", "damageDealt_team_to_match_ratio_min", "headshotKills_team_match_median", "longestKill_team_to_match_ratio_min", "multi_killer_team_match_mean", "assists_team_match_min", "teamKills_team_to_match_ratio_max", "killStreaks_team_match_median", "skill_team_match_max", "roadKills_team_to_match_ratio_median", "non_leathal_input_team_to_match_ratio_min", "vehicleDestroys_team_to_match_ratio_sum", "teamKills_team_match_std", "roadKills_team_match_var", "multi_killer_team_to_match_ratio_max", "killStreakrate_team_to_match_ratio_min", "multi_killer_team_match_sum", "roadKills_team_to_match_ratio_sum", "kill_to_team_kills_ratio_team_match_std", "roadKills_team_match_mean", "vehicleDestroys_team_match_mean", "teamKills_team_match_sum", "kills_and_assists_team_to_match_ratio_min", "sniper_match_median", "kill_to_team_kills_ratio_team_match_max", "weaponsAcquired_match_min", "total_distance_match_sum", "kills_without_moving_match_sum", "killsPerWalkDistance_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_match_median", "killStreaks_team_match_min", "revives_team_match_min", "kills_without_moving_team_match_var", "skill_team_match_median", "picker_match_min", "kills_team_to_match_ratio_min", "headshotKills_team_match_min", "assists_match_median", "killsPerWalkDistance_match_min", "multi_killer_team_match_median", "multi_killer_team_match_std", "skill_team_match_min", "longestKill_match_min", "matchDuration", "headshotKills_match_median", "teamKills_team_match_max", "roadKills_team_match_std", "vehicleDestroys_team_match_sum", "killStreaks_team_to_match_ratio_min", "kills_without_moving_team_match_mean", "vehicleDestroys_team_match_max", "damageDealt_match_min", "roadKills_team_to_match_ratio_max", "killStreakrate_match_min", "assists_team_to_match_ratio_min", "teamKills_team_match_median", "vehicleDestroys_team_match_std", "multi_killer_team_match_max", "skill_match_median", "roadKills_team_match_sum", "kills_without_moving_team_match_sum", "headshot_rate_match_median", "roadKills_team_match_max", "DBNOs_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_match_min", "teamKills_team_match_min", "kills_without_moving_team_match_std", "killStreakrate_match_max", "healsAndBoostsPerWalkDistance_match_min", "kills_match_min", "health_items_team_to_match_ratio_min", "healsAndBoostsPerWalkDistance_team_to_match_ratio_min", "revives_match_median", "rideDistance_match_min", "rideDistance_team_to_match_ratio_min", "multi_killer_team_match_min", "vehicleDestroys_team_to_match_ratio_max", "vehicleDestroys_team_match_median", "health_items_match_min", "killStreaks_match_min", "non_leathal_input_match_min", "roadKills_team_match_median", "swimDistance_match_median", "healsPerWalkDistance_team_to_match_ratio_min", "headshot_rate_match_min", "kills_without_moving_team_match_median", "roadKills_match_median", "teamKills_match_median", "vehicleDestroys_match_median", "kills_without_moving_match_median", "kill_to_team_kills_ratio_match_median", "multi_killer_match_median", "roadKills_team_match_min", "vehicleDestroys_team_match_min", "kills_without_moving_team_match_min", "assists_match_min", "boosts_match_min", "DBNOs_match_min", "headshotKills_match_min", "heals_match_min", "killPlace_match_min", "revives_match_min", "roadKills_match_min", "swimDistance_match_min", "teamKills_match_min", "vehicleDestroys_match_min", "kills_and_assists_match_min", "kills_without_moving_match_min", "boostsPerWalkDistance_match_min", "healsPerWalkDistance_match_min", "skill_match_min", "kill_to_team_kills_ratio_match_min", "multi_killer_match_min", "sniper_match_min", "boosts_team_to_match_ratio_min", "headshotKills_team_to_match_ratio_min", "heals_team_to_match_ratio_min", "revives_team_to_match_ratio_min", "roadKills_team_to_match_ratio_min", "swimDistance_team_to_match_ratio_min", "teamKills_team_to_match_ratio_min", "vehicleDestroys_team_to_match_ratio_min", "boostsPerWalkDistance_team_to_match_ratio_min", "headshot_rate_team_to_match_ratio_min", "skill_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_to_match_ratio_min", "multi_killer_team_to_match_ratio_min", "sniper_team_to_match_ratio_min", "kills_without_moving_team_match_max", "kills_without_moving_match_max", "kill_to_team_kills_ratio_team_match_median_rank", "healsAndBoostsPerWalkDistance_team_to_match_ratio_max", "kills_and_assists_team_to_match_ratio_mean", "longestKill_team_match_max", "health_items_team_match_var", "killPoints_match_median", "healsAndBoostsPerWalkDistance_team_match_min_rank", "DBNOs_team_match_var", "headshot_rate_team_match_median_rank", "boostsPerWalkDistance_team_match_min_rank", "healsPerWalkDistance_team_match_median_rank", "killPoints_team_match_max", "assists_team_match_max_rank", "headshot_rate_team_to_match_ratio_sum", "walkDistance_match_min", "DBNOs_team_to_match_ratio_median", "rideDistance_team_match_max_rank", "non_leathal_input_match_max", "health_items_team_match_max_rank", "winPoints_match_mean", "heals_team_match_sum_rank", "total_distance_match_min", "revives_team_to_match_ratio_mean", "healsAndBoostsPerWalkDistance_team_to_match_ratio_sum", "heals_team_match_max_rank", "damageDealt_team_match_min", "healsPerWalkDistance_team_match_var", "assists_team_to_match_ratio_mean", "healsPerWalkDistance_team_to_match_ratio_max", "boostsPerWalkDistance_team_to_match_ratio_mean", "teamKills_team_match_min_rank", "boostsPerWalkDistance_team_match_var", "boostsPerWalkDistance_team_match_sum", "killStreakrate_team_match_var", "winPoints_team_match_min", "killPoints_team_match_min", "healsAndBoostsPerWalkDistance_team_match_sum_rank", "health_items_team_to_match_ratio_sum", "rideDistance_match_median", "revives_team_match_max_rank", "boostsPerWalkDistance_team_match_max", "killPoints_match_min", "boostsPerWalkDistance_team_match_mean", "boostsPerWalkDistance_team_match_sum_rank", "killStreaks_team_to_match_ratio_median", "killStreaks_team_to_match_ratio_sum", "healsAndBoostsPerWalkDistance_team_match_max", "healsPerWalkDistance_team_to_match_ratio_sum", "killsPerWalkDistance_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_max_rank", "heals_team_to_match_ratio_mean", "heals_team_match_var", "kills_and_assists_team_match_var", "boostsPerWalkDistance_team_match_max_rank", "non_leathal_input_team_to_match_ratio_median", "boosts_team_match_var", "healsPerWalkDistance_team_match_sum", "kills_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_mean", "heals_team_to_match_ratio_sum", "healsAndBoostsPerWalkDistance_team_to_match_ratio_mean", "healsAndBoostsPerWalkDistance_team_match_sum", "multi_killer_team_match_median_rank", "healsPerWalkDistance_team_to_match_ratio_mean", "healsPerWalkDistance_team_match_sum_rank", "healsPerWalkDistance_team_match_mean", "healsPerWalkDistance_team_match_max_rank", "health_items_team_to_match_ratio_max", "boostsPerWalkDistance_team_to_match_ratio_median", "longestKill_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_to_match_ratio_median", "kills_and_assists_team_to_match_ratio_median", "longestKill_team_match_median", "boosts_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_median", "killsPerWalkDistance_team_match_median", "health_items_team_to_match_ratio_median", "healsPerWalkDistance_team_to_match_ratio_median", "healsPerWalkDistance_team_match_median", "killPlace_over_maxPlace_team_match_sum_rank", "kills_and_assists_team_match_mean", "killPoints_team_to_match_ratio_mean", "rankPoints_team_match_sum", "health_items_team_match_sum_rank", "damageDealt_team_match_std", "rankPoints_match_median", "killPoints_match_sum", "weaponsAcquired_match_max", "vehicleDestroys_team_match_mean_rank", "damageDealt_team_to_match_ratio_median", "boostsPerWalkDistance_team_to_match_ratio_sum", "boostsPerWalkDistance_team_match_median_rank", "damageDealt_team_match_median", "assists_team_match_sum_rank", "longestKill_team_to_match_ratio_max", "headshot_rate_team_match_mean_rank", "winPoints_team_match_median_rank", "non_leathal_input_match_sum", "killPoints_team_match_sum_rank", "killStreakrate_team_match_median_rank", "winPoints_team_match_sum_rank", "killPoints_team_match_median_rank", "winPoints_team_to_match_ratio_median", "DBNOs_team_to_match_ratio_sum", "killStreaks_team_match_sum_rank", "assists_match_sum", "killPoints_team_match_mean_rank", "healsPerWalkDistance_team_match_min_rank", "boostsPerWalkDistance_team_to_match_ratio_max", "revives_team_match_sum_rank", "roadKills_match_mean", "DBNOs_match_sum", "killPoints_match_mean", "winPoints_team_match_mean_rank", "headshotKills_match_sum", "longestKill_team_match_mean", "killStreaks_team_match_median_rank", "skill_match_sum", "killsPerWalkDistance_team_match_mean", "multi_killer_team_match_mean_rank", "boosts_match_max", "longestKill_team_to_match_ratio_mean", "rankPoints_team_match_max", "longestKill_team_match_sum_rank", "longestKill_team_match_max_rank", "killsPerWalkDistance_team_match_sum", "killsPerWalkDistance_team_match_var", "non_leathal_input_team_match_var", "sniper_team_match_max_rank", "rankPoints_team_match_min", "non_leathal_input_team_to_match_ratio_sum", "revives_match_sum", "sniper_team_to_match_ratio_sum", "kills_and_assists_team_to_match_ratio_sum", "rankPoints_team_match_mean", "killPlace_over_maxPlace_team_match_std", "group_size", "killPlace_over_maxPlace_team_to_match_ratio_sum", "killPlace_over_maxPlace_team_match_min_rank", "killPlace_over_maxPlace_team_match_mean_rank", "kills_without_moving_team_match_mean_rank", "killPlace_team_to_match_ratio_min", "killPlace_match_sum", "killPlace_over_maxPlace_match_min", "heals_match_max", "distance_over_weapons_team_match_sum", "damageDealt_team_match_mean", "boosts_team_match_mean_rank", "revives_team_match_min_rank", "damageDealt_team_to_match_ratio_mean", "weaponsAcquired_team_match_mean_rank", "damageDealt_team_to_match_ratio_sum", "killPoints_team_match_min_rank", "picker_team_to_match_ratio_median", "killPoints_team_to_match_ratio_sum", "total_distance_team_match_min_rank", "non_leathal_input_team_match_max_rank", "kills_match_sum", "rankPoints_team_match_median_rank", "kills_and_assists_team_match_mean_rank", "kill_to_team_kills_ratio_team_match_mean_rank", "distance_over_weapons_team_match_median_rank", "headshotKills_team_match_min_rank", "rankPoints_team_match_mean_rank", "picker_team_match_median_rank", "DBNOs_team_match_max_rank", "damageDealt_team_match_max", "distance_over_weapons_team_match_mean", "distance_over_weapons_team_match_median", "healsAndBoostsPerWalkDistance_team_match_median_rank", "total_distance_team_match_sum", "winPoints_team_to_match_ratio_sum", "non_leathal_input_team_match_mean_rank", "DBNOs_team_match_sum_rank", "non_leathal_input_team_match_sum_rank", "distance_over_weapons_team_match_mean_rank", "total_distance_team_match_max", "kills_and_assists_team_match_sum_rank", "DBNOs_team_match_mean_rank", "picker_team_to_match_ratio_max", "total_distance_team_match_std", "walkDistance_team_to_match_ratio_min", "total_distance_team_match_min", "walkDistance_team_match_std", "teamKills_team_match_median_rank", "non_leathal_input_team_to_match_ratio_mean", "damageDealt_team_match_sum", "killStreakrate_team_to_match_ratio_median", "killStreaks_team_match_mean_rank", "rideDistance_team_match_sum_rank", "killPoints_match_max", "rankPoints_match_mean", "total_distance_team_match_mean", "weaponsAcquired_team_to_match_ratio_max", "winPoints_match_sum", "kills_and_assists_team_match_max_rank", "killPoints_team_match_var", "killPlace_team_match_sum", "killStreakrate_match_median", "rideDistance_match_sum", "winPoints_match_max", "boosts_team_to_match_ratio_sum", "boosts_team_to_match_ratio_mean", "total_distance_team_to_match_ratio_min", "headshot_rate_match_max", "skill_team_match_min_rank", "winPoints_team_match_var", "picker_match_max", "winPoints_match_min", "kills_team_match_sum_rank", "kills_match_max", "kills_team_match_var", "kills_and_assists_match_max", "killPlace_over_maxPlace_team_to_match_ratio_mean", "killStreakrate_team_match_min_rank", "killsPerWalkDistance_team_match_min_rank", "killPlace_over_maxPlace_team_match_median_rank", "killPlaceRankWithinKills_team_to_match_ratio_max", "killsPerWalkDistance_team_match_min", "killStreakrate_team_match_min", "killPlaceRankWithinKills_match_max")

# v36 minus some least usable features
# UNWANTED_FEATURES = ("healsPerWalkDistance_team_match_max", "rankPoints_team_match_median", "heals_team_to_match_ratio_max", "picker_match_median", "sniper_team_to_match_ratio_mean", "longestKill_team_match_min", "kill_to_team_kills_ratio_match_sum", "longestKill_team_match_std", "roadKills_team_match_mean_rank", "healsAndBoostsPerWalkDistance_team_match_min", "sniper_team_match_mean_rank", "boostsPerWalkDistance_team_match_median", "rideDistance_team_to_match_ratio_mean", "winPoints_team_match_max", "headshot_rate_team_to_match_ratio_mean", "winPoints_match_median", "swimDistance_team_match_sum_rank", "killPoints_team_match_median", "vehicleDestroys_team_match_median_rank", "DBNOs_team_to_match_ratio_max", "killPoints_team_match_mean", "multi_killer_match_sum", "weaponsAcquired_team_match_mean", "healsPerWalkDistance_team_match_min", "heals_team_to_match_ratio_median", "kills_team_to_match_ratio_max", "headshot_rate_team_match_max_rank", "rideDistance_team_match_std", "boostsPerWalkDistance_team_match_min", "killPoints_team_match_sum", "rideDistance_team_to_match_ratio_max", "sniper_team_match_sum_rank", "health_items_match_median", "winPoints_team_match_sum", "revives_team_to_match_ratio_sum", "non_leathal_input_team_to_match_ratio_max", "DBNOs_match_max", "kills_and_assists_team_to_match_ratio_max", "headshot_rate_team_match_sum_rank", "killStreaks_team_match_var", "skill_team_match_sum_rank", "assists_team_to_match_ratio_sum", "revives_team_to_match_ratio_median", "headshotKills_match_max", "headshot_rate_team_match_var", "distance_over_weapons_team_match_std", "rideDistance_team_match_sum", "headshotKills_team_match_max_rank", "winPoints_team_match_mean", "rideDistance_team_to_match_ratio_median", "sniper_team_match_var", "sniper_team_match_median_rank", "boosts_team_to_match_ratio_max", "headshotKills_team_to_match_ratio_mean", "rideDistance_team_match_max", "headshot_rate_team_match_min_rank", "teamKills_team_match_sum_rank", "weaponsAcquired_match_median", "assists_team_to_match_ratio_median", "winPoints_team_match_median", "rideDistance_team_to_match_ratio_sum", "skill_match_max", "picker_team_match_sum", "rideDistance_team_match_mean", "kill_to_team_kills_ratio_team_match_min_rank", "swimDistance_team_to_match_ratio_mean", "assists_match_max", "killStreakrate_team_match_mean", "skill_team_match_max_rank", "teamKills_match_sum", "kill_to_team_kills_ratio_match_max", "sniper_team_to_match_ratio_median", "roadKills_team_match_median_rank", "killStreaks_match_max", "rideDistance_team_match_min", "headshotKills_team_match_sum_rank", "kills_and_assists_match_median", "assists_team_match_var", "weaponsAcquired_team_match_sum", "teamKills_team_to_match_ratio_mean", "rideDistance_team_match_var", "killStreakrate_team_match_sum", "revives_team_match_var", "headshotKills_team_to_match_ratio_sum", "headshot_rate_team_to_match_ratio_max", "headshot_rate_team_to_match_ratio_median", "health_items_team_match_mean", "skill_team_to_match_ratio_mean", "rideDistance_team_match_median", "swimDistance_team_to_match_ratio_sum", "revives_match_max", "health_items_team_match_sum", "kill_to_team_kills_ratio_team_match_sum_rank", "heals_team_match_mean", "rankPoints_team_match_std", "skill_team_to_match_ratio_sum", "headshotKills_team_match_var", "killPlace_team_match_std", "swimDistance_team_match_var", "vehicleDestroys_team_match_min_rank", "swimDistance_team_to_match_ratio_median", "swimDistance_team_to_match_ratio_max", "roadKills_team_match_min_rank", "boosts_match_median", "heals_team_match_sum", "picker_team_match_std", "non_leathal_input_team_match_mean", "heals_match_median", "picker_team_match_max", "boosts_team_match_mean", "swimDistance_team_match_max_rank", "DBNOs_match_median", "headshotKills_team_to_match_ratio_median", "weaponsAcquired_team_match_std", "skill_team_to_match_ratio_median", "killPoints_team_match_std", "DBNOs_team_match_mean", "picker_team_match_median", "sniper_team_match_mean", "sniper_team_match_sum", "swimDistance_team_match_max", "winPoints_team_match_std", "sniper_team_match_min_rank", "headshot_rate_team_match_mean", "swimDistance_team_match_mean", "picker_team_match_min", "health_items_team_match_std", "vehicleDestroys_team_match_sum_rank", "killStreaks_team_to_match_ratio_max", "weaponsAcquired_team_match_median", "multi_killer_team_match_min_rank", "kills_match_median", "swimDistance_team_match_std", "assists_team_to_match_ratio_max", "swimDistance_team_match_sum", "teamKills_team_match_max_rank", "health_items_team_match_max", "healsAndBoostsPerWalkDistance_team_match_std", "weaponsAcquired_team_match_max", "kill_to_team_kills_ratio_team_to_match_ratio_mean", "revives_team_match_mean", "killStreakrate_team_match_median", "non_leathal_input_match_median", "killsPerWalkDistance_team_match_std", "swimDistance_team_match_min", "multi_killer_team_match_sum_rank", "assists_team_match_mean", "heals_team_match_median", "teamKills_match_max", "distance_over_weapons_team_to_match_ratio_min", "headshot_rate_team_match_sum", "multi_killer_match_max", "swimDistance_team_match_median", "health_items_team_match_median", "boosts_team_match_sum", "boosts_team_match_max", "skill_team_match_var", "healsPerWalkDistance_team_match_std", "boostsPerWalkDistance_team_match_std", "kills_team_match_mean", "heals_team_match_max", "killStreakrate_team_match_std", "weaponsAcquired_team_match_min", "roadKills_team_match_sum_rank", "headshotKills_team_to_match_ratio_max", "revives_team_to_match_ratio_max", "heals_team_match_std", "vehicleDestroys_team_to_match_ratio_mean", "non_leathal_input_team_match_std", "health_items_team_match_min", "non_leathal_input_team_match_sum", "kills_without_moving_match_mean", "boosts_team_match_std", "kills_team_match_std", "killStreaks_team_match_mean", "kills_and_assists_team_match_std", "walkDistance_team_to_match_ratio_sum", "teamKills_team_to_match_ratio_median", "DBNOs_team_match_sum", "picker_team_to_match_ratio_min", "DBNOs_team_match_std", "teamKills_team_to_match_ratio_sum", "teamKills_team_match_var", "headshot_rate_team_match_max", "kills_and_assists_team_match_median", "multi_killer_team_to_match_ratio_mean", "distance_over_weapons_match_min", "sniper_team_match_median", "weaponsAcquired_team_to_match_ratio_min", "kills_and_assists_team_match_sum", "kills_team_match_max", "kills_without_moving_team_match_median_rank", "vehicleDestroys_match_sum", "roadKills_team_to_match_ratio_mean", "boosts_team_match_median", "headshot_rate_team_match_median", "kills_team_match_min", "heals_team_match_min", "kills_without_moving_team_match_min_rank", "headshot_rate_team_match_std", "skill_team_to_match_ratio_max", "headshotKills_team_match_mean", "kills_team_match_median", "killStreaks_match_median", "sniper_team_match_std", "non_leathal_input_team_match_max", "kills_team_match_sum", "assists_team_match_sum", "DBNOs_team_match_max", "sniper_team_match_max", "assists_team_match_std", "non_leathal_input_team_match_median", "killStreaks_team_match_std", "kills_and_assists_team_match_max", "teamKills_team_match_mean", "kill_to_team_kills_ratio_team_match_var", "total_distance_team_to_match_ratio_sum", "kill_to_team_kills_ratio_team_match_max_rank", "vehicleDestroys_team_match_max_rank", "roadKills_match_sum", "DBNOs_team_match_median", "killPlace_match_max", "sniper_team_to_match_ratio_max", "kill_to_team_kills_ratio_team_to_match_ratio_sum", "revives_team_match_std", "boosts_team_match_min", "kill_to_team_kills_ratio_team_match_mean", "killStreaks_team_match_sum", "revives_team_match_sum", "roadKills_match_max", "killStreakrate_team_match_max", "kill_to_team_kills_ratio_team_to_match_ratio_median", "non_leathal_input_team_match_min", "skill_team_match_mean", "kills_without_moving_team_match_sum_rank", "walkDistance_match_sum", "kills_and_assists_team_match_min", "headshotKills_team_match_sum", "multi_killer_team_match_max_rank", "sniper_team_match_min", "vehicleDestroys_match_max", "multi_killer_team_match_var", "DBNOs_team_match_min", "headshot_rate_team_match_min", "kills_without_moving_team_match_max_rank", "roadKills_team_match_max_rank", "assists_team_match_max", "vehicleDestroys_team_to_match_ratio_median", "revives_team_match_max", "skill_team_match_sum", "killStreaks_team_match_max", "headshotKills_team_match_max", "killStreakrate_team_to_match_ratio_max", "sniper_match_max", "revives_team_match_median", "skill_team_match_std", "kill_to_team_kills_ratio_team_match_sum", "vehicleDestroys_team_match_var", "headshotKills_team_match_std", "multi_killer_team_to_match_ratio_sum", "assists_team_match_median", "multi_killer_team_to_match_ratio_median", "kill_to_team_kills_ratio_team_to_match_ratio_max", "damageDealt_team_to_match_ratio_min", "headshotKills_team_match_median", "longestKill_team_to_match_ratio_min", "multi_killer_team_match_mean", "assists_team_match_min", "teamKills_team_to_match_ratio_max", "killStreaks_team_match_median", "skill_team_match_max", "roadKills_team_to_match_ratio_median", "non_leathal_input_team_to_match_ratio_min", "vehicleDestroys_team_to_match_ratio_sum", "teamKills_team_match_std", "roadKills_team_match_var", "multi_killer_team_to_match_ratio_max", "killStreakrate_team_to_match_ratio_min", "multi_killer_team_match_sum", "roadKills_team_to_match_ratio_sum", "kill_to_team_kills_ratio_team_match_std", "roadKills_team_match_mean", "vehicleDestroys_team_match_mean", "teamKills_team_match_sum", "kills_and_assists_team_to_match_ratio_min", "sniper_match_median", "kill_to_team_kills_ratio_team_match_max", "weaponsAcquired_match_min", "total_distance_match_sum", "kills_without_moving_match_sum", "killsPerWalkDistance_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_match_median", "killStreaks_team_match_min", "revives_team_match_min", "kills_without_moving_team_match_var", "skill_team_match_median", "picker_match_min", "kills_team_to_match_ratio_min", "headshotKills_team_match_min", "assists_match_median", "killsPerWalkDistance_match_min", "multi_killer_team_match_median", "multi_killer_team_match_std", "skill_team_match_min", "longestKill_match_min", "matchDuration", "headshotKills_match_median", "teamKills_team_match_max", "roadKills_team_match_std", "vehicleDestroys_team_match_sum", "killStreaks_team_to_match_ratio_min", "kills_without_moving_team_match_mean", "vehicleDestroys_team_match_max", "damageDealt_match_min", "roadKills_team_to_match_ratio_max", "killStreakrate_match_min", "assists_team_to_match_ratio_min", "teamKills_team_match_median", "vehicleDestroys_team_match_std", "multi_killer_team_match_max", "skill_match_median", "roadKills_team_match_sum", "kills_without_moving_team_match_sum", "headshot_rate_match_median", "roadKills_team_match_max", "DBNOs_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_match_min", "teamKills_team_match_min", "kills_without_moving_team_match_std", "killStreakrate_match_max", "healsAndBoostsPerWalkDistance_match_min", "kills_match_min", "health_items_team_to_match_ratio_min", "healsAndBoostsPerWalkDistance_team_to_match_ratio_min", "revives_match_median", "rideDistance_match_min", "rideDistance_team_to_match_ratio_min", "multi_killer_team_match_min", "vehicleDestroys_team_to_match_ratio_max", "vehicleDestroys_team_match_median", "health_items_match_min", "killStreaks_match_min", "non_leathal_input_match_min", "roadKills_team_match_median", "swimDistance_match_median", "healsPerWalkDistance_team_to_match_ratio_min", "headshot_rate_match_min", "kills_without_moving_team_match_median", "roadKills_match_median", "teamKills_match_median", "vehicleDestroys_match_median", "kills_without_moving_match_median", "kill_to_team_kills_ratio_match_median", "multi_killer_match_median", "roadKills_team_match_min", "vehicleDestroys_team_match_min", "kills_without_moving_team_match_min", "assists_match_min", "boosts_match_min", "DBNOs_match_min", "headshotKills_match_min", "heals_match_min", "killPlace_match_min", "revives_match_min", "roadKills_match_min", "swimDistance_match_min", "teamKills_match_min", "vehicleDestroys_match_min", "kills_and_assists_match_min", "kills_without_moving_match_min", "boostsPerWalkDistance_match_min", "healsPerWalkDistance_match_min", "skill_match_min", "kill_to_team_kills_ratio_match_min", "multi_killer_match_min", "sniper_match_min", "boosts_team_to_match_ratio_min", "headshotKills_team_to_match_ratio_min", "heals_team_to_match_ratio_min", "revives_team_to_match_ratio_min", "roadKills_team_to_match_ratio_min", "swimDistance_team_to_match_ratio_min", "teamKills_team_to_match_ratio_min", "vehicleDestroys_team_to_match_ratio_min", "boostsPerWalkDistance_team_to_match_ratio_min", "headshot_rate_team_to_match_ratio_min", "skill_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_to_match_ratio_min", "multi_killer_team_to_match_ratio_min", "sniper_team_to_match_ratio_min", "kills_without_moving_team_match_max", "kills_without_moving_match_max", "kill_to_team_kills_ratio_team_match_median_rank", "healsAndBoostsPerWalkDistance_team_to_match_ratio_max", "kills_and_assists_team_to_match_ratio_mean", "longestKill_team_match_max", "health_items_team_match_var", "killPoints_match_median", "healsAndBoostsPerWalkDistance_team_match_min_rank", "DBNOs_team_match_var", "headshot_rate_team_match_median_rank", "boostsPerWalkDistance_team_match_min_rank", "healsPerWalkDistance_team_match_median_rank", "killPoints_team_match_max", "assists_team_match_max_rank", "headshot_rate_team_to_match_ratio_sum", "walkDistance_match_min", "DBNOs_team_to_match_ratio_median", "rideDistance_team_match_max_rank", "non_leathal_input_match_max", "health_items_team_match_max_rank", "winPoints_match_mean", "heals_team_match_sum_rank", "total_distance_match_min", "revives_team_to_match_ratio_mean", "healsAndBoostsPerWalkDistance_team_to_match_ratio_sum", "heals_team_match_max_rank", "damageDealt_team_match_min", "healsPerWalkDistance_team_match_var", "assists_team_to_match_ratio_mean", "healsPerWalkDistance_team_to_match_ratio_max", "boostsPerWalkDistance_team_to_match_ratio_mean", "teamKills_team_match_min_rank", "boostsPerWalkDistance_team_match_var", "boostsPerWalkDistance_team_match_sum", "killStreakrate_team_match_var", "winPoints_team_match_min", "killPoints_team_match_min", "healsAndBoostsPerWalkDistance_team_match_sum_rank", "health_items_team_to_match_ratio_sum", "rideDistance_match_median", "revives_team_match_max_rank", "boostsPerWalkDistance_team_match_max", "killPoints_match_min", "boostsPerWalkDistance_team_match_mean", "boostsPerWalkDistance_team_match_sum_rank", "killStreaks_team_to_match_ratio_median", "killStreaks_team_to_match_ratio_sum", "healsAndBoostsPerWalkDistance_team_match_max", "healsPerWalkDistance_team_to_match_ratio_sum", "killsPerWalkDistance_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_max_rank", "heals_team_to_match_ratio_mean", "heals_team_match_var", "kills_and_assists_team_match_var", "boostsPerWalkDistance_team_match_max_rank", "non_leathal_input_team_to_match_ratio_median", "boosts_team_match_var", "healsPerWalkDistance_team_match_sum", "kills_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_mean", "heals_team_to_match_ratio_sum", "healsAndBoostsPerWalkDistance_team_to_match_ratio_mean", "healsAndBoostsPerWalkDistance_team_match_sum", "multi_killer_team_match_median_rank", "healsPerWalkDistance_team_to_match_ratio_mean", "healsPerWalkDistance_team_match_sum_rank", "healsPerWalkDistance_team_match_mean", "healsPerWalkDistance_team_match_max_rank", "health_items_team_to_match_ratio_max", "boostsPerWalkDistance_team_to_match_ratio_median", "longestKill_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_to_match_ratio_median", "kills_and_assists_team_to_match_ratio_median", "longestKill_team_match_median", "boosts_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_median", "killsPerWalkDistance_team_match_median", "health_items_team_to_match_ratio_median", "healsPerWalkDistance_team_to_match_ratio_median", "healsPerWalkDistance_team_match_median", "killPlace_over_maxPlace_team_match_sum_rank", "kills_and_assists_team_match_mean", "killPoints_team_to_match_ratio_mean", "rankPoints_team_match_sum", "health_items_team_match_sum_rank", "damageDealt_team_match_std", "rankPoints_match_median", "killPoints_match_sum", "weaponsAcquired_match_max", "vehicleDestroys_team_match_mean_rank", "damageDealt_team_to_match_ratio_median", "boostsPerWalkDistance_team_to_match_ratio_sum", "boostsPerWalkDistance_team_match_median_rank", "damageDealt_team_match_median", "assists_team_match_sum_rank", "longestKill_team_to_match_ratio_max", "headshot_rate_team_match_mean_rank", "winPoints_team_match_median_rank", "non_leathal_input_match_sum", "killPoints_team_match_sum_rank", "killStreakrate_team_match_median_rank", "winPoints_team_match_sum_rank", "killPoints_team_match_median_rank", "winPoints_team_to_match_ratio_median", "DBNOs_team_to_match_ratio_sum", "killStreaks_team_match_sum_rank", "assists_match_sum", "killPoints_team_match_mean_rank", "healsPerWalkDistance_team_match_min_rank", "boostsPerWalkDistance_team_to_match_ratio_max", "revives_team_match_sum_rank", "roadKills_match_mean", "DBNOs_match_sum", "killPoints_match_mean", "winPoints_team_match_mean_rank", "headshotKills_match_sum", "longestKill_team_match_mean", "killStreaks_team_match_median_rank", "skill_match_sum", "killsPerWalkDistance_team_match_mean", "multi_killer_team_match_mean_rank", "boosts_match_max", "longestKill_team_to_match_ratio_mean", "rankPoints_team_match_max", "longestKill_team_match_sum_rank", "longestKill_team_match_max_rank", "killsPerWalkDistance_team_match_sum", "killsPerWalkDistance_team_match_var", "non_leathal_input_team_match_var", "sniper_team_match_max_rank", "rankPoints_team_match_min", "non_leathal_input_team_to_match_ratio_sum", "revives_match_sum", "sniper_team_to_match_ratio_sum", "kills_and_assists_team_to_match_ratio_sum", "rankPoints_team_match_mean", "killPlace_over_maxPlace_team_match_std", "group_size", "killPlace_over_maxPlace_team_to_match_ratio_sum", "killPlace_over_maxPlace_team_match_min_rank", "killPlace_over_maxPlace_team_match_mean_rank", "kills_without_moving_team_match_mean_rank", "killPlace_team_to_match_ratio_min", "killPlace_match_sum", "killPlace_over_maxPlace_match_min", "heals_match_max", "distance_over_weapons_team_match_sum", "damageDealt_team_match_mean", "boosts_team_match_mean_rank", "revives_team_match_min_rank", "damageDealt_team_to_match_ratio_mean", "weaponsAcquired_team_match_mean_rank", "damageDealt_team_to_match_ratio_sum", "killPoints_team_match_min_rank", "picker_team_to_match_ratio_median", "killPoints_team_to_match_ratio_sum", "total_distance_team_match_min_rank", "non_leathal_input_team_match_max_rank", "kills_match_sum", "rankPoints_team_match_median_rank", "kills_and_assists_team_match_mean_rank", "kill_to_team_kills_ratio_team_match_mean_rank", "distance_over_weapons_team_match_median_rank", "headshotKills_team_match_min_rank", "rankPoints_team_match_mean_rank", "picker_team_match_median_rank", "DBNOs_team_match_max_rank", "damageDealt_team_match_max", "distance_over_weapons_team_match_mean", "distance_over_weapons_team_match_median", "healsAndBoostsPerWalkDistance_team_match_median_rank", "total_distance_team_match_sum", "winPoints_team_to_match_ratio_sum", "non_leathal_input_team_match_mean_rank", "DBNOs_team_match_sum_rank", "non_leathal_input_team_match_sum_rank", "distance_over_weapons_team_match_mean_rank", "total_distance_team_match_max", "kills_and_assists_team_match_sum_rank", "DBNOs_team_match_mean_rank", "picker_team_to_match_ratio_max", "total_distance_team_match_std", "walkDistance_team_to_match_ratio_min", "total_distance_team_match_min", "walkDistance_team_match_std", "teamKills_team_match_median_rank", "non_leathal_input_team_to_match_ratio_mean", "damageDealt_team_match_sum", "killStreakrate_team_to_match_ratio_median", "killStreaks_team_match_mean_rank", "rideDistance_team_match_sum_rank", "killPoints_match_max", "rankPoints_match_mean", "total_distance_team_match_mean", "weaponsAcquired_team_to_match_ratio_max", "winPoints_match_sum", "kills_and_assists_team_match_max_rank", "killPoints_team_match_var", "killPlace_team_match_sum", "killStreakrate_match_median", "rideDistance_match_sum", "winPoints_match_max", "boosts_team_to_match_ratio_sum", "boosts_team_to_match_ratio_mean", "total_distance_team_to_match_ratio_min", "headshot_rate_match_max", "skill_team_match_min_rank", "winPoints_team_match_var", "picker_match_max", "winPoints_match_min", "kills_team_match_sum_rank", "kills_match_max", "kills_team_match_var", "kills_and_assists_match_max"
#                      #,"killPlace_over_maxPlace_team_match_max", "distance_over_weapons_team_match_max", "killsPerWalkDistance_team_match_max", "damageDealt_team_to_match_ratio_max", "killPlace_team_to_match_ratio_max", "killPoints_team_to_match_ratio_max", "rankPoints_team_to_match_ratio_max", "walkDistance_team_to_match_ratio_max", "winPoints_team_to_match_ratio_max", "total_distance_team_to_match_ratio_max", "killPlace_over_maxPlace_team_to_match_ratio_max", "distance_over_weapons_team_to_match_ratio_max", "killsPerWalkDistance_team_to_match_ratio_max"
#                      #, "walkDistanceRankWithinKills_team_to_match_ratio_max", "walkDistanceRankWithinKills_match_sum", "assistsRankWithinKills_match_max", "killStreakrate_team_match_min", "healsRankWithinKills_match_sum", "assistsRankWithinKills_match_sum", "healsRankWithinKills_match_max", "boostsRankWithinKills_match_max", "boostsRankWithinKills_match_sum", "damageDealtRankWithinKills_match_sum", "damageDealtRankWithinKills_match_max", "walkDistanceRankWithinKills_match_max", "killPlaceRankWithinKills_match_max"
#                      , "damageDealtRankWithinKills_team_match_sum", "health_items_team_to_match_ratio_mean", "walkDistance_team_match_median_rank", "killStreaks_team_to_match_ratio_mean", "killStreaks_match_sum", "killPlaceRankWithinKills_team_to_match_ratio_sum", "walkDistanceRankWithinKills_team_match_median", "damageDealtRankWithinKills_team_to_match_ratio_sum", "walkDistanceRankWithinKills_team_to_match_ratio_median", "total_distance_team_match_median_rank", "killsPerWalkDistance_team_match_max", "total_distance_team_to_match_ratio_median", "walkDistanceRankWithinKills_team_match_max", "killsPerWalkDistance_team_to_match_ratio_mean", "walkDistance_team_to_match_ratio_median", "walkDistanceRankWithinKills_team_match_mean", "walkDistanceRankWithinKills_team_to_match_ratio_mean", "kills_team_match_mean_rank", "killPlace_over_maxPlace_team_match_min", "assistsRankWithinKills_team_match_sum", "killsPerWalkDistance_team_to_match_ratio_sum", "killsPerWalkDistance_team_match_mean_rank", "walkDistance_team_match_mean", "healsRankWithinKills_team_match_sum", "walkDistance_team_match_median", "killPlace_over_maxPlace_team_match_median", "healsRankWithinKills_team_to_match_ratio_sum", "boostsRankWithinKills_team_to_match_ratio_sum", "killPlace_over_maxPlace_team_match_sum", "boostsRankWithinKills_team_match_sum", "walkDistanceRankWithinKills_team_match_sum", "killPlace_over_maxPlace_team_match_mean", "walkDistanceRankWithinKills_team_to_match_ratio_sum", "killsPerWalkDistance_team_match_max_rank", "picker_team_match_mean", "killsPerWalkDistance_team_match_median_rank", "killsPerWalkDistance_team_match_sum_rank", "killPlace_team_to_match_ratio_median", "damageDealtRankWithinKills_team_match_std", "killPlace_team_to_match_ratio_mean", "killPlaceRankWithinKills_match_sum", "kills_team_match_max_rank", "kills_team_to_match_ratio_mean", "killPlace_team_match_max", "killPlaceRankWithinKills_match_min", "playersJoined", "walkDistanceRankWithinKills_team_match_std", "killStreaks_team_match_min_rank", "killPlaceRankWithinKills_team_match_std", "healsRankWithinKills_team_match_std", "boostsRankWithinKills_team_match_std", "healsRankWithinKills_team_to_match_ratio_max", "numGroups", "maxPlace", "boostsRankWithinKills_team_to_match_ratio_max", "assistsRankWithinKills_team_match_std", "killPlace_over_maxPlace_team_to_match_ratio_median", "damageDealtRankWithinKills_team_to_match_ratio_max", "killStreakrate_team_match_min_rank", "killPlace_over_maxPlace_team_to_match_ratio_mean", "damageDealtRankWithinKills_match_min", "killsPerWalkDistance_team_match_min_rank", "killsPerWalkDistance_team_match_min", "killPlace_over_maxPlace_team_match_median_rank", "walkDistanceRankWithinKills_team_to_match_ratio_max", "walkDistanceRankWithinKills_match_sum", "assistsRankWithinKills_match_max", "killStreakrate_team_match_min", "healsRankWithinKills_match_sum", "assistsRankWithinKills_match_sum", "healsRankWithinKills_match_max", "boostsRankWithinKills_match_max", "boostsRankWithinKills_match_sum", "damageDealtRankWithinKills_match_sum", "damageDealtRankWithinKills_match_max", "walkDistanceRankWithinKills_match_max", "killPlaceRankWithinKills_match_max"
#                      )

# v37
#UNWANTED_FEATURES = ("healsPerWalkDistance_team_match_max", "rankPoints_team_match_median", "heals_team_to_match_ratio_max", "picker_match_median", "sniper_team_to_match_ratio_mean", "longestKill_team_match_min", "kill_to_team_kills_ratio_match_sum", "longestKill_team_match_std", "roadKills_team_match_mean_rank", "healsAndBoostsPerWalkDistance_team_match_min", "sniper_team_match_mean_rank", "boostsPerWalkDistance_team_match_median", "rideDistance_team_to_match_ratio_mean", "winPoints_team_match_max", "headshot_rate_team_to_match_ratio_mean", "winPoints_match_median", "swimDistance_team_match_sum_rank", "killPoints_team_match_median", "vehicleDestroys_team_match_median_rank", "DBNOs_team_to_match_ratio_max", "killPoints_team_match_mean", "multi_killer_match_sum", "weaponsAcquired_team_match_mean", "healsPerWalkDistance_team_match_min", "heals_team_to_match_ratio_median", "kills_team_to_match_ratio_max", "headshot_rate_team_match_max_rank", "rideDistance_team_match_std", "boostsPerWalkDistance_team_match_min", "killPoints_team_match_sum", "rideDistance_team_to_match_ratio_max", "sniper_team_match_sum_rank", "health_items_match_median", "winPoints_team_match_sum", "revives_team_to_match_ratio_sum", "non_leathal_input_team_to_match_ratio_max", "DBNOs_match_max", "kills_and_assists_team_to_match_ratio_max", "headshot_rate_team_match_sum_rank", "killStreaks_team_match_var", "skill_team_match_sum_rank", "assists_team_to_match_ratio_sum", "revives_team_to_match_ratio_median", "headshotKills_match_max", "headshot_rate_team_match_var", "distance_over_weapons_team_match_std", "rideDistance_team_match_sum", "headshotKills_team_match_max_rank", "winPoints_team_match_mean", "rideDistance_team_to_match_ratio_median", "sniper_team_match_var", "sniper_team_match_median_rank", "boosts_team_to_match_ratio_max", "headshotKills_team_to_match_ratio_mean", "rideDistance_team_match_max", "headshot_rate_team_match_min_rank", "teamKills_team_match_sum_rank", "weaponsAcquired_match_median", "assists_team_to_match_ratio_median", "winPoints_team_match_median", "rideDistance_team_to_match_ratio_sum", "skill_match_max", "picker_team_match_sum", "rideDistance_team_match_mean", "kill_to_team_kills_ratio_team_match_min_rank", "swimDistance_team_to_match_ratio_mean", "assists_match_max", "killStreakrate_team_match_mean", "skill_team_match_max_rank", "teamKills_match_sum", "kill_to_team_kills_ratio_match_max", "sniper_team_to_match_ratio_median", "roadKills_team_match_median_rank", "killStreaks_match_max", "rideDistance_team_match_min", "headshotKills_team_match_sum_rank", "kills_and_assists_match_median", "assists_team_match_var", "weaponsAcquired_team_match_sum", "teamKills_team_to_match_ratio_mean", "rideDistance_team_match_var", "killStreakrate_team_match_sum", "revives_team_match_var", "headshotKills_team_to_match_ratio_sum", "headshot_rate_team_to_match_ratio_max", "headshot_rate_team_to_match_ratio_median", "health_items_team_match_mean", "skill_team_to_match_ratio_mean", "rideDistance_team_match_median", "swimDistance_team_to_match_ratio_sum", "revives_match_max", "health_items_team_match_sum", "kill_to_team_kills_ratio_team_match_sum_rank", "heals_team_match_mean", "rankPoints_team_match_std", "skill_team_to_match_ratio_sum", "headshotKills_team_match_var", "killPlace_team_match_std", "swimDistance_team_match_var", "vehicleDestroys_team_match_min_rank", "swimDistance_team_to_match_ratio_median", "swimDistance_team_to_match_ratio_max", "roadKills_team_match_min_rank", "boosts_match_median", "heals_team_match_sum", "picker_team_match_std", "non_leathal_input_team_match_mean", "heals_match_median", "picker_team_match_max", "boosts_team_match_mean", "swimDistance_team_match_max_rank", "DBNOs_match_median", "headshotKills_team_to_match_ratio_median", "weaponsAcquired_team_match_std", "skill_team_to_match_ratio_median", "killPoints_team_match_std", "DBNOs_team_match_mean", "picker_team_match_median", "sniper_team_match_mean", "sniper_team_match_sum", "swimDistance_team_match_max", "winPoints_team_match_std", "sniper_team_match_min_rank", "headshot_rate_team_match_mean", "swimDistance_team_match_mean", "picker_team_match_min", "health_items_team_match_std", "vehicleDestroys_team_match_sum_rank", "killStreaks_team_to_match_ratio_max", "weaponsAcquired_team_match_median", "multi_killer_team_match_min_rank", "kills_match_median", "swimDistance_team_match_std", "assists_team_to_match_ratio_max", "swimDistance_team_match_sum", "teamKills_team_match_max_rank", "health_items_team_match_max", "healsAndBoostsPerWalkDistance_team_match_std", "weaponsAcquired_team_match_max", "kill_to_team_kills_ratio_team_to_match_ratio_mean", "revives_team_match_mean", "killStreakrate_team_match_median", "non_leathal_input_match_median", "killsPerWalkDistance_team_match_std", "swimDistance_team_match_min", "multi_killer_team_match_sum_rank", "assists_team_match_mean", "heals_team_match_median", "teamKills_match_max", "distance_over_weapons_team_to_match_ratio_min", "headshot_rate_team_match_sum", "multi_killer_match_max", "swimDistance_team_match_median", "health_items_team_match_median", "boosts_team_match_sum", "boosts_team_match_max", "skill_team_match_var", "healsPerWalkDistance_team_match_std", "boostsPerWalkDistance_team_match_std", "kills_team_match_mean", "heals_team_match_max", "killStreakrate_team_match_std", "weaponsAcquired_team_match_min", "roadKills_team_match_sum_rank", "headshotKills_team_to_match_ratio_max", "revives_team_to_match_ratio_max", "heals_team_match_std", "vehicleDestroys_team_to_match_ratio_mean", "non_leathal_input_team_match_std", "health_items_team_match_min", "non_leathal_input_team_match_sum", "kills_without_moving_match_mean", "boosts_team_match_std", "kills_team_match_std", "killStreaks_team_match_mean", "kills_and_assists_team_match_std", "walkDistance_team_to_match_ratio_sum", "teamKills_team_to_match_ratio_median", "DBNOs_team_match_sum", "picker_team_to_match_ratio_min", "DBNOs_team_match_std", "teamKills_team_to_match_ratio_sum", "teamKills_team_match_var", "headshot_rate_team_match_max", "kills_and_assists_team_match_median", "multi_killer_team_to_match_ratio_mean", "distance_over_weapons_match_min", "sniper_team_match_median", "weaponsAcquired_team_to_match_ratio_min", "kills_and_assists_team_match_sum", "kills_team_match_max", "kills_without_moving_team_match_median_rank", "vehicleDestroys_match_sum", "roadKills_team_to_match_ratio_mean", "boosts_team_match_median", "headshot_rate_team_match_median", "kills_team_match_min", "heals_team_match_min", "kills_without_moving_team_match_min_rank", "headshot_rate_team_match_std", "skill_team_to_match_ratio_max", "headshotKills_team_match_mean", "kills_team_match_median", "killStreaks_match_median", "sniper_team_match_std", "non_leathal_input_team_match_max", "kills_team_match_sum", "assists_team_match_sum", "DBNOs_team_match_max", "sniper_team_match_max", "assists_team_match_std", "non_leathal_input_team_match_median", "killStreaks_team_match_std", "kills_and_assists_team_match_max", "teamKills_team_match_mean", "kill_to_team_kills_ratio_team_match_var", "total_distance_team_to_match_ratio_sum", "kill_to_team_kills_ratio_team_match_max_rank", "vehicleDestroys_team_match_max_rank", "roadKills_match_sum", "DBNOs_team_match_median", "killPlace_match_max", "sniper_team_to_match_ratio_max", "kill_to_team_kills_ratio_team_to_match_ratio_sum", "revives_team_match_std", "boosts_team_match_min", "kill_to_team_kills_ratio_team_match_mean", "killStreaks_team_match_sum", "revives_team_match_sum", "roadKills_match_max", "killStreakrate_team_match_max", "kill_to_team_kills_ratio_team_to_match_ratio_median", "non_leathal_input_team_match_min", "skill_team_match_mean", "kills_without_moving_team_match_sum_rank", "walkDistance_match_sum", "kills_and_assists_team_match_min", "headshotKills_team_match_sum", "multi_killer_team_match_max_rank", "sniper_team_match_min", "vehicleDestroys_match_max", "multi_killer_team_match_var", "DBNOs_team_match_min", "headshot_rate_team_match_min", "kills_without_moving_team_match_max_rank", "roadKills_team_match_max_rank", "assists_team_match_max", "vehicleDestroys_team_to_match_ratio_median", "revives_team_match_max", "skill_team_match_sum", "killStreaks_team_match_max", "headshotKills_team_match_max", "killStreakrate_team_to_match_ratio_max", "sniper_match_max", "revives_team_match_median", "skill_team_match_std", "kill_to_team_kills_ratio_team_match_sum", "vehicleDestroys_team_match_var", "headshotKills_team_match_std", "multi_killer_team_to_match_ratio_sum", "assists_team_match_median", "multi_killer_team_to_match_ratio_median", "kill_to_team_kills_ratio_team_to_match_ratio_max", "damageDealt_team_to_match_ratio_min", "headshotKills_team_match_median", "longestKill_team_to_match_ratio_min", "multi_killer_team_match_mean", "assists_team_match_min", "teamKills_team_to_match_ratio_max", "killStreaks_team_match_median", "skill_team_match_max", "roadKills_team_to_match_ratio_median", "non_leathal_input_team_to_match_ratio_min", "vehicleDestroys_team_to_match_ratio_sum", "teamKills_team_match_std", "roadKills_team_match_var", "multi_killer_team_to_match_ratio_max", "killStreakrate_team_to_match_ratio_min", "multi_killer_team_match_sum", "roadKills_team_to_match_ratio_sum", "kill_to_team_kills_ratio_team_match_std", "roadKills_team_match_mean", "vehicleDestroys_team_match_mean", "teamKills_team_match_sum", "kills_and_assists_team_to_match_ratio_min", "sniper_match_median", "kill_to_team_kills_ratio_team_match_max", "weaponsAcquired_match_min", "total_distance_match_sum", "kills_without_moving_match_sum", "killsPerWalkDistance_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_match_median", "killStreaks_team_match_min", "revives_team_match_min", "kills_without_moving_team_match_var", "skill_team_match_median", "picker_match_min", "kills_team_to_match_ratio_min", "headshotKills_team_match_min", "assists_match_median", "killsPerWalkDistance_match_min", "multi_killer_team_match_median", "multi_killer_team_match_std", "skill_team_match_min", "longestKill_match_min", "matchDuration", "headshotKills_match_median", "teamKills_team_match_max", "roadKills_team_match_std", "vehicleDestroys_team_match_sum", "killStreaks_team_to_match_ratio_min", "kills_without_moving_team_match_mean", "vehicleDestroys_team_match_max", "damageDealt_match_min", "roadKills_team_to_match_ratio_max", "killStreakrate_match_min", "assists_team_to_match_ratio_min", "teamKills_team_match_median", "vehicleDestroys_team_match_std", "multi_killer_team_match_max", "skill_match_median", "roadKills_team_match_sum", "kills_without_moving_team_match_sum", "headshot_rate_match_median", "roadKills_team_match_max", "DBNOs_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_match_min", "teamKills_team_match_min", "kills_without_moving_team_match_std", "killStreakrate_match_max", "healsAndBoostsPerWalkDistance_match_min", "kills_match_min", "health_items_team_to_match_ratio_min", "healsAndBoostsPerWalkDistance_team_to_match_ratio_min", "revives_match_median", "rideDistance_match_min", "rideDistance_team_to_match_ratio_min", "multi_killer_team_match_min", "vehicleDestroys_team_to_match_ratio_max", "vehicleDestroys_team_match_median", "health_items_match_min", "killStreaks_match_min", "non_leathal_input_match_min", "roadKills_team_match_median", "swimDistance_match_median", "healsPerWalkDistance_team_to_match_ratio_min", "headshot_rate_match_min", "kills_without_moving_team_match_median", "roadKills_match_median", "teamKills_match_median", "vehicleDestroys_match_median", "kills_without_moving_match_median", "kill_to_team_kills_ratio_match_median", "multi_killer_match_median", "roadKills_team_match_min", "vehicleDestroys_team_match_min", "kills_without_moving_team_match_min", "assists_match_min", "boosts_match_min", "DBNOs_match_min", "headshotKills_match_min", "heals_match_min", "killPlace_match_min", "revives_match_min", "roadKills_match_min", "swimDistance_match_min", "teamKills_match_min", "vehicleDestroys_match_min", "kills_and_assists_match_min", "kills_without_moving_match_min", "boostsPerWalkDistance_match_min", "healsPerWalkDistance_match_min", "skill_match_min", "kill_to_team_kills_ratio_match_min", "multi_killer_match_min", "sniper_match_min", "boosts_team_to_match_ratio_min", "headshotKills_team_to_match_ratio_min", "heals_team_to_match_ratio_min", "revives_team_to_match_ratio_min", "roadKills_team_to_match_ratio_min", "swimDistance_team_to_match_ratio_min", "teamKills_team_to_match_ratio_min", "vehicleDestroys_team_to_match_ratio_min", "boostsPerWalkDistance_team_to_match_ratio_min", "headshot_rate_team_to_match_ratio_min", "skill_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_to_match_ratio_min", "multi_killer_team_to_match_ratio_min", "sniper_team_to_match_ratio_min", "kills_without_moving_team_match_max", "kills_without_moving_match_max", "kill_to_team_kills_ratio_team_match_median_rank", "healsAndBoostsPerWalkDistance_team_to_match_ratio_max", "kills_and_assists_team_to_match_ratio_mean", "longestKill_team_match_max", "health_items_team_match_var", "killPoints_match_median", "healsAndBoostsPerWalkDistance_team_match_min_rank", "DBNOs_team_match_var", "headshot_rate_team_match_median_rank", "boostsPerWalkDistance_team_match_min_rank", "healsPerWalkDistance_team_match_median_rank", "killPoints_team_match_max", "assists_team_match_max_rank", "headshot_rate_team_to_match_ratio_sum", "walkDistance_match_min", "DBNOs_team_to_match_ratio_median", "rideDistance_team_match_max_rank", "non_leathal_input_match_max", "health_items_team_match_max_rank", "winPoints_match_mean", "heals_team_match_sum_rank", "total_distance_match_min", "revives_team_to_match_ratio_mean", "healsAndBoostsPerWalkDistance_team_to_match_ratio_sum", "heals_team_match_max_rank", "damageDealt_team_match_min", "healsPerWalkDistance_team_match_var", "assists_team_to_match_ratio_mean", "healsPerWalkDistance_team_to_match_ratio_max", "boostsPerWalkDistance_team_to_match_ratio_mean", "teamKills_team_match_min_rank", "boostsPerWalkDistance_team_match_var", "boostsPerWalkDistance_team_match_sum", "killStreakrate_team_match_var", "winPoints_team_match_min", "killPoints_team_match_min", "healsAndBoostsPerWalkDistance_team_match_sum_rank", "health_items_team_to_match_ratio_sum", "rideDistance_match_median", "revives_team_match_max_rank", "boostsPerWalkDistance_team_match_max", "killPoints_match_min", "boostsPerWalkDistance_team_match_mean", "boostsPerWalkDistance_team_match_sum_rank", "killStreaks_team_to_match_ratio_median", "killStreaks_team_to_match_ratio_sum", "healsAndBoostsPerWalkDistance_team_match_max", "healsPerWalkDistance_team_to_match_ratio_sum", "killsPerWalkDistance_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_max_rank", "heals_team_to_match_ratio_mean", "heals_team_match_var", "kills_and_assists_team_match_var", "boostsPerWalkDistance_team_match_max_rank", "non_leathal_input_team_to_match_ratio_median", "boosts_team_match_var", "healsPerWalkDistance_team_match_sum", "kills_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_mean", "heals_team_to_match_ratio_sum", "healsAndBoostsPerWalkDistance_team_to_match_ratio_mean", "healsAndBoostsPerWalkDistance_team_match_sum", "multi_killer_team_match_median_rank", "healsPerWalkDistance_team_to_match_ratio_mean", "healsPerWalkDistance_team_match_sum_rank", "healsPerWalkDistance_team_match_mean", "healsPerWalkDistance_team_match_max_rank", "health_items_team_to_match_ratio_max", "boostsPerWalkDistance_team_to_match_ratio_median", "longestKill_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_to_match_ratio_median", "kills_and_assists_team_to_match_ratio_median", "longestKill_team_match_median", "boosts_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_median", "killsPerWalkDistance_team_match_median", "health_items_team_to_match_ratio_median", "healsPerWalkDistance_team_to_match_ratio_median", "healsPerWalkDistance_team_match_median", "killPlace_over_maxPlace_team_match_sum_rank", "kills_and_assists_team_match_mean", "killPoints_team_to_match_ratio_mean", "rankPoints_team_match_sum", "health_items_team_match_sum_rank", "damageDealt_team_match_std", "rankPoints_match_median", "killPoints_match_sum", "weaponsAcquired_match_max", "vehicleDestroys_team_match_mean_rank", "damageDealt_team_to_match_ratio_median", "boostsPerWalkDistance_team_to_match_ratio_sum", "boostsPerWalkDistance_team_match_median_rank", "damageDealt_team_match_median", "assists_team_match_sum_rank", "longestKill_team_to_match_ratio_max", "headshot_rate_team_match_mean_rank", "winPoints_team_match_median_rank", "non_leathal_input_match_sum", "killPoints_team_match_sum_rank", "killStreakrate_team_match_median_rank", "winPoints_team_match_sum_rank", "killPoints_team_match_median_rank", "winPoints_team_to_match_ratio_median", "DBNOs_team_to_match_ratio_sum", "killStreaks_team_match_sum_rank", "assists_match_sum", "killPoints_team_match_mean_rank", "healsPerWalkDistance_team_match_min_rank", "boostsPerWalkDistance_team_to_match_ratio_max", "revives_team_match_sum_rank", "roadKills_match_mean", "DBNOs_match_sum", "killPoints_match_mean", "winPoints_team_match_mean_rank", "headshotKills_match_sum", "longestKill_team_match_mean", "killStreaks_team_match_median_rank", "skill_match_sum", "killsPerWalkDistance_team_match_mean", "multi_killer_team_match_mean_rank", "boosts_match_max", "longestKill_team_to_match_ratio_mean", "rankPoints_team_match_max", "longestKill_team_match_sum_rank", "longestKill_team_match_max_rank", "killsPerWalkDistance_team_match_sum", "killsPerWalkDistance_team_match_var", "non_leathal_input_team_match_var", "sniper_team_match_max_rank", "rankPoints_team_match_min", "non_leathal_input_team_to_match_ratio_sum", "revives_match_sum", "sniper_team_to_match_ratio_sum", "kills_and_assists_team_to_match_ratio_sum", "rankPoints_team_match_mean", "killPlace_over_maxPlace_team_match_std", "group_size", "killPlace_over_maxPlace_team_to_match_ratio_sum", "killPlace_over_maxPlace_team_match_min_rank", "killPlace_over_maxPlace_team_match_mean_rank", "kills_without_moving_team_match_mean_rank", "killPlace_team_to_match_ratio_min", "killPlace_match_sum", "killPlace_over_maxPlace_match_min", "heals_match_max", "distance_over_weapons_team_match_sum", "damageDealt_team_match_mean", "boosts_team_match_mean_rank", "revives_team_match_min_rank", "damageDealt_team_to_match_ratio_mean", "weaponsAcquired_team_match_mean_rank", "damageDealt_team_to_match_ratio_sum", "killPoints_team_match_min_rank", "picker_team_to_match_ratio_median", "killPoints_team_to_match_ratio_sum", "total_distance_team_match_min_rank", "non_leathal_input_team_match_max_rank", "kills_match_sum", "rankPoints_team_match_median_rank", "kills_and_assists_team_match_mean_rank", "kill_to_team_kills_ratio_team_match_mean_rank", "distance_over_weapons_team_match_median_rank", "headshotKills_team_match_min_rank", "rankPoints_team_match_mean_rank", "picker_team_match_median_rank", "DBNOs_team_match_max_rank", "damageDealt_team_match_max", "distance_over_weapons_team_match_mean", "distance_over_weapons_team_match_median", "healsAndBoostsPerWalkDistance_team_match_median_rank", "total_distance_team_match_sum", "winPoints_team_to_match_ratio_sum", "non_leathal_input_team_match_mean_rank", "DBNOs_team_match_sum_rank", "non_leathal_input_team_match_sum_rank", "distance_over_weapons_team_match_mean_rank", "total_distance_team_match_max", "kills_and_assists_team_match_sum_rank", "DBNOs_team_match_mean_rank", "picker_team_to_match_ratio_max", "total_distance_team_match_std", "walkDistance_team_to_match_ratio_min", "total_distance_team_match_min", "walkDistance_team_match_std", "teamKills_team_match_median_rank", "non_leathal_input_team_to_match_ratio_mean", "damageDealt_team_match_sum", "killStreakrate_team_to_match_ratio_median", "killStreaks_team_match_mean_rank", "rideDistance_team_match_sum_rank", "killPoints_match_max", "rankPoints_match_mean", "total_distance_team_match_mean", "weaponsAcquired_team_to_match_ratio_max", "winPoints_match_sum", "kills_and_assists_team_match_max_rank", "killPoints_team_match_var", "killPlace_team_match_sum", "killStreakrate_match_median", "rideDistance_match_sum", "winPoints_match_max", "boosts_team_to_match_ratio_sum", "boosts_team_to_match_ratio_mean", "total_distance_team_to_match_ratio_min", "headshot_rate_match_max", "skill_team_match_min_rank", "winPoints_team_match_var", "picker_match_max", "winPoints_match_min", "kills_team_match_sum_rank", "kills_match_max", "kills_team_match_var", "kills_and_assists_match_max", "damageDealtRankWithinKills_team_match_sum", "health_items_team_to_match_ratio_mean", "walkDistance_team_match_median_rank", "killStreaks_team_to_match_ratio_mean", "killStreaks_match_sum", "killPlaceRankWithinKills_team_to_match_ratio_sum", "walkDistanceRankWithinKills_team_match_median", "damageDealtRankWithinKills_team_to_match_ratio_sum", "walkDistanceRankWithinKills_team_to_match_ratio_median", "total_distance_team_match_median_rank", "killsPerWalkDistance_team_match_max", "total_distance_team_to_match_ratio_median", "walkDistanceRankWithinKills_team_match_max", "killsPerWalkDistance_team_to_match_ratio_mean", "walkDistance_team_to_match_ratio_median", "walkDistanceRankWithinKills_team_match_mean", "walkDistanceRankWithinKills_team_to_match_ratio_mean", "kills_team_match_mean_rank", "killPlace_over_maxPlace_team_match_min", "assistsRankWithinKills_team_match_sum", "killsPerWalkDistance_team_to_match_ratio_sum", "killsPerWalkDistance_team_match_mean_rank", "walkDistance_team_match_mean", "healsRankWithinKills_team_match_sum", "walkDistance_team_match_median", "killPlace_over_maxPlace_team_match_median", "healsRankWithinKills_team_to_match_ratio_sum", "boostsRankWithinKills_team_to_match_ratio_sum", "killPlace_over_maxPlace_team_match_sum", "boostsRankWithinKills_team_match_sum", "walkDistanceRankWithinKills_team_match_sum", "killPlace_over_maxPlace_team_match_mean", "walkDistanceRankWithinKills_team_to_match_ratio_sum", "killsPerWalkDistance_team_match_max_rank", "picker_team_match_mean", "killsPerWalkDistance_team_match_median_rank", "killsPerWalkDistance_team_match_sum_rank", "killPlace_team_to_match_ratio_median", "damageDealtRankWithinKills_team_match_std", "killPlace_team_to_match_ratio_mean", "killPlaceRankWithinKills_match_sum", "kills_team_match_max_rank", "kills_team_to_match_ratio_mean", "killPlace_team_match_max", "killPlaceRankWithinKills_match_min", "playersJoined", "walkDistanceRankWithinKills_team_match_std", "killStreaks_team_match_min_rank", "killPlaceRankWithinKills_team_match_std", "healsRankWithinKills_team_match_std", "boostsRankWithinKills_team_match_std", "healsRankWithinKills_team_to_match_ratio_max", "numGroups", "maxPlace", "boostsRankWithinKills_team_to_match_ratio_max", "assistsRankWithinKills_team_match_std", "killPlace_over_maxPlace_team_to_match_ratio_median", "damageDealtRankWithinKills_team_to_match_ratio_max", "killStreakrate_team_match_min_rank", "killPlace_over_maxPlace_team_to_match_ratio_mean", "damageDealtRankWithinKills_match_min", "killsPerWalkDistance_team_match_min_rank", "killsPerWalkDistance_team_match_min", "killPlace_over_maxPlace_team_match_median_rank", "walkDistanceRankWithinKills_team_to_match_ratio_max", "walkDistanceRankWithinKills_match_sum", "assistsRankWithinKills_match_max", "killStreakrate_team_match_min", "healsRankWithinKills_match_sum", "assistsRankWithinKills_match_sum", "healsRankWithinKills_match_max", "boostsRankWithinKills_match_max", "boostsRankWithinKills_match_sum", "damageDealtRankWithinKills_match_sum", "damageDealtRankWithinKills_match_max", "walkDistanceRankWithinKills_match_max", "killPlaceRankWithinKills_match_max")
# v60
#UNWANTED_FEATURES = ("healsPerWalkDistance_team_match_max", "rankPoints_team_match_median", "heals_team_to_match_ratio_max", "picker_match_median", "sniper_team_to_match_ratio_mean", "longestKill_team_match_min", "kill_to_team_kills_ratio_match_sum", "longestKill_team_match_std", "roadKills_team_match_mean_rank", "healsAndBoostsPerWalkDistance_team_match_min", "sniper_team_match_mean_rank", "boostsPerWalkDistance_team_match_median", "rideDistance_team_to_match_ratio_mean", "winPoints_team_match_max", "headshot_rate_team_to_match_ratio_mean", "winPoints_match_median", "swimDistance_team_match_sum_rank", "killPoints_team_match_median", "vehicleDestroys_team_match_median_rank", "DBNOs_team_to_match_ratio_max", "killPoints_team_match_mean", "multi_killer_match_sum", "weaponsAcquired_team_match_mean", "healsPerWalkDistance_team_match_min", "heals_team_to_match_ratio_median", "kills_team_to_match_ratio_max", "headshot_rate_team_match_max_rank", "rideDistance_team_match_std", "boostsPerWalkDistance_team_match_min", "killPoints_team_match_sum", "rideDistance_team_to_match_ratio_max", "sniper_team_match_sum_rank", "health_items_match_median", "winPoints_team_match_sum", "revives_team_to_match_ratio_sum", "non_leathal_input_team_to_match_ratio_max", "DBNOs_match_max", "kills_and_assists_team_to_match_ratio_max", "headshot_rate_team_match_sum_rank", "killStreaks_team_match_var", "skill_team_match_sum_rank", "assists_team_to_match_ratio_sum", "revives_team_to_match_ratio_median", "headshotKills_match_max", "headshot_rate_team_match_var", "distance_over_weapons_team_match_std", "rideDistance_team_match_sum", "headshotKills_team_match_max_rank", "winPoints_team_match_mean", "rideDistance_team_to_match_ratio_median", "sniper_team_match_var", "sniper_team_match_median_rank", "boosts_team_to_match_ratio_max", "headshotKills_team_to_match_ratio_mean", "rideDistance_team_match_max", "headshot_rate_team_match_min_rank", "teamKills_team_match_sum_rank", "weaponsAcquired_match_median", "assists_team_to_match_ratio_median", "winPoints_team_match_median", "rideDistance_team_to_match_ratio_sum", "skill_match_max", "picker_team_match_sum", "rideDistance_team_match_mean", "kill_to_team_kills_ratio_team_match_min_rank", "swimDistance_team_to_match_ratio_mean", "assists_match_max", "killStreakrate_team_match_mean", "skill_team_match_max_rank", "teamKills_match_sum", "kill_to_team_kills_ratio_match_max", "sniper_team_to_match_ratio_median", "roadKills_team_match_median_rank", "killStreaks_match_max", "rideDistance_team_match_min", "headshotKills_team_match_sum_rank", "kills_and_assists_match_median", "assists_team_match_var", "weaponsAcquired_team_match_sum", "teamKills_team_to_match_ratio_mean", "rideDistance_team_match_var", "killStreakrate_team_match_sum", "revives_team_match_var", "headshotKills_team_to_match_ratio_sum", "headshot_rate_team_to_match_ratio_max", "headshot_rate_team_to_match_ratio_median", "health_items_team_match_mean", "skill_team_to_match_ratio_mean", "rideDistance_team_match_median", "swimDistance_team_to_match_ratio_sum", "revives_match_max", "health_items_team_match_sum", "kill_to_team_kills_ratio_team_match_sum_rank", "heals_team_match_mean", "rankPoints_team_match_std", "skill_team_to_match_ratio_sum", "headshotKills_team_match_var", "killPlace_team_match_std", "swimDistance_team_match_var", "vehicleDestroys_team_match_min_rank", "swimDistance_team_to_match_ratio_median", "swimDistance_team_to_match_ratio_max", "roadKills_team_match_min_rank", "boosts_match_median", "heals_team_match_sum", "picker_team_match_std", "non_leathal_input_team_match_mean", "heals_match_median", "picker_team_match_max", "boosts_team_match_mean", "swimDistance_team_match_max_rank", "DBNOs_match_median", "headshotKills_team_to_match_ratio_median", "weaponsAcquired_team_match_std", "skill_team_to_match_ratio_median", "killPoints_team_match_std", "DBNOs_team_match_mean", "picker_team_match_median", "sniper_team_match_mean", "sniper_team_match_sum", "swimDistance_team_match_max", "winPoints_team_match_std", "sniper_team_match_min_rank", "headshot_rate_team_match_mean", "swimDistance_team_match_mean", "picker_team_match_min", "health_items_team_match_std", "vehicleDestroys_team_match_sum_rank", "killStreaks_team_to_match_ratio_max", "weaponsAcquired_team_match_median", "multi_killer_team_match_min_rank", "kills_match_median", "swimDistance_team_match_std", "assists_team_to_match_ratio_max", "swimDistance_team_match_sum", "teamKills_team_match_max_rank", "health_items_team_match_max", "healsAndBoostsPerWalkDistance_team_match_std", "weaponsAcquired_team_match_max", "kill_to_team_kills_ratio_team_to_match_ratio_mean", "revives_team_match_mean", "killStreakrate_team_match_median", "non_leathal_input_match_median", "killsPerWalkDistance_team_match_std", "swimDistance_team_match_min", "multi_killer_team_match_sum_rank", "assists_team_match_mean", "heals_team_match_median", "teamKills_match_max", "distance_over_weapons_team_to_match_ratio_min", "headshot_rate_team_match_sum", "multi_killer_match_max", "swimDistance_team_match_median", "health_items_team_match_median", "boosts_team_match_sum", "boosts_team_match_max", "skill_team_match_var", "healsPerWalkDistance_team_match_std", "boostsPerWalkDistance_team_match_std", "kills_team_match_mean", "heals_team_match_max", "killStreakrate_team_match_std", "weaponsAcquired_team_match_min", "roadKills_team_match_sum_rank", "headshotKills_team_to_match_ratio_max", "revives_team_to_match_ratio_max", "heals_team_match_std", "vehicleDestroys_team_to_match_ratio_mean", "non_leathal_input_team_match_std", "health_items_team_match_min", "non_leathal_input_team_match_sum", "kills_without_moving_match_mean", "boosts_team_match_std", "kills_team_match_std", "killStreaks_team_match_mean", "kills_and_assists_team_match_std", "walkDistance_team_to_match_ratio_sum", "teamKills_team_to_match_ratio_median", "DBNOs_team_match_sum", "picker_team_to_match_ratio_min", "DBNOs_team_match_std", "teamKills_team_to_match_ratio_sum", "teamKills_team_match_var", "headshot_rate_team_match_max", "kills_and_assists_team_match_median", "multi_killer_team_to_match_ratio_mean", "distance_over_weapons_match_min", "sniper_team_match_median", "weaponsAcquired_team_to_match_ratio_min", "kills_and_assists_team_match_sum", "kills_team_match_max", "kills_without_moving_team_match_median_rank", "vehicleDestroys_match_sum", "roadKills_team_to_match_ratio_mean", "boosts_team_match_median", "headshot_rate_team_match_median", "kills_team_match_min", "heals_team_match_min", "kills_without_moving_team_match_min_rank", "headshot_rate_team_match_std", "skill_team_to_match_ratio_max", "headshotKills_team_match_mean", "kills_team_match_median", "killStreaks_match_median", "sniper_team_match_std", "non_leathal_input_team_match_max", "kills_team_match_sum", "assists_team_match_sum", "DBNOs_team_match_max", "sniper_team_match_max", "assists_team_match_std", "non_leathal_input_team_match_median", "killStreaks_team_match_std", "kills_and_assists_team_match_max", "teamKills_team_match_mean", "kill_to_team_kills_ratio_team_match_var", "total_distance_team_to_match_ratio_sum", "kill_to_team_kills_ratio_team_match_max_rank", "vehicleDestroys_team_match_max_rank", "roadKills_match_sum", "DBNOs_team_match_median", "killPlace_match_max", "sniper_team_to_match_ratio_max", "kill_to_team_kills_ratio_team_to_match_ratio_sum", "revives_team_match_std", "boosts_team_match_min", "kill_to_team_kills_ratio_team_match_mean", "killStreaks_team_match_sum", "revives_team_match_sum", "roadKills_match_max", "killStreakrate_team_match_max", "kill_to_team_kills_ratio_team_to_match_ratio_median", "non_leathal_input_team_match_min", "skill_team_match_mean", "kills_without_moving_team_match_sum_rank", "walkDistance_match_sum", "kills_and_assists_team_match_min", "headshotKills_team_match_sum", "multi_killer_team_match_max_rank", "sniper_team_match_min", "vehicleDestroys_match_max", "multi_killer_team_match_var", "DBNOs_team_match_min", "headshot_rate_team_match_min", "kills_without_moving_team_match_max_rank", "roadKills_team_match_max_rank", "assists_team_match_max", "vehicleDestroys_team_to_match_ratio_median", "revives_team_match_max", "skill_team_match_sum", "killStreaks_team_match_max", "headshotKills_team_match_max", "killStreakrate_team_to_match_ratio_max", "sniper_match_max", "revives_team_match_median", "skill_team_match_std", "kill_to_team_kills_ratio_team_match_sum", "vehicleDestroys_team_match_var", "headshotKills_team_match_std", "multi_killer_team_to_match_ratio_sum", "assists_team_match_median", "multi_killer_team_to_match_ratio_median", "kill_to_team_kills_ratio_team_to_match_ratio_max", "damageDealt_team_to_match_ratio_min", "headshotKills_team_match_median", "longestKill_team_to_match_ratio_min", "multi_killer_team_match_mean", "assists_team_match_min", "teamKills_team_to_match_ratio_max", "killStreaks_team_match_median", "skill_team_match_max", "roadKills_team_to_match_ratio_median", "non_leathal_input_team_to_match_ratio_min", "vehicleDestroys_team_to_match_ratio_sum", "teamKills_team_match_std", "roadKills_team_match_var", "multi_killer_team_to_match_ratio_max", "killStreakrate_team_to_match_ratio_min", "multi_killer_team_match_sum", "roadKills_team_to_match_ratio_sum", "kill_to_team_kills_ratio_team_match_std", "roadKills_team_match_mean", "vehicleDestroys_team_match_mean", "teamKills_team_match_sum", "kills_and_assists_team_to_match_ratio_min", "sniper_match_median", "kill_to_team_kills_ratio_team_match_max", "weaponsAcquired_match_min", "total_distance_match_sum", "kills_without_moving_match_sum", "killsPerWalkDistance_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_match_median", "killStreaks_team_match_min", "revives_team_match_min", "kills_without_moving_team_match_var", "skill_team_match_median", "picker_match_min", "kills_team_to_match_ratio_min", "headshotKills_team_match_min", "assists_match_median", "killsPerWalkDistance_match_min", "multi_killer_team_match_median", "multi_killer_team_match_std", "skill_team_match_min", "longestKill_match_min", "matchDuration", "headshotKills_match_median", "teamKills_team_match_max", "roadKills_team_match_std", "vehicleDestroys_team_match_sum", "killStreaks_team_to_match_ratio_min", "kills_without_moving_team_match_mean", "vehicleDestroys_team_match_max", "damageDealt_match_min", "roadKills_team_to_match_ratio_max", "killStreakrate_match_min", "assists_team_to_match_ratio_min", "teamKills_team_match_median", "vehicleDestroys_team_match_std", "multi_killer_team_match_max", "skill_match_median", "roadKills_team_match_sum", "kills_without_moving_team_match_sum", "headshot_rate_match_median", "roadKills_team_match_max", "DBNOs_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_match_min", "teamKills_team_match_min", "kills_without_moving_team_match_std", "killStreakrate_match_max", "healsAndBoostsPerWalkDistance_match_min", "kills_match_min", "health_items_team_to_match_ratio_min", "healsAndBoostsPerWalkDistance_team_to_match_ratio_min", "revives_match_median", "rideDistance_match_min", "rideDistance_team_to_match_ratio_min", "multi_killer_team_match_min", "vehicleDestroys_team_to_match_ratio_max", "vehicleDestroys_team_match_median", "health_items_match_min", "killStreaks_match_min", "non_leathal_input_match_min", "roadKills_team_match_median", "swimDistance_match_median", "healsPerWalkDistance_team_to_match_ratio_min", "headshot_rate_match_min", "kills_without_moving_team_match_median", "roadKills_match_median", "teamKills_match_median", "vehicleDestroys_match_median", "kills_without_moving_match_median", "kill_to_team_kills_ratio_match_median", "multi_killer_match_median", "roadKills_team_match_min", "vehicleDestroys_team_match_min", "kills_without_moving_team_match_min", "assists_match_min", "boosts_match_min", "DBNOs_match_min", "headshotKills_match_min", "heals_match_min", "killPlace_match_min", "revives_match_min", "roadKills_match_min", "swimDistance_match_min", "teamKills_match_min", "vehicleDestroys_match_min", "kills_and_assists_match_min", "kills_without_moving_match_min", "boostsPerWalkDistance_match_min", "healsPerWalkDistance_match_min", "skill_match_min", "kill_to_team_kills_ratio_match_min", "multi_killer_match_min", "sniper_match_min", "boosts_team_to_match_ratio_min", "headshotKills_team_to_match_ratio_min", "heals_team_to_match_ratio_min", "revives_team_to_match_ratio_min", "roadKills_team_to_match_ratio_min", "swimDistance_team_to_match_ratio_min", "teamKills_team_to_match_ratio_min", "vehicleDestroys_team_to_match_ratio_min", "boostsPerWalkDistance_team_to_match_ratio_min", "headshot_rate_team_to_match_ratio_min", "skill_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_to_match_ratio_min", "multi_killer_team_to_match_ratio_min", "sniper_team_to_match_ratio_min", "kills_without_moving_team_match_max", "kills_without_moving_match_max", "kill_to_team_kills_ratio_team_match_median_rank", "healsAndBoostsPerWalkDistance_team_to_match_ratio_max", "kills_and_assists_team_to_match_ratio_mean", "longestKill_team_match_max", "health_items_team_match_var", "killPoints_match_median", "healsAndBoostsPerWalkDistance_team_match_min_rank", "DBNOs_team_match_var", "headshot_rate_team_match_median_rank", "boostsPerWalkDistance_team_match_min_rank", "healsPerWalkDistance_team_match_median_rank", "killPoints_team_match_max", "assists_team_match_max_rank", "headshot_rate_team_to_match_ratio_sum", "walkDistance_match_min", "DBNOs_team_to_match_ratio_median", "rideDistance_team_match_max_rank", "non_leathal_input_match_max", "health_items_team_match_max_rank", "winPoints_match_mean", "heals_team_match_sum_rank", "total_distance_match_min", "revives_team_to_match_ratio_mean", "healsAndBoostsPerWalkDistance_team_to_match_ratio_sum", "heals_team_match_max_rank", "damageDealt_team_match_min", "healsPerWalkDistance_team_match_var", "assists_team_to_match_ratio_mean", "healsPerWalkDistance_team_to_match_ratio_max", "boostsPerWalkDistance_team_to_match_ratio_mean", "teamKills_team_match_min_rank", "boostsPerWalkDistance_team_match_var", "boostsPerWalkDistance_team_match_sum", "killStreakrate_team_match_var", "winPoints_team_match_min", "killPoints_team_match_min", "healsAndBoostsPerWalkDistance_team_match_sum_rank", "health_items_team_to_match_ratio_sum", "rideDistance_match_median", "revives_team_match_max_rank", "boostsPerWalkDistance_team_match_max", "killPoints_match_min", "boostsPerWalkDistance_team_match_mean", "boostsPerWalkDistance_team_match_sum_rank", "killStreaks_team_to_match_ratio_median", "killStreaks_team_to_match_ratio_sum", "healsAndBoostsPerWalkDistance_team_match_max", "healsPerWalkDistance_team_to_match_ratio_sum", "killsPerWalkDistance_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_max_rank", "heals_team_to_match_ratio_mean", "heals_team_match_var", "kills_and_assists_team_match_var", "boostsPerWalkDistance_team_match_max_rank", "non_leathal_input_team_to_match_ratio_median", "boosts_team_match_var", "healsPerWalkDistance_team_match_sum", "kills_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_mean", "heals_team_to_match_ratio_sum", "healsAndBoostsPerWalkDistance_team_to_match_ratio_mean", "healsAndBoostsPerWalkDistance_team_match_sum", "multi_killer_team_match_median_rank", "healsPerWalkDistance_team_to_match_ratio_mean", "healsPerWalkDistance_team_match_sum_rank", "healsPerWalkDistance_team_match_mean", "healsPerWalkDistance_team_match_max_rank", "health_items_team_to_match_ratio_max", "boostsPerWalkDistance_team_to_match_ratio_median", "longestKill_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_to_match_ratio_median", "kills_and_assists_team_to_match_ratio_median", "longestKill_team_match_median", "boosts_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_median", "killsPerWalkDistance_team_match_median", "health_items_team_to_match_ratio_median", "healsPerWalkDistance_team_to_match_ratio_median", "healsPerWalkDistance_team_match_median", "killPlace_over_maxPlace_team_match_sum_rank", "kills_and_assists_team_match_mean", "killPoints_team_to_match_ratio_mean", "rankPoints_team_match_sum", "health_items_team_match_sum_rank", "damageDealt_team_match_std", "rankPoints_match_median", "killPoints_match_sum", "weaponsAcquired_match_max", "vehicleDestroys_team_match_mean_rank", "damageDealt_team_to_match_ratio_median", "boostsPerWalkDistance_team_to_match_ratio_sum", "boostsPerWalkDistance_team_match_median_rank", "damageDealt_team_match_median", "assists_team_match_sum_rank", "longestKill_team_to_match_ratio_max", "headshot_rate_team_match_mean_rank", "winPoints_team_match_median_rank", "non_leathal_input_match_sum", "killPoints_team_match_sum_rank", "killStreakrate_team_match_median_rank", "winPoints_team_match_sum_rank", "killPoints_team_match_median_rank", "winPoints_team_to_match_ratio_median", "DBNOs_team_to_match_ratio_sum", "killStreaks_team_match_sum_rank", "assists_match_sum", "killPoints_team_match_mean_rank", "healsPerWalkDistance_team_match_min_rank", "boostsPerWalkDistance_team_to_match_ratio_max", "revives_team_match_sum_rank", "roadKills_match_mean", "DBNOs_match_sum", "killPoints_match_mean", "winPoints_team_match_mean_rank", "headshotKills_match_sum", "longestKill_team_match_mean", "killStreaks_team_match_median_rank", "skill_match_sum", "killsPerWalkDistance_team_match_mean", "multi_killer_team_match_mean_rank", "boosts_match_max", "longestKill_team_to_match_ratio_mean", "rankPoints_team_match_max", "longestKill_team_match_sum_rank", "longestKill_team_match_max_rank", "killsPerWalkDistance_team_match_sum", "killsPerWalkDistance_team_match_var", "non_leathal_input_team_match_var", "sniper_team_match_max_rank", "rankPoints_team_match_min", "non_leathal_input_team_to_match_ratio_sum", "revives_match_sum", "sniper_team_to_match_ratio_sum", "kills_and_assists_team_to_match_ratio_sum", "rankPoints_team_match_mean", "killPlace_over_maxPlace_team_match_std", "group_size", "killPlace_over_maxPlace_team_to_match_ratio_sum", "killPlace_over_maxPlace_team_match_min_rank", "killPlace_over_maxPlace_team_match_mean_rank", "kills_without_moving_team_match_mean_rank", "killPlace_team_to_match_ratio_min", "killPlace_match_sum", "killPlace_over_maxPlace_match_min", "heals_match_max", "distance_over_weapons_team_match_sum", "damageDealt_team_match_mean", "boosts_team_match_mean_rank", "revives_team_match_min_rank", "damageDealt_team_to_match_ratio_mean", "weaponsAcquired_team_match_mean_rank", "damageDealt_team_to_match_ratio_sum", "killPoints_team_match_min_rank", "picker_team_to_match_ratio_median", "killPoints_team_to_match_ratio_sum", "total_distance_team_match_min_rank", "non_leathal_input_team_match_max_rank", "kills_match_sum", "rankPoints_team_match_median_rank", "kills_and_assists_team_match_mean_rank", "kill_to_team_kills_ratio_team_match_mean_rank", "distance_over_weapons_team_match_median_rank", "headshotKills_team_match_min_rank", "rankPoints_team_match_mean_rank", "picker_team_match_median_rank", "DBNOs_team_match_max_rank", "damageDealt_team_match_max", "distance_over_weapons_team_match_mean", "distance_over_weapons_team_match_median", "healsAndBoostsPerWalkDistance_team_match_median_rank", "total_distance_team_match_sum", "winPoints_team_to_match_ratio_sum", "non_leathal_input_team_match_mean_rank", "DBNOs_team_match_sum_rank", "non_leathal_input_team_match_sum_rank", "distance_over_weapons_team_match_mean_rank", "total_distance_team_match_max", "kills_and_assists_team_match_sum_rank", "DBNOs_team_match_mean_rank", "picker_team_to_match_ratio_max", "total_distance_team_match_std", "walkDistance_team_to_match_ratio_min", "total_distance_team_match_min", "walkDistance_team_match_std", "teamKills_team_match_median_rank", "non_leathal_input_team_to_match_ratio_mean", "damageDealt_team_match_sum", "killStreakrate_team_to_match_ratio_median", "killStreaks_team_match_mean_rank", "rideDistance_team_match_sum_rank", "killPoints_match_max", "rankPoints_match_mean", "total_distance_team_match_mean", "weaponsAcquired_team_to_match_ratio_max", "winPoints_match_sum", "kills_and_assists_team_match_max_rank", "killPoints_team_match_var", "killPlace_team_match_sum", "killStreakrate_match_median", "rideDistance_match_sum", "winPoints_match_max", "boosts_team_to_match_ratio_sum", "boosts_team_to_match_ratio_mean", "total_distance_team_to_match_ratio_min", "headshot_rate_match_max", "skill_team_match_min_rank", "winPoints_team_match_var", "picker_match_max", "winPoints_match_min", "kills_team_match_sum_rank", "kills_match_max", "kills_team_match_var", "kills_and_assists_match_max", "damageDealtRankWithinKills_team_match_sum", "health_items_team_to_match_ratio_mean", "walkDistance_team_match_median_rank", "killStreaks_team_to_match_ratio_mean", "killStreaks_match_sum", "killPlaceRankWithinKills_team_to_match_ratio_sum", "walkDistanceRankWithinKills_team_match_median", "damageDealtRankWithinKills_team_to_match_ratio_sum", "walkDistanceRankWithinKills_team_to_match_ratio_median", "total_distance_team_match_median_rank", "killsPerWalkDistance_team_match_max", "total_distance_team_to_match_ratio_median", "walkDistanceRankWithinKills_team_match_max", "killsPerWalkDistance_team_to_match_ratio_mean", "walkDistance_team_to_match_ratio_median", "walkDistanceRankWithinKills_team_match_mean", "walkDistanceRankWithinKills_team_to_match_ratio_mean", "kills_team_match_mean_rank", "killPlace_over_maxPlace_team_match_min", "assistsRankWithinKills_team_match_sum", "killsPerWalkDistance_team_to_match_ratio_sum", "killsPerWalkDistance_team_match_mean_rank", "walkDistance_team_match_mean", "healsRankWithinKills_team_match_sum", "walkDistance_team_match_median", "killPlace_over_maxPlace_team_match_median", "healsRankWithinKills_team_to_match_ratio_sum", "boostsRankWithinKills_team_to_match_ratio_sum", "killPlace_over_maxPlace_team_match_sum", "boostsRankWithinKills_team_match_sum", "walkDistanceRankWithinKills_team_match_sum", "killPlace_over_maxPlace_team_match_mean", "walkDistanceRankWithinKills_team_to_match_ratio_sum", "killsPerWalkDistance_team_match_max_rank", "picker_team_match_mean", "killsPerWalkDistance_team_match_median_rank", "killsPerWalkDistance_team_match_sum_rank", "killPlace_team_to_match_ratio_median", "damageDealtRankWithinKills_team_match_std", "killPlace_team_to_match_ratio_mean", "killPlaceRankWithinKills_match_sum", "kills_team_match_max_rank", "kills_team_to_match_ratio_mean", "killPlace_team_match_max", "killPlaceRankWithinKills_match_min", "playersJoined", "walkDistanceRankWithinKills_team_match_std", "killStreaks_team_match_min_rank", "killPlaceRankWithinKills_team_match_std", "healsRankWithinKills_team_match_std", "boostsRankWithinKills_team_match_std", "healsRankWithinKills_team_to_match_ratio_max", "numGroups", "maxPlace", "boostsRankWithinKills_team_to_match_ratio_max", "assistsRankWithinKills_team_match_std", "killPlace_over_maxPlace_team_to_match_ratio_median", "damageDealtRankWithinKills_team_to_match_ratio_max", "killStreakrate_team_match_min_rank", "killPlace_over_maxPlace_team_to_match_ratio_mean", "damageDealtRankWithinKills_match_min", "killsPerWalkDistance_team_match_min_rank", "killsPerWalkDistance_team_match_min", "killPlace_over_maxPlace_team_match_median_rank", "walkDistanceRankWithinKills_team_to_match_ratio_max", "walkDistanceRankWithinKills_match_sum", "assistsRankWithinKills_match_max", "killStreakrate_team_match_min", "healsRankWithinKills_match_sum", "assistsRankWithinKills_match_sum", "healsRankWithinKills_match_max", "boostsRankWithinKills_match_max", "boostsRankWithinKills_match_sum", "damageDealtRankWithinKills_match_sum", "damageDealtRankWithinKills_match_max", "walkDistanceRankWithinKills_match_max", "killPlaceRankWithinKills_match_max", "killPlace_over_maxPlace_team_match_max_rank", "killPlace_over_maxPlace_team_to_match_ratio_min", "killPlace_over_maxPlace_team_to_match_ratio_max", "killPlaceRankWithinKills_0_team_to_match_ratio_max", "killPlaceRankWithinKills_1_team_to_match_ratio_max", "killPlaceRankWithinKills_5_team_to_match_ratio_max", "killPlaceRankWithinKills_9_team_to_match_ratio_median", "killPlaceRankWithinKills_3_team_match_std", "killPlaceRankWithinKills_3_team_match_median", "killPlaceRankWithinKills_5_team_match_var", "killPlaceRankWithinKills_2_team_match_min", "killPlaceRankWithinKills_5_team_match_sum", "killPlaceRankWithinKills_6_team_to_match_ratio_max", "killPlaceRankWithinKills_5_team_match_mean", "killPlaceRankWithinKills_5_team_match_max", "killPlaceRankWithinKills_6_team_match_var", "killPlaceRankWithinKills_6_team_match_sum", "killPlaceRankWithinKills_7_team_to_match_ratio_max", "killPlaceRankWithinKills_4_team_match_std", "killPlaceRankWithinKills_4_team_match_median", "killPlaceRankWithinKills_6_team_match_mean", "killPlaceRankWithinKills_7_team_match_var", "killPlaceRankWithinKills_6_team_match_max", "killPlaceRankWithinKills_8_team_to_match_ratio_max", "killPlaceRankWithinKills_6_match_max", "killPlaceRankWithinKills_1_team_to_match_ratio_min", "killPlaceRankWithinKills_3_team_match_min", "killPlaceRankWithinKills_7_team_match_sum", "killPlaceRankWithinKills_5_team_match_std", "killPlaceRankWithinKills_5_match_max", "killPlaceRankWithinKills_10andMore_team_match_var", "killPlaceRankWithinKills_7_team_match_mean", "killPlaceRankWithinKills_1_match_median", "killPlaceRankWithinKills_8_team_match_var", "killPlaceRankWithinKills_10andMore_team_to_match_ratio_max", "killPlaceRankWithinKills_9_team_to_match_ratio_max", "killPlaceRankWithinKills_7_match_max", "killPlaceRankWithinKills_10andMore_team_match_sum", "killPlaceRankWithinKills_4_match_max", "killPlaceRankWithinKills_8_team_match_sum", "killPlaceRankWithinKills_5_team_match_median", "killPlaceRankWithinKills_8_match_max", "killPlaceRankWithinKills_9_team_match_var", "killPlaceRankWithinKills_7_team_match_max", "killPlaceRankWithinKills_6_team_match_std", "killPlaceRankWithinKills_8_team_match_mean", "killPlaceRankWithinKills_9_team_match_sum", "killPlaceRankWithinKills_10andMore_team_match_mean", "killPlaceRankWithinKills_6_team_match_median", "killPlaceRankWithinKills_4_team_match_min", "killPlaceRankWithinKills_10andMore_match_max", "killPlaceRankWithinKills_8_team_match_max", "killPlaceRankWithinKills_9_team_match_mean", "killPlaceRankWithinKills_9_match_max", "killPlaceRankWithinKills_7_team_match_std", "killPlaceRankWithinKills_2_team_to_match_ratio_min", "killPlaceRankWithinKills_5_team_match_min", "killPlaceRankWithinKills_8_team_match_std", "killPlaceRankWithinKills_10andMore_team_match_std", "killPlaceRankWithinKills_10andMore_team_match_median", "killPlaceRankWithinKills_9_team_match_max", "killPlaceRankWithinKills_10andMore_match_median", "killPlaceRankWithinKills_10andMore_team_match_max", "killPlaceRankWithinKills_6_team_match_min", "killPlaceRankWithinKills_3_match_max", "killPlaceRankWithinKills_3_team_to_match_ratio_min", "killPlaceRankWithinKills_2_match_max", "killPlaceRankWithinKills_7_team_match_median", "killPlaceRankWithinKills_7_team_match_min", "killPlaceRankWithinKills_8_team_match_min", "killPlaceRankWithinKills_1_match_max", "killPlaceRankWithinKills_8_team_match_median", "killPlaceRankWithinKills_9_team_match_std", "killPlaceRankWithinKills_10andMore_team_match_min", "killPlaceRankWithinKills_2_match_median", "killPlaceRankWithinKills_3_match_median", "killPlaceRankWithinKills_4_match_median", "killPlaceRankWithinKills_5_match_median", "killPlaceRankWithinKills_6_match_median", "killPlaceRankWithinKills_7_match_median", "killPlaceRankWithinKills_8_match_median", "killPlaceRankWithinKills_9_match_median", "killPlaceRankWithinKills_9_team_match_median", "killPlaceRankWithinKills_0_match_min", "killPlaceRankWithinKills_1_match_min", "killPlaceRankWithinKills_2_match_min", "killPlaceRankWithinKills_3_match_min", "killPlaceRankWithinKills_4_match_min", "killPlaceRankWithinKills_5_match_min", "killPlaceRankWithinKills_6_match_min", "killPlaceRankWithinKills_7_match_min", "killPlaceRankWithinKills_8_match_min", "killPlaceRankWithinKills_9_match_min", "killPlaceRankWithinKills_10andMore_match_min", "killPlaceRankWithinKills_9_team_match_min", "killPlaceRankWithinKills_4_team_to_match_ratio_min", "killPlaceRankWithinKills_5_team_to_match_ratio_min", "killPlaceRankWithinKills_6_team_to_match_ratio_min", "killPlaceRankWithinKills_7_team_to_match_ratio_min", "killPlaceRankWithinKills_8_team_to_match_ratio_min", "killPlaceRankWithinKills_9_team_to_match_ratio_min", "killPlaceRankWithinKills_10andMore_team_to_match_ratio_min", "killPlaceRankWithinKills_0_match_max")
# v61
UNWANTED_FEATURES = ("healsPerWalkDistance_team_match_max", "rankPoints_team_match_median", "heals_team_to_match_ratio_max", "picker_match_median", "sniper_team_to_match_ratio_mean", "longestKill_team_match_min", "kill_to_team_kills_ratio_match_sum", "longestKill_team_match_std", "roadKills_team_match_mean_rank", "healsAndBoostsPerWalkDistance_team_match_min", "sniper_team_match_mean_rank", "boostsPerWalkDistance_team_match_median", "rideDistance_team_to_match_ratio_mean", "winPoints_team_match_max", "headshot_rate_team_to_match_ratio_mean", "winPoints_match_median", "swimDistance_team_match_sum_rank", "killPoints_team_match_median", "vehicleDestroys_team_match_median_rank", "DBNOs_team_to_match_ratio_max", "killPoints_team_match_mean", "multi_killer_match_sum", "weaponsAcquired_team_match_mean", "healsPerWalkDistance_team_match_min", "heals_team_to_match_ratio_median", "kills_team_to_match_ratio_max", "headshot_rate_team_match_max_rank", "rideDistance_team_match_std", "boostsPerWalkDistance_team_match_min", "killPoints_team_match_sum", "rideDistance_team_to_match_ratio_max", "sniper_team_match_sum_rank", "health_items_match_median", "winPoints_team_match_sum", "revives_team_to_match_ratio_sum", "non_leathal_input_team_to_match_ratio_max", "DBNOs_match_max", "kills_and_assists_team_to_match_ratio_max", "headshot_rate_team_match_sum_rank", "killStreaks_team_match_var", "skill_team_match_sum_rank", "assists_team_to_match_ratio_sum", "revives_team_to_match_ratio_median", "headshotKills_match_max", "headshot_rate_team_match_var", "distance_over_weapons_team_match_std", "rideDistance_team_match_sum", "headshotKills_team_match_max_rank", "winPoints_team_match_mean", "rideDistance_team_to_match_ratio_median", "sniper_team_match_var", "sniper_team_match_median_rank", "boosts_team_to_match_ratio_max", "headshotKills_team_to_match_ratio_mean", "rideDistance_team_match_max", "headshot_rate_team_match_min_rank", "teamKills_team_match_sum_rank", "weaponsAcquired_match_median", "assists_team_to_match_ratio_median", "winPoints_team_match_median", "rideDistance_team_to_match_ratio_sum", "skill_match_max", "picker_team_match_sum", "rideDistance_team_match_mean", "kill_to_team_kills_ratio_team_match_min_rank", "swimDistance_team_to_match_ratio_mean", "assists_match_max", "killStreakrate_team_match_mean", "skill_team_match_max_rank", "teamKills_match_sum", "kill_to_team_kills_ratio_match_max", "sniper_team_to_match_ratio_median", "roadKills_team_match_median_rank", "killStreaks_match_max", "rideDistance_team_match_min", "headshotKills_team_match_sum_rank", "kills_and_assists_match_median", "assists_team_match_var", "weaponsAcquired_team_match_sum", "teamKills_team_to_match_ratio_mean", "rideDistance_team_match_var", "killStreakrate_team_match_sum", "revives_team_match_var", "headshotKills_team_to_match_ratio_sum", "headshot_rate_team_to_match_ratio_max", "headshot_rate_team_to_match_ratio_median", "health_items_team_match_mean", "skill_team_to_match_ratio_mean", "rideDistance_team_match_median", "swimDistance_team_to_match_ratio_sum", "revives_match_max", "health_items_team_match_sum", "kill_to_team_kills_ratio_team_match_sum_rank", "heals_team_match_mean", "rankPoints_team_match_std", "skill_team_to_match_ratio_sum", "headshotKills_team_match_var", "killPlace_team_match_std", "swimDistance_team_match_var", "vehicleDestroys_team_match_min_rank", "swimDistance_team_to_match_ratio_median", "swimDistance_team_to_match_ratio_max", "roadKills_team_match_min_rank", "boosts_match_median", "heals_team_match_sum", "picker_team_match_std", "non_leathal_input_team_match_mean", "heals_match_median", "picker_team_match_max", "boosts_team_match_mean", "swimDistance_team_match_max_rank", "DBNOs_match_median", "headshotKills_team_to_match_ratio_median", "weaponsAcquired_team_match_std", "skill_team_to_match_ratio_median", "killPoints_team_match_std", "DBNOs_team_match_mean", "picker_team_match_median", "sniper_team_match_mean", "sniper_team_match_sum", "swimDistance_team_match_max", "winPoints_team_match_std", "sniper_team_match_min_rank", "headshot_rate_team_match_mean", "swimDistance_team_match_mean", "picker_team_match_min", "health_items_team_match_std", "vehicleDestroys_team_match_sum_rank", "killStreaks_team_to_match_ratio_max", "weaponsAcquired_team_match_median", "multi_killer_team_match_min_rank", "kills_match_median", "swimDistance_team_match_std", "assists_team_to_match_ratio_max", "swimDistance_team_match_sum", "teamKills_team_match_max_rank", "health_items_team_match_max", "healsAndBoostsPerWalkDistance_team_match_std", "weaponsAcquired_team_match_max", "kill_to_team_kills_ratio_team_to_match_ratio_mean", "revives_team_match_mean", "killStreakrate_team_match_median", "non_leathal_input_match_median", "killsPerWalkDistance_team_match_std", "swimDistance_team_match_min", "multi_killer_team_match_sum_rank", "assists_team_match_mean", "heals_team_match_median", "teamKills_match_max", "distance_over_weapons_team_to_match_ratio_min", "headshot_rate_team_match_sum", "multi_killer_match_max", "swimDistance_team_match_median", "health_items_team_match_median", "boosts_team_match_sum", "boosts_team_match_max", "skill_team_match_var", "healsPerWalkDistance_team_match_std", "boostsPerWalkDistance_team_match_std", "kills_team_match_mean", "heals_team_match_max", "killStreakrate_team_match_std", "weaponsAcquired_team_match_min", "roadKills_team_match_sum_rank", "headshotKills_team_to_match_ratio_max", "revives_team_to_match_ratio_max", "heals_team_match_std", "vehicleDestroys_team_to_match_ratio_mean", "non_leathal_input_team_match_std", "health_items_team_match_min", "non_leathal_input_team_match_sum", "kills_without_moving_match_mean", "boosts_team_match_std", "kills_team_match_std", "killStreaks_team_match_mean", "kills_and_assists_team_match_std", "walkDistance_team_to_match_ratio_sum", "teamKills_team_to_match_ratio_median", "DBNOs_team_match_sum", "picker_team_to_match_ratio_min", "DBNOs_team_match_std", "teamKills_team_to_match_ratio_sum", "teamKills_team_match_var", "headshot_rate_team_match_max", "kills_and_assists_team_match_median", "multi_killer_team_to_match_ratio_mean", "distance_over_weapons_match_min", "sniper_team_match_median", "weaponsAcquired_team_to_match_ratio_min", "kills_and_assists_team_match_sum", "kills_team_match_max", "kills_without_moving_team_match_median_rank", "vehicleDestroys_match_sum", "roadKills_team_to_match_ratio_mean", "boosts_team_match_median", "headshot_rate_team_match_median", "kills_team_match_min", "heals_team_match_min", "kills_without_moving_team_match_min_rank", "headshot_rate_team_match_std", "skill_team_to_match_ratio_max", "headshotKills_team_match_mean", "kills_team_match_median", "killStreaks_match_median", "sniper_team_match_std", "non_leathal_input_team_match_max", "kills_team_match_sum", "assists_team_match_sum", "DBNOs_team_match_max", "sniper_team_match_max", "assists_team_match_std", "non_leathal_input_team_match_median", "killStreaks_team_match_std", "kills_and_assists_team_match_max", "teamKills_team_match_mean", "kill_to_team_kills_ratio_team_match_var", "total_distance_team_to_match_ratio_sum", "kill_to_team_kills_ratio_team_match_max_rank", "vehicleDestroys_team_match_max_rank", "roadKills_match_sum", "DBNOs_team_match_median", "killPlace_match_max", "sniper_team_to_match_ratio_max", "kill_to_team_kills_ratio_team_to_match_ratio_sum", "revives_team_match_std", "boosts_team_match_min", "kill_to_team_kills_ratio_team_match_mean", "killStreaks_team_match_sum", "revives_team_match_sum", "roadKills_match_max", "killStreakrate_team_match_max", "kill_to_team_kills_ratio_team_to_match_ratio_median", "non_leathal_input_team_match_min", "skill_team_match_mean", "kills_without_moving_team_match_sum_rank", "walkDistance_match_sum", "kills_and_assists_team_match_min", "headshotKills_team_match_sum", "multi_killer_team_match_max_rank", "sniper_team_match_min", "vehicleDestroys_match_max", "multi_killer_team_match_var", "DBNOs_team_match_min", "headshot_rate_team_match_min", "kills_without_moving_team_match_max_rank", "roadKills_team_match_max_rank", "assists_team_match_max", "vehicleDestroys_team_to_match_ratio_median", "revives_team_match_max", "skill_team_match_sum", "killStreaks_team_match_max", "headshotKills_team_match_max", "killStreakrate_team_to_match_ratio_max", "sniper_match_max", "revives_team_match_median", "skill_team_match_std", "kill_to_team_kills_ratio_team_match_sum", "vehicleDestroys_team_match_var", "headshotKills_team_match_std", "multi_killer_team_to_match_ratio_sum", "assists_team_match_median", "multi_killer_team_to_match_ratio_median", "kill_to_team_kills_ratio_team_to_match_ratio_max", "damageDealt_team_to_match_ratio_min", "headshotKills_team_match_median", "longestKill_team_to_match_ratio_min", "multi_killer_team_match_mean", "assists_team_match_min", "teamKills_team_to_match_ratio_max", "killStreaks_team_match_median", "skill_team_match_max", "roadKills_team_to_match_ratio_median", "non_leathal_input_team_to_match_ratio_min", "vehicleDestroys_team_to_match_ratio_sum", "teamKills_team_match_std", "roadKills_team_match_var", "multi_killer_team_to_match_ratio_max", "killStreakrate_team_to_match_ratio_min", "multi_killer_team_match_sum", "roadKills_team_to_match_ratio_sum", "kill_to_team_kills_ratio_team_match_std", "roadKills_team_match_mean", "vehicleDestroys_team_match_mean", "teamKills_team_match_sum", "kills_and_assists_team_to_match_ratio_min", "sniper_match_median", "kill_to_team_kills_ratio_team_match_max", "weaponsAcquired_match_min", "total_distance_match_sum", "kills_without_moving_match_sum", "killsPerWalkDistance_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_match_median", "killStreaks_team_match_min", "revives_team_match_min", "kills_without_moving_team_match_var", "skill_team_match_median", "picker_match_min", "kills_team_to_match_ratio_min", "headshotKills_team_match_min", "assists_match_median", "killsPerWalkDistance_match_min", "multi_killer_team_match_median", "multi_killer_team_match_std", "skill_team_match_min", "longestKill_match_min", "matchDuration", "headshotKills_match_median", "teamKills_team_match_max", "roadKills_team_match_std", "vehicleDestroys_team_match_sum", "killStreaks_team_to_match_ratio_min", "kills_without_moving_team_match_mean", "vehicleDestroys_team_match_max", "damageDealt_match_min", "roadKills_team_to_match_ratio_max", "killStreakrate_match_min", "assists_team_to_match_ratio_min", "teamKills_team_match_median", "vehicleDestroys_team_match_std", "multi_killer_team_match_max", "skill_match_median", "roadKills_team_match_sum", "kills_without_moving_team_match_sum", "headshot_rate_match_median", "roadKills_team_match_max", "DBNOs_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_match_min", "teamKills_team_match_min", "kills_without_moving_team_match_std", "killStreakrate_match_max", "healsAndBoostsPerWalkDistance_match_min", "kills_match_min", "health_items_team_to_match_ratio_min", "healsAndBoostsPerWalkDistance_team_to_match_ratio_min", "revives_match_median", "rideDistance_match_min", "rideDistance_team_to_match_ratio_min", "multi_killer_team_match_min", "vehicleDestroys_team_to_match_ratio_max", "vehicleDestroys_team_match_median", "health_items_match_min", "killStreaks_match_min", "non_leathal_input_match_min", "roadKills_team_match_median", "swimDistance_match_median", "healsPerWalkDistance_team_to_match_ratio_min", "headshot_rate_match_min", "kills_without_moving_team_match_median", "roadKills_match_median", "teamKills_match_median", "vehicleDestroys_match_median", "kills_without_moving_match_median", "kill_to_team_kills_ratio_match_median", "multi_killer_match_median", "roadKills_team_match_min", "vehicleDestroys_team_match_min", "kills_without_moving_team_match_min", "assists_match_min", "boosts_match_min", "DBNOs_match_min", "headshotKills_match_min", "heals_match_min", "killPlace_match_min", "revives_match_min", "roadKills_match_min", "swimDistance_match_min", "teamKills_match_min", "vehicleDestroys_match_min", "kills_and_assists_match_min", "kills_without_moving_match_min", "boostsPerWalkDistance_match_min", "healsPerWalkDistance_match_min", "skill_match_min", "kill_to_team_kills_ratio_match_min", "multi_killer_match_min", "sniper_match_min", "boosts_team_to_match_ratio_min", "headshotKills_team_to_match_ratio_min", "heals_team_to_match_ratio_min", "revives_team_to_match_ratio_min", "roadKills_team_to_match_ratio_min", "swimDistance_team_to_match_ratio_min", "teamKills_team_to_match_ratio_min", "vehicleDestroys_team_to_match_ratio_min", "boostsPerWalkDistance_team_to_match_ratio_min", "headshot_rate_team_to_match_ratio_min", "skill_team_to_match_ratio_min", "kill_to_team_kills_ratio_team_to_match_ratio_min", "multi_killer_team_to_match_ratio_min", "sniper_team_to_match_ratio_min", "kills_without_moving_team_match_max", "kills_without_moving_match_max", "kill_to_team_kills_ratio_team_match_median_rank", "healsAndBoostsPerWalkDistance_team_to_match_ratio_max", "kills_and_assists_team_to_match_ratio_mean", "longestKill_team_match_max", "health_items_team_match_var", "killPoints_match_median", "healsAndBoostsPerWalkDistance_team_match_min_rank", "DBNOs_team_match_var", "headshot_rate_team_match_median_rank", "boostsPerWalkDistance_team_match_min_rank", "healsPerWalkDistance_team_match_median_rank", "killPoints_team_match_max", "assists_team_match_max_rank", "headshot_rate_team_to_match_ratio_sum", "walkDistance_match_min", "DBNOs_team_to_match_ratio_median", "rideDistance_team_match_max_rank", "non_leathal_input_match_max", "health_items_team_match_max_rank", "winPoints_match_mean", "heals_team_match_sum_rank", "total_distance_match_min", "revives_team_to_match_ratio_mean", "healsAndBoostsPerWalkDistance_team_to_match_ratio_sum", "heals_team_match_max_rank", "damageDealt_team_match_min", "healsPerWalkDistance_team_match_var", "assists_team_to_match_ratio_mean", "healsPerWalkDistance_team_to_match_ratio_max", "boostsPerWalkDistance_team_to_match_ratio_mean", "teamKills_team_match_min_rank", "boostsPerWalkDistance_team_match_var", "boostsPerWalkDistance_team_match_sum", "killStreakrate_team_match_var", "winPoints_team_match_min", "killPoints_team_match_min", "healsAndBoostsPerWalkDistance_team_match_sum_rank", "health_items_team_to_match_ratio_sum", "rideDistance_match_median", "revives_team_match_max_rank", "boostsPerWalkDistance_team_match_max", "killPoints_match_min", "boostsPerWalkDistance_team_match_mean", "boostsPerWalkDistance_team_match_sum_rank", "killStreaks_team_to_match_ratio_median", "killStreaks_team_to_match_ratio_sum", "healsAndBoostsPerWalkDistance_team_match_max", "healsPerWalkDistance_team_to_match_ratio_sum", "killsPerWalkDistance_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_max_rank", "heals_team_to_match_ratio_mean", "heals_team_match_var", "kills_and_assists_team_match_var", "boostsPerWalkDistance_team_match_max_rank", "non_leathal_input_team_to_match_ratio_median", "boosts_team_match_var", "healsPerWalkDistance_team_match_sum", "kills_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_mean", "heals_team_to_match_ratio_sum", "healsAndBoostsPerWalkDistance_team_to_match_ratio_mean", "healsAndBoostsPerWalkDistance_team_match_sum", "multi_killer_team_match_median_rank", "healsPerWalkDistance_team_to_match_ratio_mean", "healsPerWalkDistance_team_match_sum_rank", "healsPerWalkDistance_team_match_mean", "healsPerWalkDistance_team_match_max_rank", "health_items_team_to_match_ratio_max", "boostsPerWalkDistance_team_to_match_ratio_median", "longestKill_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_to_match_ratio_median", "kills_and_assists_team_to_match_ratio_median", "longestKill_team_match_median", "boosts_team_to_match_ratio_median", "healsAndBoostsPerWalkDistance_team_match_median", "killsPerWalkDistance_team_match_median", "health_items_team_to_match_ratio_median", "healsPerWalkDistance_team_to_match_ratio_median", "healsPerWalkDistance_team_match_median", "killPlace_over_maxPlace_team_match_sum_rank", "kills_and_assists_team_match_mean", "killPoints_team_to_match_ratio_mean", "rankPoints_team_match_sum", "health_items_team_match_sum_rank", "damageDealt_team_match_std", "rankPoints_match_median", "killPoints_match_sum", "weaponsAcquired_match_max", "vehicleDestroys_team_match_mean_rank", "damageDealt_team_to_match_ratio_median", "boostsPerWalkDistance_team_to_match_ratio_sum", "boostsPerWalkDistance_team_match_median_rank", "damageDealt_team_match_median", "assists_team_match_sum_rank", "longestKill_team_to_match_ratio_max", "headshot_rate_team_match_mean_rank", "winPoints_team_match_median_rank", "non_leathal_input_match_sum", "killPoints_team_match_sum_rank", "killStreakrate_team_match_median_rank", "winPoints_team_match_sum_rank", "killPoints_team_match_median_rank", "winPoints_team_to_match_ratio_median", "DBNOs_team_to_match_ratio_sum", "killStreaks_team_match_sum_rank", "assists_match_sum", "killPoints_team_match_mean_rank", "healsPerWalkDistance_team_match_min_rank", "boostsPerWalkDistance_team_to_match_ratio_max", "revives_team_match_sum_rank", "roadKills_match_mean", "DBNOs_match_sum", "killPoints_match_mean", "winPoints_team_match_mean_rank", "headshotKills_match_sum", "longestKill_team_match_mean", "killStreaks_team_match_median_rank", "skill_match_sum", "killsPerWalkDistance_team_match_mean", "multi_killer_team_match_mean_rank", "boosts_match_max", "longestKill_team_to_match_ratio_mean", "rankPoints_team_match_max", "longestKill_team_match_sum_rank", "longestKill_team_match_max_rank", "killsPerWalkDistance_team_match_sum", "killsPerWalkDistance_team_match_var", "non_leathal_input_team_match_var", "sniper_team_match_max_rank", "rankPoints_team_match_min", "non_leathal_input_team_to_match_ratio_sum", "revives_match_sum", "sniper_team_to_match_ratio_sum", "kills_and_assists_team_to_match_ratio_sum", "rankPoints_team_match_mean", "killPlace_over_maxPlace_team_match_std", "group_size", "killPlace_over_maxPlace_team_to_match_ratio_sum", "killPlace_over_maxPlace_team_match_min_rank", "killPlace_over_maxPlace_team_match_mean_rank", "kills_without_moving_team_match_mean_rank", "killPlace_team_to_match_ratio_min", "killPlace_match_sum", "killPlace_over_maxPlace_match_min", "heals_match_max", "distance_over_weapons_team_match_sum", "damageDealt_team_match_mean", "boosts_team_match_mean_rank", "revives_team_match_min_rank", "damageDealt_team_to_match_ratio_mean", "weaponsAcquired_team_match_mean_rank", "damageDealt_team_to_match_ratio_sum", "killPoints_team_match_min_rank", "picker_team_to_match_ratio_median", "killPoints_team_to_match_ratio_sum", "total_distance_team_match_min_rank", "non_leathal_input_team_match_max_rank", "kills_match_sum", "rankPoints_team_match_median_rank", "kills_and_assists_team_match_mean_rank", "kill_to_team_kills_ratio_team_match_mean_rank", "distance_over_weapons_team_match_median_rank", "headshotKills_team_match_min_rank", "rankPoints_team_match_mean_rank", "picker_team_match_median_rank", "DBNOs_team_match_max_rank", "damageDealt_team_match_max", "distance_over_weapons_team_match_mean", "distance_over_weapons_team_match_median", "healsAndBoostsPerWalkDistance_team_match_median_rank", "total_distance_team_match_sum", "winPoints_team_to_match_ratio_sum", "non_leathal_input_team_match_mean_rank", "DBNOs_team_match_sum_rank", "non_leathal_input_team_match_sum_rank", "distance_over_weapons_team_match_mean_rank", "total_distance_team_match_max", "kills_and_assists_team_match_sum_rank", "DBNOs_team_match_mean_rank", "picker_team_to_match_ratio_max", "total_distance_team_match_std", "walkDistance_team_to_match_ratio_min", "total_distance_team_match_min", "walkDistance_team_match_std", "teamKills_team_match_median_rank", "non_leathal_input_team_to_match_ratio_mean", "damageDealt_team_match_sum", "killStreakrate_team_to_match_ratio_median", "killStreaks_team_match_mean_rank", "rideDistance_team_match_sum_rank", "killPoints_match_max", "rankPoints_match_mean", "total_distance_team_match_mean", "weaponsAcquired_team_to_match_ratio_max", "winPoints_match_sum", "kills_and_assists_team_match_max_rank", "killPoints_team_match_var", "killPlace_team_match_sum", "killStreakrate_match_median", "rideDistance_match_sum", "winPoints_match_max", "boosts_team_to_match_ratio_sum", "boosts_team_to_match_ratio_mean", "total_distance_team_to_match_ratio_min", "headshot_rate_match_max", "skill_team_match_min_rank", "winPoints_team_match_var", "picker_match_max", "winPoints_match_min", "kills_team_match_sum_rank", "kills_match_max", "kills_team_match_var", "kills_and_assists_match_max", "damageDealtRankWithinKills_team_match_sum", "health_items_team_to_match_ratio_mean", "walkDistance_team_match_median_rank", "killStreaks_team_to_match_ratio_mean", "killStreaks_match_sum", "killPlaceRankWithinKills_team_to_match_ratio_sum", "walkDistanceRankWithinKills_team_match_median", "damageDealtRankWithinKills_team_to_match_ratio_sum", "walkDistanceRankWithinKills_team_to_match_ratio_median", "total_distance_team_match_median_rank", "killsPerWalkDistance_team_match_max", "total_distance_team_to_match_ratio_median", "walkDistanceRankWithinKills_team_match_max", "killsPerWalkDistance_team_to_match_ratio_mean", "walkDistance_team_to_match_ratio_median", "walkDistanceRankWithinKills_team_match_mean", "walkDistanceRankWithinKills_team_to_match_ratio_mean", "kills_team_match_mean_rank", "killPlace_over_maxPlace_team_match_min", "assistsRankWithinKills_team_match_sum", "killsPerWalkDistance_team_to_match_ratio_sum", "killsPerWalkDistance_team_match_mean_rank", "walkDistance_team_match_mean", "healsRankWithinKills_team_match_sum", "walkDistance_team_match_median", "killPlace_over_maxPlace_team_match_median", "healsRankWithinKills_team_to_match_ratio_sum", "boostsRankWithinKills_team_to_match_ratio_sum", "killPlace_over_maxPlace_team_match_sum", "boostsRankWithinKills_team_match_sum", "walkDistanceRankWithinKills_team_match_sum", "killPlace_over_maxPlace_team_match_mean", "walkDistanceRankWithinKills_team_to_match_ratio_sum", "killsPerWalkDistance_team_match_max_rank", "picker_team_match_mean", "killsPerWalkDistance_team_match_median_rank", "killsPerWalkDistance_team_match_sum_rank", "killPlace_team_to_match_ratio_median", "damageDealtRankWithinKills_team_match_std", "killPlace_team_to_match_ratio_mean", "killPlaceRankWithinKills_match_sum", "kills_team_match_max_rank", "kills_team_to_match_ratio_mean", "killPlace_team_match_max", "killPlaceRankWithinKills_match_min", "playersJoined", "walkDistanceRankWithinKills_team_match_std", "killStreaks_team_match_min_rank", "killPlaceRankWithinKills_team_match_std", "healsRankWithinKills_team_match_std", "boostsRankWithinKills_team_match_std", "healsRankWithinKills_team_to_match_ratio_max", "numGroups", "maxPlace", "boostsRankWithinKills_team_to_match_ratio_max", "assistsRankWithinKills_team_match_std", "killPlace_over_maxPlace_team_to_match_ratio_median", "damageDealtRankWithinKills_team_to_match_ratio_max", "killStreakrate_team_match_min_rank", "killPlace_over_maxPlace_team_to_match_ratio_mean", "damageDealtRankWithinKills_match_min", "killsPerWalkDistance_team_match_min_rank", "killsPerWalkDistance_team_match_min", "killPlace_over_maxPlace_team_match_median_rank", "walkDistanceRankWithinKills_team_to_match_ratio_max", "walkDistanceRankWithinKills_match_sum", "assistsRankWithinKills_match_max", "killStreakrate_team_match_min", "healsRankWithinKills_match_sum", "assistsRankWithinKills_match_sum", "healsRankWithinKills_match_max", "boostsRankWithinKills_match_max", "boostsRankWithinKills_match_sum", "damageDealtRankWithinKills_match_sum", "damageDealtRankWithinKills_match_max", "walkDistanceRankWithinKills_match_max", "killPlaceRankWithinKills_match_max", "killPlaceRankWithinKills_10andMore_team_match_mean_rank", "killPlaceRankWithinKills_6_team_match_median_rank", "killPlaceRankWithinKills_6_team_to_match_ratio_sum", "killPlaceRankWithinKills_9_match_sum", "killPlaceRankWithinKills_8_team_match_sum_rank", "killPlaceRankWithinKills_9_team_match_mean_rank", "killPlaceRankWithinKills_1_team_match_sum", "killPlaceRankWithinKills_4_team_match_min_rank", "killPlaceRankWithinKills_6_team_to_match_ratio_mean", "killPlaceRankWithinKills_7_team_to_match_ratio_sum", "killPlaceRankWithinKills_4_team_match_max_rank", "killPlaceRankWithinKills_8_match_mean", "killPlaceRankWithinKills_2_team_match_var", "killPlaceRankWithinKills_9_team_match_sum_rank", "killPlaceRankWithinKills_10andMore_team_match_sum_rank", "killPlaceRankWithinKills_7_team_match_median_rank", "killPlaceRankWithinKills_1_team_match_mean", "killPlaceRankWithinKills_5_team_to_match_ratio_median", "killPlaceRankWithinKills_8_team_to_match_ratio_sum", "killPlaceRankWithinKills_10andMore_team_to_match_ratio_sum", "killPlaceRankWithinKills_1_team_to_match_ratio_max", "killPlaceRankWithinKills_10andMore_match_mean", "killPlaceRankWithinKills_5_team_match_max_rank", "killPlaceRankWithinKills_8_team_match_median_rank", "killPlaceRankWithinKills_2_team_match_sum", "killPlaceRankWithinKills_9_match_mean", "killPlaceRankWithinKills_7_team_to_match_ratio_mean", "killPlaceRankWithinKills_6_team_match_max_rank", "killPlaceRankWithinKills_2_team_match_mean", "killPlaceRankWithinKills_10andMore_team_match_median_rank", "killPlaceRankWithinKills_1_team_match_std", "killPlaceRankWithinKills_6_team_match_min_rank", "killPlaceRankWithinKills_6_team_to_match_ratio_median", "killPlaceRankWithinKills_1_team_match_median", "killPlaceRankWithinKills_7_team_match_max_rank", "killPlaceRankWithinKills_3_team_match_var", "killPlaceRankWithinKills_2_team_to_match_ratio_max", "killPlaceRankWithinKills_8_team_to_match_ratio_mean", "killPlaceRankWithinKills_10andMore_team_match_max_rank", "killPlaceRankWithinKills_3_team_match_max", "killPlaceRankWithinKills_8_team_match_max_rank", "killPlaceRankWithinKills_3_team_match_mean", "killPlaceRankWithinKills_3_team_match_sum", "killPlaceRankWithinKills_3_team_to_match_ratio_max", "killPlaceRankWithinKills_10andMore_team_to_match_ratio_mean", "killPlaceRankWithinKills_10andMore_team_match_min_rank", "killPlaceRankWithinKills_9_team_match_max_rank", "killPlaceRankWithinKills_7_team_to_match_ratio_median", "killPlaceRankWithinKills_9_team_to_match_ratio_mean", "killPlaceRankWithinKills_2_team_match_median", "killPlaceRankWithinKills_4_team_to_match_ratio_max", "killPlaceRankWithinKills_2_team_match_std", "killPlaceRankWithinKills_0_team_to_match_ratio_min", "killPlaceRankWithinKills_1_team_match_min", "killPlaceRankWithinKills_8_team_to_match_ratio_median", "killPlaceRankWithinKills_4_team_match_var", "killPlaceRankWithinKills_4_team_match_mean", "killPlaceRankWithinKills_4_team_match_max", "killPlaceRankWithinKills_10andMore_team_to_match_ratio_median", "killPlaceRankWithinKills_4_team_match_sum", "killPlaceRankWithinKills_5_team_to_match_ratio_max", "killPlaceRankWithinKills_9_team_to_match_ratio_median", "killPlaceRankWithinKills_3_team_match_std", "killPlaceRankWithinKills_3_team_match_median", "killPlaceRankWithinKills_5_team_match_var", "killPlaceRankWithinKills_2_team_match_min", "killPlaceRankWithinKills_5_team_match_sum", "killPlaceRankWithinKills_6_team_to_match_ratio_max", "killPlaceRankWithinKills_5_team_match_mean", "killPlaceRankWithinKills_5_team_match_max", "killPlaceRankWithinKills_6_team_match_var", "killPlaceRankWithinKills_6_team_match_sum", "killPlaceRankWithinKills_7_team_to_match_ratio_max", "killPlaceRankWithinKills_4_team_match_std", "killPlaceRankWithinKills_4_team_match_median", "killPlaceRankWithinKills_6_team_match_mean", "killPlaceRankWithinKills_7_team_match_var", "killPlaceRankWithinKills_6_team_match_max", "killPlaceRankWithinKills_8_team_to_match_ratio_max", "killPlaceRankWithinKills_6_match_max", "killPlaceRankWithinKills_1_team_to_match_ratio_min", "killPlaceRankWithinKills_3_team_match_min", "killPlaceRankWithinKills_7_team_match_sum", "killPlaceRankWithinKills_5_team_match_std", "killPlaceRankWithinKills_5_match_max", "killPlaceRankWithinKills_10andMore_team_match_var", "killPlaceRankWithinKills_7_team_match_mean", "killPlaceRankWithinKills_1_match_median", "killPlaceRankWithinKills_8_team_match_var", "killPlaceRankWithinKills_10andMore_team_to_match_ratio_max", "killPlaceRankWithinKills_9_team_to_match_ratio_max", "killPlaceRankWithinKills_7_match_max", "killPlaceRankWithinKills_10andMore_team_match_sum", "killPlaceRankWithinKills_4_match_max", "killPlaceRankWithinKills_8_team_match_sum", "killPlaceRankWithinKills_5_team_match_median", "killPlaceRankWithinKills_8_match_max", "killPlaceRankWithinKills_9_team_match_var", "killPlaceRankWithinKills_7_team_match_max", "killPlaceRankWithinKills_6_team_match_std", "killPlaceRankWithinKills_8_team_match_mean", "killPlaceRankWithinKills_9_team_match_sum", "killPlaceRankWithinKills_10andMore_team_match_mean", "killPlaceRankWithinKills_6_team_match_median", "killPlaceRankWithinKills_4_team_match_min", "killPlaceRankWithinKills_10andMore_match_max", "killPlaceRankWithinKills_8_team_match_max", "killPlaceRankWithinKills_9_team_match_mean", "killPlaceRankWithinKills_9_match_max", "killPlaceRankWithinKills_7_team_match_std", "killPlaceRankWithinKills_2_team_to_match_ratio_min", "killPlaceRankWithinKills_5_team_match_min", "killPlaceRankWithinKills_8_team_match_std", "killPlaceRankWithinKills_10andMore_team_match_std", "killPlaceRankWithinKills_10andMore_team_match_median", "killPlaceRankWithinKills_9_team_match_max", "killPlaceRankWithinKills_10andMore_match_median", "killPlaceRankWithinKills_10andMore_team_match_max", "killPlaceRankWithinKills_6_team_match_min", "killPlaceRankWithinKills_3_match_max", "killPlaceRankWithinKills_3_team_to_match_ratio_min", "killPlaceRankWithinKills_2_match_max", "killPlaceRankWithinKills_7_team_match_median", "killPlaceRankWithinKills_7_team_match_min", "killPlaceRankWithinKills_8_team_match_min", "killPlaceRankWithinKills_1_match_max", "killPlaceRankWithinKills_8_team_match_median", "killPlaceRankWithinKills_9_team_match_std", "killPlaceRankWithinKills_10andMore_team_match_min", "killPlaceRankWithinKills_2_match_median", "killPlaceRankWithinKills_3_match_median", "killPlaceRankWithinKills_4_match_median", "killPlaceRankWithinKills_5_match_median", "killPlaceRankWithinKills_6_match_median", "killPlaceRankWithinKills_7_match_median", "killPlaceRankWithinKills_8_match_median", "killPlaceRankWithinKills_9_match_median", "killPlaceRankWithinKills_9_team_match_median", "killPlaceRankWithinKills_0_match_min", "killPlaceRankWithinKills_1_match_min", "killPlaceRankWithinKills_2_match_min", "killPlaceRankWithinKills_3_match_min", "killPlaceRankWithinKills_4_match_min", "killPlaceRankWithinKills_5_match_min", "killPlaceRankWithinKills_6_match_min", "killPlaceRankWithinKills_7_match_min", "killPlaceRankWithinKills_8_match_min", "killPlaceRankWithinKills_9_match_min", "killPlaceRankWithinKills_10andMore_match_min", "killPlaceRankWithinKills_9_team_match_min", "killPlaceRankWithinKills_4_team_to_match_ratio_min", "killPlaceRankWithinKills_5_team_to_match_ratio_min", "killPlaceRankWithinKills_6_team_to_match_ratio_min", "killPlaceRankWithinKills_7_team_to_match_ratio_min", "killPlaceRankWithinKills_8_team_to_match_ratio_min", "killPlaceRankWithinKills_9_team_to_match_ratio_min", "killPlaceRankWithinKills_10andMore_team_to_match_ratio_min", "killPlaceRankWithinKills_0_match_max", "killPlace_over_maxPlace_team_match_max_rank", "killPlace_over_maxPlace_team_to_match_ratio_min", "killPlace_over_maxPlace_team_to_match_ratio_max", "killPlaceRankWithinKills_0_team_to_match_ratio_max")
#v63
                    # , "killPlaceRankWithinKills_7_team_match_min_rank", "killPlaceRankWithinKills_8_team_match_min_rank", "killPlaceRankWithinKills_9_team_match_min_rank")

def remove_unwanted_features(df):
    for c in df.columns:
        if c in UNWANTED_FEATURES:
            df.drop(labels=c, axis=1, inplace=True)
            
    return df
            
def reduce_columns(columns_list, postfix):
    
    columns_list_copy = columns_list.copy()
    
    for c in columns_list:
        if str(c + postfix) in UNWANTED_FEATURES:
            columns_list_copy.remove(c)
            
    return columns_list_copy

def get_team_extract(data, dropDuplicates):
    """
    Produces for the given data DataFrame statistics of team.
    :param data: Pandas DataFrame with data.
    :param dropDuplicates: Whether to drop duplicate rows (dropping is good for learning only).
    :return Team data that can not be joined with the original data (rows do not contain "Id" column).
    """
    
    if "winPlacePerc" in data.columns:
        new_data = data[["matchId","groupId", "matchDuration", "maxPlace", "numGroups", "playersJoined", "average_team_size", "missing_groups_perc", "winPlacePerc"]]
    else:
        new_data = data[["matchId","groupId", "matchDuration", "maxPlace", "numGroups", "playersJoined", "average_team_size", "missing_groups_perc"]]
        
    new_data.drop(labels="missing_groups_perc", axis=1, inplace=True)
    # new_data["max_place"] = new_data["maxPlace"]
    # new_data["num_groups"] = new_data["numGroups"]
    
    columns_to_process = list(data.columns)
    columns_to_process.remove("Id")
    columns_to_process.remove("groupId")
    columns_to_process.remove("matchId")
    columns_to_process.remove("matchType")
    columns_to_process.remove("matchDuration")
    columns_to_process.remove("maxPlace")
    columns_to_process.remove("numGroups")
    columns_to_process.remove("playersJoined")
    columns_to_process.remove("average_team_size")
    columns_to_process.remove("missing_groups_perc")
    if "winPlacePerc" in data.columns:
        columns_to_process.remove("winPlacePerc")
    
    new_data = new_data.merge(
        data.groupby(["matchId", "groupId"])["total_distance"].sum().to_frame("team_total_distance").reset_index(), 
        on=["matchId", "groupId"], how="left")
    
    new_data["team_total_distance_per_player"] = new_data["team_total_distance"] / new_data["playersJoined"] 
    
    data = data.groupby(["matchId", "groupId"])[columns_to_process]
    
    # if not ("group_size_to_match_average" in UNWANTED_FEATURES and "group_size" in UNWANTED_FEATURES):
    new_agg_data = data.size().reset_index(name="group_size")
    new_agg_data = reduce_mem_usage(new_agg_data)
    new_data = new_data.merge(new_agg_data, how='left', on=["matchId", "groupId"])
    del new_agg_data
    new_agg_data = None
    
    # new_data["group_size_to_match_size"] = new_data["group_size"] / new_data["playersJoined"]
    # new_data["group_size_to_match_size_rank"] = new_data.groupby(["matchId"])["group_size_to_match_size"].rank(ascending=False, pct=True)

    gc.collect()
    if not "group_size_to_match_average" in UNWANTED_FEATURES:
        new_data["group_size_to_match_average"] = new_data["group_size"] / new_data["average_team_size"]
    if "group_size" in UNWANTED_FEATURES and "group_size" in new_data.columns:
        new_data.drop(labels=["group_size"], axis=1, inplace=True)
    
    gc.collect()
    
#     new_data.drop(labels=["average_team_size"], axis=1, inplace=True)
    #this will be kind of match type
#     new_data.loc[(new_data.average_team_size > 1) & (new_data.average_team_size <= 2), "average_team_size"] = 2
#     new_data.loc[new_data.average_team_size > 2, "average_team_size"] = 3
    
    print("mean")
    
    new_agg_data = data.agg("mean")
    new_agg_data = reduce_mem_usage(new_agg_data)
    
    new_agg_data_match = new_agg_data.groupby(["matchId"])[columns_to_process].agg("mean")
    new_agg_data_match = reduce_mem_usage(new_agg_data_match)
    new_agg_data_match = new_agg_data_match.add_suffix("_match_mean")
    new_data = new_data.merge(new_agg_data_match.reset_index(), how="left", on=["matchId"])
    del new_agg_data_match
    new_agg_data_match = None
    gc.collect()
    
    new_agg_data_rank = new_agg_data.groupby('matchId')[reduce_columns(columns_to_process, "_team_match_mean_rank")].rank(pct=True)
    new_agg_data_rank = reduce_mem_usage(new_agg_data_rank)
    new_agg_data_rank = new_agg_data_rank.add_suffix("_team_match_mean_rank")
    new_data = new_data.merge(new_agg_data_rank.reset_index(), how="left", on=["matchId", "groupId"])
    del new_agg_data_rank
    new_agg_data_rank = None
    gc.collect()
        
    new_agg_data = new_agg_data.add_suffix("_team_match_mean")
    new_data = new_data.merge(new_agg_data.reset_index(), how="left", on=["matchId", "groupId"])
    del new_agg_data
    new_agg_data = None
    gc.collect()
    
    for c in reduce_columns(columns_to_process, "_team_to_match_ratio_mean"):
        if c != "kills_without_moving":
            new_data[c + "_team_to_match_ratio_mean"] = new_data[c + "_team_match_mean"] / new_data[c + "_match_mean"]
            
#         new_data.drop(labels=[c + "_match_mean"], axis=1, inplace=True)
    
    new_data = reduce_mem_usage(new_data)
    gc.collect()
    
    
    print("median")
    
    new_agg_data = data.agg("median")
    new_agg_data = reduce_mem_usage(new_agg_data)
    
    new_agg_data_match = new_agg_data.groupby(["matchId"])[columns_to_process].agg("median")
    new_agg_data_match = reduce_mem_usage(new_agg_data_match)
    new_agg_data_match = new_agg_data_match.add_suffix("_match_median")
    new_data = new_data.merge(new_agg_data_match.reset_index(), how="left", on=["matchId"])
    del new_agg_data_match
    new_agg_data_match = None
    gc.collect()
    
    new_agg_data_rank = new_agg_data.groupby('matchId')[reduce_columns(columns_to_process, "_team_match_median_rank")].rank(pct=True)
    new_agg_data_rank = reduce_mem_usage(new_agg_data_rank)
    new_agg_data_rank = new_agg_data_rank.add_suffix("_team_match_median_rank")
    new_data = new_data.merge(new_agg_data_rank.reset_index(), how="left", on=["matchId", "groupId"])
    del new_agg_data_rank
    new_agg_data_rank = None
    gc.collect() 
        
    new_agg_data = new_agg_data.add_suffix("_team_match_median")
    new_data = new_data.merge(new_agg_data.reset_index(), how="left", on=["matchId", "groupId"])
    del new_agg_data
    new_agg_data = None
    gc.collect()
    
    for c in reduce_columns(columns_to_process, "_team_to_match_ratio_median"):
        if c != "kills_without_moving":
            new_data[c + "_team_to_match_ratio_median"] = new_data[c + "_team_match_median"] / new_data[c + "_match_mean"]
#             
#             new_data.drop(labels=[c + "_team_match_median"], axis=1, inplace=True)
            
#         new_data.drop(labels=[c + "_match_median"], axis=1, inplace=True)
    
    remove_unwanted_features(new_data)
    new_data = reduce_mem_usage(new_data)
    gc.collect()
    
    
    print("var")
    new_agg_data = data.agg("var")
    new_agg_data = reduce_mem_usage(new_agg_data)
    new_agg_data = new_agg_data.add_suffix("_team_match_var")
    remove_unwanted_features(new_agg_data) 
    new_data = new_data.merge(new_agg_data.reset_index(), how="left", on=["matchId", "groupId"])
    del new_agg_data
    new_agg_data = None
    gc.collect()
    
    remove_unwanted_features(new_data)
    new_data = reduce_mem_usage(new_data)
    gc.collect()
    
    
    
    print("sum")
    
    new_agg_data = data.agg("sum")
    new_agg_data = reduce_mem_usage(new_agg_data)
    
    new_agg_data_match = new_agg_data.groupby(["matchId"])[columns_to_process].agg("sum")
    new_agg_data_match = reduce_mem_usage(new_agg_data_match)
    new_agg_data_match = new_agg_data_match.add_suffix("_match_sum")
    new_data = new_data.merge(new_agg_data_match.reset_index(), how="left", on=["matchId"])
    del new_agg_data_match
    new_agg_data_match = None
    gc.collect()
    
    new_agg_data_rank = new_agg_data.groupby('matchId')[reduce_columns(columns_to_process, "_team_match_sum_rank")].rank(pct=True)
    new_agg_data_rank = reduce_mem_usage(new_agg_data_rank)
    new_agg_data_rank = new_agg_data_rank.add_suffix("_team_match_sum_rank")
    new_data = new_data.merge(new_agg_data_rank.reset_index(), how="left", on=["matchId", "groupId"])
    del new_agg_data_rank
    new_agg_data_rank = None
    gc.collect()
    
    new_agg_data = new_agg_data.add_suffix("_team_match_sum")
    new_data = new_data.merge(new_agg_data.reset_index(), how="left", on=["matchId", "groupId"])
    del new_agg_data
    new_agg_data = None
    gc.collect()
    
    for c in reduce_columns(columns_to_process, "_team_to_match_ratio_sum"):
        if c != "kills_without_moving":
            new_data[c + "_team_to_match_ratio_sum"] = new_data[c + "_team_match_sum"] / new_data[c + "_match_sum"]
        
#         new_data.drop(labels=[c + "_match_sum"], axis=1, inplace=True)
        
    remove_unwanted_features(new_data)
    new_data = reduce_mem_usage(new_data)
    gc.collect()
    
    
    print("min")
    
    new_agg_data = data.agg("min")
    new_agg_data = reduce_mem_usage(new_agg_data)
    
    new_agg_data_match = new_agg_data.groupby(["matchId"])[columns_to_process].agg("min")
    new_agg_data_match = reduce_mem_usage(new_agg_data_match)
    new_agg_data_match = new_agg_data_match.add_suffix("_match_min")
    new_data = new_data.merge(new_agg_data_match.reset_index(), how="left", on=["matchId"])
    del new_agg_data_match
    new_agg_data_match = None
    gc.collect()
    
    new_agg_data_rank = new_agg_data.groupby('matchId')[reduce_columns(columns_to_process, "_team_match_min_rank")].rank(pct=True)
    new_agg_data_rank = reduce_mem_usage(new_agg_data_rank)
    new_agg_data_rank = new_agg_data_rank.add_suffix("_team_match_min_rank")
    new_data = new_data.merge(new_agg_data_rank.reset_index(), how="left", on=["matchId", "groupId"])
    del new_agg_data_rank
    new_agg_data_rank = None
    gc.collect()
    
    new_agg_data = new_agg_data.add_suffix("_team_match_min")
    new_data = new_data.merge(new_agg_data.reset_index(), how="left", on=["matchId", "groupId"])
    del new_agg_data
    new_agg_data = None
    gc.collect()
    
    for c in reduce_columns(columns_to_process, "_team_to_match_ratio_min"):
        if c != "kills_without_moving":
            new_data[c + "_team_to_match_ratio_min"] = new_data[c + "_team_match_min"] / new_data[c + "_match_min"]
            
#             new_data.drop(labels=[c + "_team_match_min"], axis=1, inplace=True)
            
#         new_data.drop(labels=[c + "_match_min"], axis=1, inplace=True)
    
    remove_unwanted_features(new_data)
    new_data = reduce_mem_usage(new_data)
    gc.collect()
    
    
    print("max")
    
    new_agg_data = data.agg("max")
    new_agg_data = reduce_mem_usage(new_agg_data)
    
    new_agg_data_match = new_agg_data.groupby(["matchId"])[columns_to_process].agg("max")
    new_agg_data_match = reduce_mem_usage(new_agg_data_match)
    new_agg_data_match = new_agg_data_match.add_suffix("_match_max")
    new_data = new_data.merge(new_agg_data_match.reset_index(), how="left", on=["matchId"])
    del new_agg_data_match
    new_agg_data_match = None
    gc.collect()
    
    new_agg_data_rank = new_agg_data.groupby('matchId')[reduce_columns(columns_to_process, "_team_match_max_rank")].rank(pct=True)
    new_agg_data_rank = reduce_mem_usage(new_agg_data_rank)
    new_agg_data_rank = new_agg_data_rank.add_suffix("_team_match_max_rank")
    new_data = new_data.merge(new_agg_data_rank.reset_index(), how="left", on=["matchId", "groupId"])
    del new_agg_data_rank
    new_agg_data_rank = None
    gc.collect()
    
    new_agg_data = new_agg_data.add_suffix("_team_match_max")
    new_data = new_data.merge(new_agg_data.reset_index(), how="left", on=["matchId", "groupId"])
    del new_agg_data
    new_agg_data = None
    gc.collect()
    
    for c in reduce_columns(columns_to_process, "_team_to_match_ratio_max"):
        if c != "kills_without_moving":
            new_data[c + "_team_to_match_ratio_max"] = new_data[c + "_team_match_max"] / new_data[c + "_match_max"]
                        
#             new_data.drop(labels=[c + "_team_match_max"], axis=1, inplace=True)
            
#         new_data.drop(labels=[c + "_match_max"], axis=1, inplace=True)
    
    remove_unwanted_features(new_data)
    new_data = reduce_mem_usage(new_data)
    gc.collect()
    
    
    print("std")
    
    new_agg_data = data.agg("std")
    new_agg_data = reduce_mem_usage(new_agg_data)
    new_agg_data = new_agg_data.add_suffix("_team_match_std")
    new_agg_data.fillna(0, inplace=True)
    remove_unwanted_features(new_agg_data)
    gc.collect()
    new_data = new_data.merge(new_agg_data.reset_index(), how="left", on=["matchId", "groupId"])
    del new_agg_data
    new_agg_data = None
    
    # scaling
    # for c in new_data.columns:
    #     # if c.endswith("_rank"):
    #     #     new_data["scaled"] = new_data[c] * new_data["max_place"] / (new_data["max_place"] - 1)
    #     #     new_data.loc[new_data["scaled"] > 1, "scaled"] = 1
    #     #     new_data.loc[new_data["missing_groups_perc"] > 0.05, c] = new_data.loc[new_data["missing_groups_perc"] > 0.05]["scaled"]
    #     #     new_data.drop(labels="scaled", axis=1, inplace=True)
            
    #     if c.endswith("_match_sum") and not c.endswith("_team_match_sum"):
    #         new_data["scaled"] = new_data[c] * new_data["max_place"] / new_data["num_groups"]
    #         new_data.loc[new_data["missing_groups_perc"] > 0.05, c] = new_data.loc[new_data["missing_groups_perc"] > 0.05]["scaled"]
    #         new_data.drop(labels="scaled", axis=1, inplace=True)
            
    #     if c.endswith("_team_to_match_ratio_sum"):
    #         new_data["scaled"] = new_data[c] / new_data["max_place"] * new_data["num_groups"]
    #         new_data.loc[new_data["missing_groups_perc"] > 0.05, c] = new_data.loc[new_data["missing_groups_perc"] > 0.05]["scaled"]
    #         new_data.drop(labels="scaled", axis=1, inplace=True)
    
    # new_data.drop(labels="missing_groups_perc", axis=1, inplace=True)
    # new_data.drop(labels="max_place", axis=1, inplace=True)
    # new_data.drop(labels="num_groups", axis=1, inplace=True)
    
    gc.collect()
    
    if dropDuplicates:
        new_data.drop_duplicates(subset=None, keep='first', inplace=True)
    
    # final cleanup
    new_data.reset_index(inplace=True, drop=True)
    remove_unwanted_features(new_data)
    new_data=reduce_mem_usage(new_data)
    gc.collect()
    
    return new_data


# def RFR_prediction(x_train, y_train, x_test, y_test, output_file_name="rfp_prediction.csv", store_prediction_to_file=True, draw_graph=True, store_predictor_to_file=True, show_feature_importance=False):
#     """
#     Does prediction with RFR.
#     :param x_train: X for training
#     :param y_train: Y for training
#     :param x_test: X for testing
#     :param y_test: Y for testing
#     :param output_file_name: Prediction results will be stored in the given file.
#     :return Prediction accuracy computed with MAE.
#     """
    
    
#     print()
#     print("Building RFR model")
    
#     #results are a little random, so maybe run k times and store the best result?
    
#     # these are parameters before final tuning - feature selection has been made with these
#     parameters = {'bootstrap': True,
#               'min_samples_leaf': 3,
#               'n_estimators': 100, 
#               'min_samples_split': 5,
#               'max_features': 'sqrt',
#               'max_depth': None,
#               'max_leaf_nodes': None, 
#               'oob_score': False,
#               'verbose': 0, # 2
#               'n_jobs': 7} # TODO -1
    
#     RF_model = RandomForestRegressor(**parameters)
      
#     print()
#     print("Running learning...")
#     RF_model.fit(x_train, y_train)
    
#     #RF_model = pickle.load(open("rfr_score_0.0_train_data_size_700.pickle", "rb"))
    
#     print()
#     print("Predicting...")
#     y_pred = RF_model.predict(x_test)
#     y_pred = np.around(y_pred, decimals=4)
    
#     print()
#     print("Computing score...")
#     rt_score = mean_absolute_error(y_test, y_pred)
#     print("Random forest regressor score: " + str(rt_score));
    
#     importances = RF_model.feature_importances_
#     indices = np.argsort(importances)[::-1]
    
#     if show_feature_importance:
#         print()
#         print("Feature ranking:")
        
#         for f in range(x_train.shape[1]):
#             print("%d. feature %d %s (%f)" % (f + 1, indices[f], x_train.columns[f], importances[indices[f]]))
        
#         print()        
    
#     if store_predictor_to_file:
#         #dump(RF_model, "rfr_score_" + str(rt_score) + ".joblib")
#         pickle.dump(RF_model, open("rfr_score_" + str(rt_score) + "_train_data_size_" + str(len(x_train.index)) + ".pickle", "wb"))
    
#     if store_prediction_to_file:
#         print()
#         print("Storing prediction to file...")
#         rt_prediction = pd.DataFrame(y_pred)
#         rt_prediction.to_csv(output_file_name, sep=',')
    
#     if draw_graph:
#         print()
#         print("Drawing predictions vs target graph...")
#         plt.subplots(figsize=(16,12))
#         data2 = pd.DataFrame({"Target" : y_test.values, "Prediction" : pd.Series(y_pred)})
#         plot = sns.scatterplot(x="Target", y="Prediction", data=data2)
#         plot.get_figure().savefig(GRAPHS_FOLDER + "rfr_target_vs_results.png")
#         plt.clf()

#     return rt_score

# def brute_force_feature_eliminator(x_train, y_train, x_test, y_test):
#     print()
#     print()
#     print("Running feature elimination")
#     print()
#     print("Columns to optimise: " + str(len(x_train.columns)))
  
#     summed_score = 0
#     tries = 3
    
#     for i in range(1, tries + 1):
#         print("Starting run " + str(i) + " / " + str(tries))
#         score = RFR_prediction(x_train, y_train, x_test, y_test, store_prediction_to_file=False, draw_graph=False, store_to_file=False)
#         print("Finished run " + str(i) + " / " + str(tries))
#         print("Score: " + str(score))
#         summed_score = summed_score + score
    
#     average_score = summed_score / tries
#     print("Initial average score for all columns: " + str(average_score)) 

#     continue_search = True
    
#     iteration_number = 1
        
#     best_score = average_score
    
#     removed_columns = list()
    
#     while continue_search:
        
#         candidates = list(x_train.columns)
#         shuffle(candidates)
        
#         scores = dict()
        
#         best_column = None
#         column_number = 1
        
#         for column in candidates:
#             print("Iteration: " + str(iteration_number) + ", column: " + str(column_number) + " / " + str(len(candidates)))
#             print("Testing with removing " + column)
            
#             summed_score = 0
            
#             for i in range(1, tries + 1):
#                 print("Starting run " + str(i) + " / " + str(tries))
#                 score = RFR_prediction(x_train.drop(labels=column, axis=1, inplace=False), y_train, x_test.drop(labels=column, axis=1, inplace=False), y_test, store_prediction_to_file=False, draw_graph=False, store_to_file=False)
#                 print("Finished run " + str(i) + " / " + str(tries))
#                 print("Score: " + str(score))
#                 summed_score = summed_score + score
            
#             average_score = summed_score / tries
#             print("Average score without " + str(column) + ": " + str(average_score))
#             scores[column] = average_score
            
#             column_number = column_number + 1
        
#             if average_score <= best_score:
#                 best_score = score
#                 best_column = column
#                 print("It was better or as good as the previous best score")
#                 break
        
#         if best_column != None:
#             print()
#             print("Selected column '" + str(best_column) + "' to be dropped - score without it: " + str(best_score))
#             removed_columns.append(best_column)
#             x_train.drop(labels=best_column, axis=1, inplace=True)
#             x_test.drop(labels=best_column, axis=1, inplace=True)
#         else:
#             print("No improvement found...")
#             continue_search = False
        
#         iteration_number = iteration_number + 1
        
#     print()
#     print("Columns after optimisation: ")
#     print("Remaining - size: " + str(len(x_train.columns)))
#     print("Remaining - list:" + str(x_train.columns))
#     for column in x_train.columns:
#         print("Remains: " + str(column))
#     print("Removed - size: " + str(len(removed_columns)))
#     print("Removed - list:" + str(removed_columns))
#     for column in removed_columns:
#         print("Removed: " + str(column))
    
    
# def RFECV(data, x_train, target):
#     print("Running RFECV feature eliminator")
    
#     from sklearn.metrics.scorer import r2_scorer
#     from sklearn.feature_selection import RFECV
        
#     # Create a model for feature selection
#     parameters = {'bootstrap': True,
#               'min_samples_leaf': 3,
#               'n_estimators': 100, 
#               'min_samples_split': 5,
#               'max_features': 'sqrt',
#               'max_depth': None,
#               'max_leaf_nodes': None, 
#               'oob_score': False,
#               'verbose': 0,
#               'n_jobs': 6}
#     estimator = RandomForestRegressor(**parameters)
    
#     # Create the object
    
#     selector = RFECV(estimator, step = 5, cv = 3, 
#                      scoring= r2_scorer,
#                       n_jobs = 6, verbose=2)
    
#     # Fit on training data
#     selector.fit(data, target)
    
#     # Transform data
#     selected_columns_indices = selector.get_support(indices=True)
    
#     print("Selected features count: " + str(len(selected_columns_indices)))
    
#     for column_index in selected_columns_indices:
#         print(str(x_train.columns[column_index]))


def features_engineering(data, learnMode=True):
    print("Features engineering...")
    
    #outliers
    data.loc[data.DBNOs > 40, "DBNOs"] = 40
    data.loc[data.headshotKills > 47, "headshotKills"] = 47
    data.loc[data.roadKills > 14, "roadKills"] = 14
    data.loc[data.swimDistance > 2750, "swimDistance"] = 2750
    data.loc[data.walkDistance > 17500, "walkDistance"] = 17500
    data.loc[data.weaponsAcquired > 150, "weaponsAcquired"] = 150
    
    
    
    # https://www.kaggle.com/deffro/eda-is-fun
    data['playersJoined'] = data.groupby('matchId')['matchId'].transform('count')
#     data['kills'] = data['kills']*((100-data['playersJoined'])/100 + 1)
#     data['damageDealt'] = data['damageDealt']*((100-data['playersJoined'])/100 + 1)
#     #added own other normed values
#     data['headshotKills'] = data['headshotKills']*((100-data['playersJoined'])/100 + 1)    
#     data['DBNOs'] = data['DBNOs']*((100-data['playersJoined'])/100 + 1)
#     data['assists'] = data['assists']*((100-data['playersJoined'])/100 + 1)
#     data['revives'] = data['revives']*((100-data['playersJoined'])/100 + 1)
#     data['roadKills'] = data['roadKills']*((100-data['playersJoined'])/100 + 1)
#     data['teamKills'] = data['teamKills']*((100-data['playersJoined'])/100 + 1)
#     data['vehicleDestroys'] = data['vehicleDestroys']*((100-data['playersJoined'])/100 + 1)
    
    # add new feature from kaggle, eg as https://www.kaggle.com/nitinaggarwal008/rf-output-feature-eng/notebook
    data['total_distance'] = data['rideDistance'] + data['walkDistance'] + data['swimDistance']
    data['kills_and_assists'] = (data['kills'] + data['assists'])
    data['kills_without_moving'] = ((data['kills'] > 0) & (data['total_distance'] == 0))
    data['health_items'] = data['heals'] + data['boosts']
    # https://www.kaggle.com/chocozzz/pubg-data-description-a-to-z-fe-with-python
    data["killStreakrate"] = data["killStreaks"]/data["kills"]
     

# TODO te dwa to nie byly uzywane    
    data['boostsPerWalkDistance'] = data['boosts']/(data['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where boosts>0 and walkDistance=0. Strange.
    data['boostsPerWalkDistance'].fillna(0, inplace=True)
    data['healsPerWalkDistance'] = data['heals']/(data['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where heals>0 and walkDistance=0. Strange.
    data['healsPerWalkDistance'].fillna(0, inplace=True)
    data['healsAndBoostsPerWalkDistance'] = data['health_items']/(data['walkDistance']+1) #The +1 is to avoid infinity.
    data['healsAndBoostsPerWalkDistance'].fillna(0, inplace=True)
    #https://www.kaggle.com/chocozzz/lightgbm-baseline
    data['killPlace_over_maxPlace'] = data['killPlace'] / data['maxPlace']
    data['headshot_rate'] = data['kills'] / data['headshotKills'] # we have sniper, but some use both sniper and 1/sniper
    data['distance_over_weapons'] = data['total_distance'] / data['weaponsAcquired']
    data['killsPerWalkDistance'] = data['kills'] / (data['walkDistance']+1)
    data["skill"] = data["headshotKills"] + data["roadKills"]
    
    # own extra features
    data["killPlaceRankWithinKills"] = data.groupby(["matchId", "kills"])["killPlace"].rank(ascending=False, pct=True)
    for k in range(10):
        data["killPlaceRankWithinKills_" + str(k)] = -10;
        data.loc[data["kills"] == k, "killPlaceRankWithinKills_" + str(k)] = data.loc[data["kills"] == k]["killPlaceRankWithinKills"]
    data["killPlaceRankWithinKills_10andMore"] = -10;
    data.loc[data["kills"] >= 10, "killPlaceRankWithinKills_10andMore"] = data.loc[data["kills"] >= 10]["killPlaceRankWithinKills"]
    data.drop(labels="killPlaceRankWithinKills", axis=1, inplace=True)
    
    data["walkDistanceRankWithinKills"] = data.groupby(["matchId", "kills"])["walkDistance"].rank(ascending=False, pct=True)
    data["healsRankWithinKills"] = data.groupby(["matchId", "kills"])["heals"].rank(ascending=False, pct=True)
    data["boostsRankWithinKills"] = data.groupby(["matchId", "kills"])["boosts"].rank(ascending=False, pct=True)
    data["damageDealtRankWithinKills"] = data.groupby(["matchId", "kills"])["damageDealt"].rank(ascending=False, pct=True)
    data["assistsRankWithinKills"] = data.groupby(["matchId", "kills"])["assists"].rank(ascending=False, pct=True)
    data["average_team_size"] = data["playersJoined"] / data["numGroups"]
    data["picker"] = data["boosts"] + data["weaponsAcquired"]
    data["kill_to_team_kills_ratio"] = data["kills"] / data["teamKills"]
    data.fillna(0, inplace=True)
    data["multi_killer"] = data["vehicleDestroys"] / data["kills"]
    data.fillna(0, inplace=True)
    data["non_leathal_input"] = data["DBNOs"] + data["assists"]
    data["sniper"] = data["headshotKills"] / data["kills"]
    data.fillna(0, inplace=True)
    
    data.loc[data.matchDuration <= 1660, "matchDuration"] = 0
    data.loc[data.matchDuration > 1660, "matchDuration"] = 1
    
    data["missing_groups_perc"] = (data["maxPlace"] - data["numGroups"]) / data["maxPlace"]
    
#     data.loc[data.assists < 7, "assists"] = 7
#     data.loc[data.boosts < 4, "boosts"] = 4
#     data.loc[data.damageDealt < 500, "damageDealt"] = 500
#     data.loc[data.DBNOs < 4, "DBNOs"] = 4
#     data.loc[data.headshotKills < 3, "headshotKills"] = 3
#     data.loc[data.heals < 5, "heals"] = 5
#     data.loc[data.killPlace < 65, "killPlace"] = 65
#     data.loc[data.killPoints == 0, "killPoints"] = 900
#     data.loc[data.killPoints > 900, "killPoints"] = 900
#     data.loc[data.kills < 4, "kills"] = 4
#     data.loc[data.killStreaks < 5, "killStreaks"] = 5
#     data.loc[data.longestKill < 180, "longestKill"] = 180
#     data.loc[(data.rankPoints >= 1000) & (data.rankPoints <= 3000), "rankPoints"] = 0
#     data.loc[data.revives < 6, "revives"] = 6
#     data.loc[data.rideDistance < 12500, "rideDistance"] = 12500
#     data.loc[data.roadKills < 2, "roadKills"] = 2
#     data.loc[data.swimDistance < 300, "swimDistance"] = 300
#     data.loc[data.teamKills < 4, "teamKills"] = 4
#     data.loc[data.vehicleDestroys < 2, "vehicleDestroys"] = 2
#     data.loc[data.walkDistance < 2500, "walkDistance"] = 2500
#     data.loc[data.weaponsAcquired < 50, "weaponsAcquired"] = 50
#     data.loc[data.winPoints > 1250, "winPoints"] = 0
    
    
    # dropping normalised parameters
#     data.drop(labels=["kills"], axis=1, inplace=True)
#     data.drop(labels=["health_items"], axis=1, inplace=True)
#     data.drop(labels=["kills_and_assists"], axis=1, inplace=True)
#     data.drop(labels=["headshotKills"], axis=1, inplace=True)
#     data.drop(labels=["DBNOs"], axis=1, inplace=True)
#     data.drop(labels=["assists"], axis=1, inplace=True)
#     data.drop(labels=["revives"], axis=1, inplace=True)
#     data.drop(labels=["roadKills"], axis=1, inplace=True)
#     data.drop(labels=["teamKills"], axis=1, inplace=True)
#     data.drop(labels=["vehicleDestroys"], axis=1, inplace=True)
#     data.drop(labels=["playersJoined"], axis=1, inplace=True)
    
    # dropping superfluous features
#     data.drop(labels=["boosts"], axis=1, inplace=True)
#     data.drop(labels=["weaponsAcquired"], axis=1, inplace=True)
#     data.drop(labels=["DBNOs"], axis=1, inplace=True)
#     data.drop(labels=["assists"], axis=1, inplace=True)
#     data.drop(labels=["kills"], axis=1, inplace=True)
#     data.drop(labels=["assists"], axis=1, inplace=True)

    gc.collect()
    
#     print()
#     print("Generating heatmap...")
#     print()
#     corrmat = data.corr()
#     plt.subplots(figsize=(24,20))
#     sns.set(font_scale=0.7)
#     hm = sns.heatmap(corrmat, annot=True)
#     hm.get_figure().savefig(GRAPHS_FOLDER + "heatmap.png")
#     plt.clf()
     
#     player_columns = list(data.columns)
#     player_columns.remove("Id")
#     player_columns.remove("groupId")
#     player_columns.remove("matchId")
#     player_columns.remove("matchType")
#     #player_columns.remove("winPlacePerc")
#     player_columns.remove("kills_without_moving")
    
    print("Extending data with team's extract...")
    data = get_team_extract(data, dropDuplicates=learnMode)
    
#     print("Removing data not matching test")
#     rowsBefore = len(data)
#     print("rows before: " + str(rowsBefore))
#     print("dropping boosts > 24")
#     data.drop(data[data["boosts_team_match_max"] > 24].index, inplace=True)
#     print("rows after: " + str(len(data)))
#     print("dropping damageDealt > 6229")
#     data.drop(data[data["damageDealt_team_match_max"] > 6229].index, inplace=True)
#     print("rows after: " + str(len(data)))
#     print("dropping headshotKills > 41")
#     data.drop(data[data["headshotKills_team_match_max"] > 41].index, inplace=True)
#     print("rows after: " + str(len(data)))
#     print("dropping kills > 58")
#     data.drop(data[data["kills_team_match_max"] > 58].index, inplace=True)
#     print("rows after: " + str(len(data)))
#     print("dropping killStreaks > 15")
#     data.drop(data[data["killStreaks_team_match_max"] > 15].index, inplace=True)
#     print("rows after: " + str(len(data)))
#     print("dropping longestKill > 1004")
#     data.drop(data[data["longestKill_team_match_max"] > 1004].index, inplace=True)
#     print("rows after: " + str(len(data)))
#     print("dropping rankPoints > 5742")
#     data.drop(data[data["rankPoints_team_match_max"] > 5742].index, inplace=True)
#     print("rows after: " + str(len(data)))
#     print("dropping revives > 20")
#     data.drop(data[data["revives_team_match_max"] > 20].index, inplace=True)
#     print("rows after: " + str(len(data)))
#     print("dropping roadKills > 15")
#     data.drop(data[data["roadKills_team_match_max"] > 15].index, inplace=True)
#     print("rows after: " + str(len(data)))
#     print("dropping swimDistance > 3271")
#     data.drop(data[data["swimDistance_team_match_max"] > 3271].index, inplace=True)
#     print("rows after: " + str(len(data)))
#     print("dropping teamKills > 9")
#     data.drop(data[data["teamKills_team_match_max"] > 9].index, inplace=True)
#     print("rows after: " + str(len(data)))
#     print("dropping walkDistance > 14910")
#     data.drop(data[data["walkDistance_team_match_max"] > 14910].index, inplace=True)
#     print("rows after: " + str(len(data)))
#     print("dropping weaponsAcquired > 153")
#     data.drop(data[data["weaponsAcquired_team_match_max"] > 153].index, inplace=True)
#     print("rows after: " + str(len(data)))
#     print("dropping winPoints > 2000")
#     data.drop(data[data["winPoints_team_match_max"] > 2000].index, inplace=True)
#     print("rows after: " + str(len(data)))
#     
#     print("rows removed in total: " + str(rowsBefore - len(data)))
  
    gc.collect()
    
#     print("Converting winPlacePerc to loosing_order")
#     data_group = data[["matchId", "groupId", "winPlacePerc"]].groupby(["matchId", "groupId"]).first().reset_index()
#     data_group["loosing_order"] = data_group.groupby(["matchId"])["winPlacePerc"].rank(ascending=True)
#     data = data.merge(data_group[["loosing_order", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
#     del data_group
#     data_group=None
#     gc.collect()
    
    columns_to_drop = ["groupId", "matchId"]
    
    print("Dropping columns: " + ', '.join(columns_to_drop))
    data.drop(labels=columns_to_drop, axis=1, inplace=True)

    return data


if __name__ == '__main__':
    gc.enable()
    
    pd.options.mode.use_inf_as_na = True
    
    print()
    print("Loading data from " + TRAIN_FILE_NAME)

    data = pd.read_csv(TRAIN_FILE_NAME)
    
    data_size = len(data)

    print("read " + str(data_size) + " rows")

    print("data shape: " + str(data.shape))
#     print("data columns: " + str(data.columns))

    print("Dropping rows with no winPlacePerc and edge cases")
    data.drop(data[np.isnan(data['winPlacePerc'])].index, inplace=True)
    data.drop(data[data["numGroups"] == 1].index, inplace=True)
    data.drop(data[data["numGroups"] == 2].index, inplace=True)
    data.drop(data[data["maxPlace"] == 0].index, inplace=True)
    data.drop(data[data["maxPlace"] == 1].index, inplace=True)
    data["missing_groups_perc"] = (data["maxPlace"] - data["numGroups"]) / data["maxPlace"]
    data.drop(data[data["missing_groups_perc"] > 0.6].index, inplace=True)
    data.drop(labels=["missing_groups_perc"], axis=1, inplace=True)
    data = reduce_mem_usage(data)
    
    print("Sorting data on matchId")
    data = data.sort_values(by=["matchId"])
    n = int(len(data) * 0.34) #0.51 or 0.34 or 0.451 as a single max
    while data.iloc[n]["matchId"] == data.iloc[n - 1]["matchId"]:
        n = n - 1
    print("Final n: " + str(n))
#     print("n - 1: " + str(data.iloc[n - 1]["matchId"]))
#     print("n: " + str(data.iloc[n]["matchId"]))
#     print("n + 1: " + str(data.iloc[n + 1]["matchId"]))
    n2 = int(len(data) * 0.67) #0.51 or 0.55 or 0.67
    while data.iloc[n2]["matchId"] == data.iloc[n2 - 1]["matchId"]:
        n2 = n2 - 1
    print("Final n2: " + str(n2))
    
    n3 = int(len(data) * 0.98) # 0.74, 0.68
    while data.iloc[n3]["matchId"] == data.iloc[n3 - 1]["matchId"]:
        n3 = n3 - 1
        
    #TODO
    # n3 = len(data) - 1
    print("Final n3: " + str(n3))
    
    data = data[0:n]
    gc.collect()

    data = features_engineering(data)
    gc.collect()
    
    
    print()
    print("Reloading data from " + TRAIN_FILE_NAME)
    
    data2 = pd.read_csv(TRAIN_FILE_NAME)
    
    print("Dropping rows with no winPlacePerc and edge cases")
    data2.drop(data2[np.isnan(data2['winPlacePerc'])].index, inplace=True)
    data2.drop(data2[data2["numGroups"] == 1].index, inplace=True)
    data2.drop(data2[data2["numGroups"] == 2].index, inplace=True)
    data2.drop(data2[data2["maxPlace"] == 0].index, inplace=True)
    data2.drop(data2[data2["maxPlace"] == 1].index, inplace=True)
    data2["missing_groups_perc"] = (data2["maxPlace"] - data2["numGroups"]) / data2["maxPlace"]
    data2.drop(data2[data2["missing_groups_perc"] > 0.6].index, inplace=True)
    data2.drop(labels=["missing_groups_perc"], axis=1, inplace=True)
    data2 = reduce_mem_usage(data2)
    
    print("Sorting data on matchId")
    data2 = data2.sort_values(by=["matchId"])
    data2 = data2[n:n2]
    gc.collect()
    
    data2 = features_engineering(data2)
    gc.collect()
    
    print("Merging data & data2")
    data = data.append(data2, sort=False, ignore_index=True)
    del data2
    data2 = None
    data.drop_duplicates(subset=None, keep='first', inplace=True)
    gc.collect()
    
    
    print()
    print("Reloading data from " + TRAIN_FILE_NAME)
    
    data2 = pd.read_csv(TRAIN_FILE_NAME)
    
    print("Dropping rows with no winPlacePerc and edge cases")
    data2.drop(data2[np.isnan(data2['winPlacePerc'])].index, inplace=True)
    data2.drop(data2[data2["numGroups"] == 1].index, inplace=True)
    data2.drop(data2[data2["numGroups"] == 2].index, inplace=True)
    data2.drop(data2[data2["maxPlace"] == 0].index, inplace=True)
    data2.drop(data2[data2["maxPlace"] == 1].index, inplace=True)
    data2["missing_groups_perc"] = (data2["maxPlace"] - data2["numGroups"]) / data2["maxPlace"]
    data2.drop(data2[data2["missing_groups_perc"] > 0.6].index, inplace=True)
    data2.drop(labels=["missing_groups_perc"], axis=1, inplace=True)
    data2 = reduce_mem_usage(data2)
    
    print("Sorting data on matchId")
    data2 = data2.sort_values(by=["matchId"])
    data2 = data2[n2:n3]
    gc.collect()
    
    data2 = features_engineering(data2)
    gc.collect()
    
    print("Merging data & data2")
    data = data.append(data2, sort=False, ignore_index=True)
    del data2
    data2 = None
    data.drop_duplicates(subset=None, keep='first', inplace=True)
    gc.collect()
    
    
    
    print("Splitting features from prediction")

    target = data["winPlacePerc"]
    data.drop(labels=["winPlacePerc"], axis=1, inplace=True)
    
    print("new data shape: " + str(data.shape))

    print()

    gc.collect()
        
    print()
    print("Building LGBM model")
    
    params = {"objective" : "regression_l1",
              "boosting" : "gbdt", # gbdt / goss 
              "metric" : "mae", 
              "num_iterations" : 5000,
              "early_stopping_rounds" : 50,
              "top_k" : 20, # try greater value (may slower, but should get better results
              "max_depth" : -1, # -1 default = no limit
              "num_leaves" : 2000, # TODO try greater values, default 31;
              "min_data_in_leaf" : 20, #default
              "learning_rate" : 0.1, 
              "bagging_fraction" : 0.7, # 0.7; 1 is default, but does nothing without setting bagging_freq
              "bagging_seed" : 3, # default 3
              "bagging_freq" : 5, # default 0
              "feature_fraction" : 0.75, # % of features to take per tree, default 1 
              "num_threads" : 4
             }
             
    dataset_params = {"max_bin" : 255, # 65530, default 255
                      "min_data_in_bin" : 5 # default 3
                     }
    
    if os.path.exists("lgb_train_data.bin"):
        os.remove("lgb_train_data.bin")
    
    print("Building dataset...")
    lgb_train = lgb.Dataset(data, label=target, params=dataset_params)
    del data
    data = None
    del target
    target = None
    gc.collect()
    print("Storing dataset into binary file")
    lgb_train.save_binary("lgb_train_data.bin")
    del lgb_train
    lgb_train = None
    gc.collect()
    print("Reloading dataset from binary file")
    lgb_train = lgb.Dataset("lgb_train_data.bin")
    
    print()
    print("Running learning...")
    
    model = lgb.train(params, lgb_train, valid_sets=[lgb_train], verbose_eval=100, keep_training_booster=True)
    
    
    os.remove("lgb_train_data.bin")
    
    del lgb_train
    lgb_train = None
    gc.collect()
    
    

    
    print("Loading test file " + str(TEST_FILE_NAME))
    
    test = pd.read_csv(TEST_FILE_NAME)
    test = reduce_mem_usage(test)
    
    print("Sorting data on matchId")
    test = test.sort_values(by=["matchId"])
    n = int(len(test) * 0.34)
    while test.iloc[n]["matchId"] == test.iloc[n - 1]["matchId"]:
        n = n - 1
    print("Final n1: " + str(n))
    n2 = int(len(test) * 0.67)
    while test.iloc[n2]["matchId"] == test.iloc[n2 - 1]["matchId"]:
        n2 = n2 - 1
    print("Final n2: " + str(n2))
    
    test = test[0:n]
    gc.collect()
    
    player_ids = test["Id"]
    test = features_engineering(test, learnMode=False)
    
    gc.collect()
    
    print()
    print("Predicting...")
    y_pred = model.predict(test.values, num_iteration=model.best_iteration)
    # y_pred = np.around(y_pred, decimals=4)
    
    print("Storing prediction to file...")
    
    y_pred=pd.DataFrame(y_pred, player_ids, columns=['winPlacePerc'])
    
    y_pred.to_csv("lgbm_submission_1.csv", sep=',')
    
    del test
    del y_pred
    test = None
    y_pred = None
    gc.collect()
    
    
    print("Reloading test file " + str(TEST_FILE_NAME))
    
    test = pd.read_csv(TEST_FILE_NAME)
    test = reduce_mem_usage(test)
    
    print("Sorting data on matchId")
    test = test.sort_values(by=["matchId"])
    
    test = test[n:n2]
    gc.collect()
    
    player_ids = test["Id"]
    test = features_engineering(test, learnMode=False)
    
    gc.collect()
    
    print()
    print("Predicting...")
    y_pred = model.predict(test.values, num_iteration=model.best_iteration)
    # y_pred = np.around(y_pred, decimals=4)
    
    print("Storing prediction to file...")
    
    y_pred=pd.DataFrame(y_pred, player_ids, columns=['winPlacePerc'])
    
    y_pred.to_csv("lgbm_submission_2.csv", sep=',')
    
    del test
    del y_pred
    test = None
    y_pred = None
    gc.collect()
    
    
    print("Reloading test file " + str(TEST_FILE_NAME))
    
    test = pd.read_csv(TEST_FILE_NAME)
    test = reduce_mem_usage(test)
    
    print("Sorting data on matchId")
    test = test.sort_values(by=["matchId"])
    
    test = test[n2:]
    gc.collect()
    
    player_ids = test["Id"]
    test = features_engineering(test, learnMode=False)
    
    gc.collect()
    
    print()
    print("Predicting...")
    y_pred = model.predict(test.values, num_iteration=model.best_iteration)
    # y_pred = np.around(y_pred, decimals=4)
    
    print("Storing prediction to file...")
    
    y_pred=pd.DataFrame(y_pred, player_ids, columns=['winPlacePerc'])
    
    y_pred.to_csv("lgbm_submission_3.csv", sep=',')
    
    features_names = model.feature_name()
    importances = model.feature_importance(importance_type="split")
    indices = np.argsort(importances)[::-1]

    print()
    print("Feature ranking:")
    
    for f in range(len(features_names)):
        print("%d. feature %d %s (%f)" % (f + 1, indices[f], features_names[f], importances[indices[f]]))
    
    print()
    
    print("Finished.")