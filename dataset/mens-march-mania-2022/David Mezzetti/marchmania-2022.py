# %% [code]
import re

import numpy as np
import pandas as pd

def seed(value):
    """
    Convert seeds to int

    Args:
        value: input value (examples X01, X16, Y12)

    Returns:
        seed as int
    """

    return int(re.sub("[A-Za-z]", "", value))

def day(value):
    """
    Transforms a day number to a game round. No tournament in 2020 and 2021 tournament skipped.
    
    Args:
        value: input day number
    
    Returns:
        game round
    """

    lookup = {136: "R64 D1", 137: "R64 D2", 138: "R32 D1", 139: "R32 D2", 143: "S16 D1", 144: "S16 D2",
              145: "E8 D1", 146: "E8 D2", 152: "F4", 154: "F2"}

    return lookup[value]

def transform(row):
    """
    Transforms a result row to have the lower (better) seed first. Adds MOV and a flag if this is a win by the lower seed.

    Args:
        row: input row

    Returns:
        transformed row
    """

    result = (row["Season"], row["Day"], row["DayNum"], row["CityID"] if not np.isnan(row["CityID"]) else -1)
    
    if row["WSeed"] > row["LSeed"]:
        result = result + (row["LSeed"], row["LTeamID"], row["LScore"], row["WSeed"], row["WTeamID"], row["WScore"], row["LScore"] - row["WScore"])
        result = result + (1 if row["LScore"] - row["WScore"] > 0 else 0,)
    else:
        result = result + (row["WSeed"], row["WTeamID"], row["WScore"], row["LSeed"], row["LTeamID"], row["LScore"], row["WScore"] - row["LScore"])
        result = result + (1 if row["WScore"] - row["LScore"] > 0 else 0,)

    result = tuple(int(x) if isinstance(x, float) else x for x in result)
    return pd.Series(result)
    
def results():
    """
    Reads historical tournament results and adds seeds, margin of victory and if the game was a win by the lower (better) seed.

    Returns:
        data frame with joined results
    """
    
    # Read data
    results = pd.read_csv("/kaggle/input/mens-march-mania-2022/MDataFiles_Stage1/MNCAATourneyCompactResults.csv")
    seeds = pd.read_csv("/kaggle/input/mens-march-mania-2022/MDataFiles_Stage1/MNCAATourneySeeds.csv")
    cities = pd.read_csv("/kaggle/input/mens-march-mania-2022/MDataFiles_Stage1/MGameCities.csv")
    
    # Join results and seeds
    results = results.merge(seeds, how="left", left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"])
    results = results.merge(seeds, how="left", left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"])
    results = results.merge(cities, how="left", left_on=["Season", "DayNum", "WTeamID", "LTeamID"], right_on=["Season", "DayNum", "WTeamID", "LTeamID"])
    
    # Remove First Four games
    results = results[results.DayNum.gt(135)]
    
    # Remove 2021 tournament
    results = results[results.Season.ne(2021)]
    
    results["WSeed"] = results["Seed_x"].apply(seed)
    results["LSeed"] = results["Seed_y"].apply(seed)
    results["Day"] = results["DayNum"].apply(day)
    
    # Transform data
    results = results.apply(transform, axis=1)
    results.columns = ["Season", "Day", "DayNum", "CityID", "LSeed", "LTeamID", "LScore", "HSeed", "HTeamID", "HScore", "MOV", "Wins"]

    return results

def averages(results):
    """
    Gets the averages across all tournament results grouped by seed combination.

    Args:
        results: all tournament results

    Returns:
        data frame with averages
    """

    averages = results.groupby(["LSeed", "HSeed"]).agg({"LScore": ["mean"], "HScore": ["mean"], "MOV": ["mean"], "Wins": ["sum", "count"]}).reset_index()

    averages["Key"] = averages.apply(lambda x: "%.0f_%.0f" % (x["LSeed"].astype(str), x["HSeed"].astype(str)), axis=1)
    averages.columns = ["LSeed", "HSeed", "LScore", "HScore", "MOV", "Wins", "Count", "Key"]

    averages["WinPct"] = averages.apply(lambda x: x["Wins"] / x["Count"], axis=1)

    return averages

def stats(averages, reset=True):
    """
    Computes statistics for an averages data frame. 
    
    Args:
        averages: result averages
        reset: if reset_index should be called before returning
        
    Returns:
        statistics
    """

    averages = averages.agg({"LScore": ["mean"], "HScore": ["mean"], "MOV": ["mean"], "Wins": ["sum"], "Count": ["sum"], "WinPct": ["mean"]}).sum()

    return averages.reset_index() if reset else averages