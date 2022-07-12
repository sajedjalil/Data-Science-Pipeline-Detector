import pandas as pd
import time_utils

RAW_DATA_DIR = "../input/walmart-recruiting-store-sales-forecasting/"
WM_DATA_DIR = "../input/week-month-data/"

DTYPE = {"Store": str, "Dept": str, "Date": str, "Weekly_Sales": float, "IsHoliday": bool, "Temperature": float,
         "Fuel_Price": float, "MarkDown1": float, "MarkDown2": float,"MarkDown3": float,"MarkDown4": float,
         "MarkDown5": float, "CPI": float, "Unemployment": float, "Type": str,
         "Size": int}


def load_dataset(dataset):
    data = pd.read_csv(RAW_DATA_DIR + "{}.csv.zip".format(dataset), dtype=DTYPE)
    data["timestamp"] = data["Date"].apply(lambda str_dt: time_utils.str_datetime_to_timestamp(str_dt, "%Y-%m-%d"))
    data["store_dept"] = data["Store"] + "_" + data["Dept"]
    return data.sort_values("timestamp")

def load_features():
    feat = pd.read_csv(RAW_DATA_DIR + "features.csv.zip", dtype=DTYPE)
    feat["timestamp"] = feat["Date"].apply(lambda str_dt: time_utils.str_datetime_to_timestamp(str_dt, "%Y-%m-%d"))
    return feat.sort_values("timestamp")

def load_stores():
    return pd.read_csv(RAW_DATA_DIR + "stores.csv", dtype=DTYPE)

def save_week_month_data(df, data_filename):
    df.to_csv(WM_DATA_DIR + data_filename + ".csv", sep=";", index=False)
    
def load_week_month_data(data_filename):
    return pd.read_csv(WM_DATA_DIR + data_filename + ".csv", sep=";")
    
    
    