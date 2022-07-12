#importing libraries
import pandas as pd
import numpy as np
#reading the files
holidays_events_df = pd.read_csv('../input/holidays_events.csv', 

low_memory=False)
items_df = pd.read_csv('../input/items.csv', low_memory=False)
oil_df = pd.read_csv('../input/oil.csv', low_memory=False)
stores_df = pd.read_csv('../input/stores.csv', low_memory=False)
transactions_df = pd.read_csv('../input/transactions.csv', low_memory=False)
# adding favourites days column for shopping
import calendar
transactions_df["year"] = transactions_df["date"].astype(str).str[:4].astype(np.int64)
transactions_df["month"] = transactions_df["date"].astype(str).str[5:7].astype

(np.int64)
transactions_df['date'] = pd.to_datetime(transactions_df['date'], errors ='coerce')
transactions_df['day_of_week'] = transactions_df['date'].dt.weekday_name
transactions_df["year"] = transactions_df["year"].astype(str)
transactions_df.head()