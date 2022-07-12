# Inspired by notebooks from Julián Peller and Usman Abbas
# Extracts only the hits column to save memory
# Writes one CSV per MAXROWS rows

import pandas as pd 
from pandas.io.json import json_normalize
from ast import literal_eval

MAXROWS = 1e5 # per CSV

i = rows = 0 
for file in ['../input/train_v2.csv', '../input/test_v2.csv']:
    key_columns = ['fullVisitorId', 'visitId']
    array_column = 'hits'
    USE_COLUMNS = key_columns+[array_column]

    reader = pd.read_csv(file, dtype={'fullVisitorId': 'str'}, usecols=USE_COLUMNS, chunksize = MAXROWS, skiprows=0, nrows=1000)
    for df in reader:
        df[array_column][df[array_column] == "[]"] = "[{}]"
        df[array_column]=df[array_column].apply(literal_eval)
        df[key_columns] = df[key_columns].astype(str)
        df['key'] = df[key_columns].apply(lambda x: '_'.join(x), axis=1)
        df = df.drop(key_columns, axis=1)
        df = df.join(df[array_column].apply(pd.Series)).drop(array_column, 1).set_index([u'key']).stack().reset_index().drop('level_1', 1).rename(columns={0:array_column})
        column_as_df = json_normalize(df[array_column])
        column_as_df.columns = [f"{array_column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(array_column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        
        df.to_csv(f"hits-{i:03d}.csv", index=False)
        rows += len(df.index)
        print(f"hits-{i:05d}.csv written ({rows} cumulative rows)")
        i += 1












