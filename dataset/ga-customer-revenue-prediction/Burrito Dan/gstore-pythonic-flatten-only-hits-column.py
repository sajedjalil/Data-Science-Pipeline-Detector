# Inspired by notebooks from Julián Peller and Usman Abbas
# Extracts only the hits column to save memory
# Writes one CSV per MAXROWS rows

import pandas as pd 
from pandas.io.json import json_normalize
from ast import literal_eval

MAXROWS = 1e5 # per CSV

i = rows = 0 
for file in ['../input/train_v2.csv', '../input/test_v2.csv']:
    reader = pd.read_csv(file, usecols=[6], chunksize = MAXROWS, skiprows=0)
    for chunk in reader:
        chunk.columns = ['hits']
        chunk['hits'][chunk['hits'] == "[]"] = "[{}]"
        chunk['hits'] = chunk['hits'].apply(literal_eval).str[0]
        chunk = json_normalize(chunk['hits'])

        # Extract the product and promo names from the complex nested structure into a simple flat list:
        if 'product' in chunk.columns:
            #print(chunk['product'][0])
            chunk['v2ProductName'] = chunk['product'].apply(lambda x: [p['v2ProductName'] for p in x] if type(x) == list else [])
            chunk['v2ProductCategory'] = chunk['product'].apply(lambda x: [p['v2ProductCategory'] for p in x] if type(x) == list else [])
            del chunk['product']
        if 'promotion' in chunk.columns:
            #print(chunk['promotion'][0])
            chunk['promoId']  = chunk['promotion'].apply(lambda x: [p['promoId'] for p in x] if type(x) == list else [])
            chunk['promoName']  = chunk['promotion'].apply(lambda x: [p['promoName'] for p in x] if type(x) == list else [])
            del chunk['promotion']

        chunk.to_csv(f"hits-{i:05d}.csv", index=False)
        rows += len(chunk.index)
        print(f"hits-{i:05d}.csv written ({rows} cumulative rows)")
        i += 1
