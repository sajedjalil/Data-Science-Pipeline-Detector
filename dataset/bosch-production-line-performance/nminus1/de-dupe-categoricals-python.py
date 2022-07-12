# Load required libraries
import pandas as pd
import numpy as np
import json

df = pd.read_csv("../input/train_categorical.csv",nrows=1)
cols=df.columns
clen=len(cols)
#number of slices that you want to cut the columnset into so that each slice can fit into memory
n = 20
col_slice = int(clen/n)
print (clen,col_slice)
# dictionary to store hash,columnlist
col_hash={}

# process each column slice of the input file
for i in range(n):
    left = i*(col_slice)
    right = (i+1)*(col_slice)+1
    print (i,left,right)
    df = pd.read_csv('../input/train_categorical.csv', dtype = str,skipinitialspace=True, usecols=cols[left:right])
    for c in cols[left:right]:
        hash_val=hash(tuple(df[c]))
        if hash_val in col_hash:
            col_hash[hash_val].append(c)
        else:
            col_hash[hash_val]=[c]
    print (len(col_hash))

# write the feature summary to file
json.dump(col_hash, open("feature_summary.json",'w'))

# print all unique columns
for key in col_hash:
    print (col_hash[key][0])




