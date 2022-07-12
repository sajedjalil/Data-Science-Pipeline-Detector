
# Standard pandas and numpy imports

import pandas as pd
import numpy as np

# Cross-validation imports
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split

# For logloss metric
import sklearn.metrics

# Load train and test sets

train_file = pd.read_csv('../input/train.csv')
test_file  = pd.read_csv('../input/test.csv')

num_train = len(train_file)

# Combine train/test sets into one dataframe for easier binning/preprocessing

all_data = pd.concat((train_file, test_file), axis=0, ignore_index=True)

# Replace all NA/NaN's with -100
all_data.fillna(-100,inplace=True)

# This converts a feature into a histogram-ish binned feature, and 
def binner(key, maxbins = 101, na = -100, percent_per_bin = 1):
    akey = all_data.loc[all_data[key] != na, key].copy()
    
    count = len(akey.unique())
    
    if count < maxbins:
        return (all_data[key], None)
    
    try:
        bins = np.unique(np.percentile(akey, np.arange(0, 100, percent_per_bin)))
        # Add a bin for NA
        if np.min(all_data[key]) == na:
            bins = np.insert(bins, 0, na + 1)
        count = len(bins)
    
        # print(key, count)
    
        return (np.digitize(all_data[key], bins), bins)
    except:
        return (all_data[key], None)

(all_data['v50_binned'], v50_bins) = binner('v50', percent_per_bin = 5)

# Split all_data back into train+test

train_all = all_data.iloc[:num_train]
test = all_data.iloc[num_train:]

# Fill up (out-of) bag for the bag ensemble later
# (You need a recent scikit to stratify the train/oob set here)
train_bag, valid_oob = train_test_split(train_all, test_size = 0.1, random_state=12345, stratify=train_all.target)

# Set up cross-validation
cv_folds = 5
skf = StratifiedKFold(train_bag.target, n_folds=cv_folds, shuffle=True, random_state=12345)

train_cv = []
train_cv_items = []
valid_cv = []
valid_cv_items = []

x = 0

for train_index, test_index in skf:
    train_cv_items.append(train_index)
    train_cv.append(train_bag.iloc[train_index])
    
    valid_cv_items.append(test_index)    
    valid_cv.append(train_bag.iloc[test_index])
    x += len(test_index)

# Allocate arrays for predictions

avg = np.sum(train_bag.target) / len(train_bag)

preds_test = np.full((cv_folds, len(test_file)), avg, dtype=np.float32)
preds_valid_oob = np.full((cv_folds, len(valid_oob)), avg, dtype=np.float32)

preds_train_cv = np.full(len(train_bag), 0, dtype=np.float32)

# Each CV fold may have a different length, so allocate them into a list
preds_valid = []
for cv in range(cv_folds):
    preds_valid.append(np.full(len(valid_cv[cv]), avg, dtype=np.float32))
    
# Now compute probabilities.  This is a very simple model based on the binned version of v50
for cv in range(cv_folds):
    # Figure out the probabilities *for this fold only*
    
    for v in train_cv[cv].v50_binned.unique():
        subset = train_cv[cv][train_cv[cv].v50_binned == v]
        
        if len(subset) > 0:
            ratio = np.sum(subset.target) / len(subset)
        
            # 1.0 causes failure on the upcoming logloss test
            if ratio == 1.0:
                ratio = .9999999
        
            preds_valid[cv][np.where(valid_cv[cv].v50_binned == v)] = ratio
            preds_valid_oob[cv][np.where(valid_oob.v50_binned == v)] = ratio
            preds_test[cv][np.where(test.v50_binned == v)] = ratio
            
    preds_train_cv[valid_cv_items[cv]] = preds_valid[cv]
    
    print('fold #', cv, 'out of bag logloss:', sklearn.metrics.log_loss(valid_oob.target, preds_valid_oob[cv]))
    
# Figure out OOB combined
preds_valid_oob_merged = np.mean(preds_valid_oob, axis=0)
print('combined OOB logloss', sklearn.metrics.log_loss(valid_oob.target, preds_valid_oob_merged))

print('Cross-validation logloss:', sklearn.metrics.log_loss(train_bag.target, preds_train_cv))

# Prepare submission (not that it's worth much ;) )

preds_test_comb = np.mean(preds_test, axis=0)

preds_out = pd.DataFrame({"ID": test_file['ID'].values, "PredictedProb": preds_test_comb})
preds_out = preds_out.set_index('ID')
preds_out.to_csv('simple-cv-v50.csv')
