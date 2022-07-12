# Script based on https://www.dataquest.io/blog/kaggle-tutorial/ by Vik Paruchuri
# Thanks a lot Vik

import numpy as np
import pandas as pd
from datetime import datetime
import operator
import random
import ml_metrics as metrics


# To make the submission file cv = False and remove nrows = 10000 when you 
# read train and test files
cv = True
n_users = 25 #100000 #250000

#######################################################################
#            Reading train data (do not run for CV)                   #
#######################################################################
train = pd.read_csv('../input/train.csv',
                    dtype={'user_id': np.int32,
                           'is_booking':bool,
                           'srch_destination_id':np.int32, 
                           'hotel_cluster':np.int32,
                           'date_time': np.str_,
                           'user_location_country': np.int32,
                           'user_location_region': np.int32,
                           'user_location_city': np.int32,
                           'hotel_market': np.int32,
                           'orig_destination_distance': np.float64
                           },
                    usecols=['user_id', 'is_booking', 'srch_destination_id', 'hotel_cluster',
                             'date_time', 'user_location_country', 'user_location_region', 
                            'user_location_city', 'hotel_market', 'orig_destination_distance'],
                    nrows = 10000
                   )
print("Train shape", train.shape)

train["date_time"] = pd.to_datetime(train["date_time"])
train["year"] = train["date_time"].dt.year
train["month"] = train["date_time"].dt.month

#######################################################################
#     Split for cross validation (do not run for CV)                  #
#######################################################################
# We create train_cv and test_cv for CV (it is much faster)
unique_users = train.user_id.unique()
print('We have',len(unique_users),'users')

sel_user_ids = [unique_users[i] for i in sorted(random.sample(range(len(unique_users)), n_users)) ]

if cv:
    sel_train = train[train.user_id.isin(sel_user_ids)]
    print('Train shape:',sel_train.shape)
    t1 = sel_train[((sel_train.year == 2013) | ((sel_train.year == 2014) & (sel_train.month < 8)))]
    t2 = sel_train[((sel_train.year == 2014) & (sel_train.month >= 8))]
    
    t2 = t2[t2.is_booking == True]
    target_test = [[l] for l in t2["hotel_cluster"]]
    t1.to_csv('train_cv.csv')
    t2.to_csv('test_cv.csv')
else:
    t1 = train

#######################################################################
#  Start here to make cv (once you have train_cv.csv and test_cv.csv) #
#######################################################################
if cv == False:
    print('Reading test...')
    t2    = pd.read_csv("../input/test.csv",
                        dtype={'id': np.int32,
                               'user_id': np.int32,
                               'srch_destination_id':np.int32, 
                               'date_time': 'O',
                               'user_location_country': np.int32,
                               'user_location_region': np.int32,
                               'user_location_city': np.int32,
                               'hotel_market': np.int32,
                               'orig_destination_distance': np.float64
                               },
                        usecols=['id', 'user_id', 'srch_destination_id','date_time',
                                'user_location_country', 'user_location_region', 
                                'user_location_city', 'hotel_market', 'orig_destination_distance'],
                        nrows = 10000
                        )
    print("Test shape", t2.shape)
    train_ids = set(t1.user_id.unique())
    test_ids = set(t2.user_id.unique())

    intersection_count = len(test_ids & train_ids)
    print(intersection_count == len(test_ids))
    
    t2["date_time"] = pd.to_datetime(t2["date_time"])
    t2["year"] = t2["date_time"].dt.year
    t2["month"] = t2["date_time"].dt.month
else:
    t1 = pd.read_csv('train_cv.csv')
    t2 = pd.read_csv('test_cv.csv')
    print("Train shape", t1.shape)
    print("Test shape", t2.shape)
    target_test = [[l] for l in t2["hotel_cluster"]]
    
#######################################################################
#                     Model 1: Most common clusters                   #
#######################################################################
most_common_clusters = list(t1.hotel_cluster.value_counts().head().index)

predictions = [most_common_clusters for i in range(t2.shape[0])]
if cv:
    print('Score:',metrics.mapk(target_test, predictions, k=5))
#Score: 0.0655603769471 (250000 users) 

#######################################################################
#                                Model 2                              #
#######################################################################
def make_key(items):
    return "_".join([str(i) for i in items])

match_cols = ["srch_destination_id"]
cluster_cols = match_cols + ['hotel_cluster']
groups = t1.groupby(cluster_cols)
top_clusters = {}
for name, group in groups:
    clicks = len(group.is_booking[group.is_booking == False])
    bookings = len(group.is_booking[group.is_booking == True])
    
    score = 1.00 * bookings + 0.15 * clicks
    
    clus_name = make_key(name[:len(match_cols)])
    if clus_name not in top_clusters:
        top_clusters[clus_name] = {}
    top_clusters[clus_name][name[-1]] = score

cluster_dict = {}
for n in top_clusters:
    tc = top_clusters[n]
    top = [l[0] for l in sorted(tc.items(), key=operator.itemgetter(1), reverse=True)[:5]]
    cluster_dict[n] = top

preds = []
for index, row in t2.iterrows():
    key = make_key([row[m] for m in match_cols])
    if key in cluster_dict:
        preds.append(cluster_dict[key])
    else:
        preds.append([])
if cv:
    print('Score:',metrics.mapk(target_test, preds, k=5))
#Score: 0.303252335447 (250,000 users)
#Score: 0.269545566807 (25,000 users)

#######################################################################
#                              Data leak                              #
#######################################################################
start = datetime.now()
match_cols = ['user_location_country', 'user_location_region', 
              'user_location_city', 'hotel_market', 'orig_destination_distance']

groups = t1.groupby(match_cols)
    
def generate_exact_matches(row, match_cols):
    index = tuple([row[t] for t in match_cols])
    try:
        group = groups.get_group(index)
    except Exception:
        return []
    clus = list(set(group.hotel_cluster))
    return clus

exact_matches = []
for i in range(t2.shape[0]):
    exact_matches.append(generate_exact_matches(t2.iloc[i], match_cols))
    if i % 100000 == 0:
        print("%s\t%s"%(i,datetime.now()-start))
print('Total time:',datetime.now()-start)
#Full model: Total time: 0:24:40.167909

#######################################################################
#                     Making ensemble of models                       #
#######################################################################
def f5(seq, idfun=None): 
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

full_preds = [f5(exact_matches[p] + preds[p] + most_common_clusters)[:5] for p in range(len(preds))]
if cv:
    print('Score:',metrics.mapk(target_test, full_preds, k=5))
# 0.48393 at LB vs 0.41650 at CV (250,000 unique_users)
# Score: 0.320157240471 at CV (25,000 users)

#######################################################################
#                            Submission file                          #
#######################################################################
if cv == False:
    write_p = [" ".join([str(l) for l in p]) for p in full_preds]
    write_frame = ["{0},{1}".format(t2["id"][i], write_p[i]) for i in range(len(full_preds))]
    write_frame = ["id,hotel_cluster"] + write_frame
    with open("tutorial_v2.csv", "w+") as f:
        f.write("\n".join(write_frame))
