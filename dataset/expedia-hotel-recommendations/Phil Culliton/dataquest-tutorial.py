import pandas as pd
import numpy as np
import random
import ml_metrics as metrics
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from itertools import chain
import operator
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

destinations = pd.read_csv("../input/destinations.csv")

train = pd.read_csv('../input/train.csv',
                            usecols=["date_time", "user_location_country", "user_location_region", "user_location_city",
                                    "user_id", "is_booking", "orig_destination_distance",
                                     "hotel_cluster", "srch_ci", "srch_co", "srch_destination_id", 
                                     "hotel_continent", "hotel_country", "hotel_market"],
                            dtype={'date_time':np.str_, 'user_location_country':np.int8, 
                                   'user_location_region':np.int8, 'user_location_city':np.int8, 
                                   'user_id':np.int32, 'is_booking':np.int8,
                                   "orig_destination_distance":np.float64,
                                   "hotel_cluster":np.int8,
                                   'srch_ci':np.str_, 'srch_co':np.str_,
                                   "srch_destination_id":np.int32,
                                   "hotel_continent":np.int8,
                                   "hotel_country":np.int8,
                                   "hotel_market":np.int8},
                            nrows=1000000
                           )
                           
test = pd.read_csv('../input/test.csv',
                           usecols=["id", "date_time", "user_location_country", "user_location_region", "user_location_city",
                                "user_id", "orig_destination_distance",
                                   "srch_ci", "srch_co", "srch_destination_id",
                                   "hotel_continent", "hotel_country", "hotel_market"],
                            dtype={'id':np.int32, 'date_time':np.str_, 'user_location_country':np.int8, 
                            'user_location_region':np.int8, 'user_location_city':np.int8, 
                            'user_id':np.int32, 
                            "orig_destination_distance":np.float64, 'srch_ci':np.str_, 'srch_co':np.str_,
                                   "srch_destination_id":np.int32,
                                   "hotel_continent":np.int8,
                                   "hotel_country":np.int8,
                                   "hotel_market":np.int8})	
train.shape
test.shape
train.head(5)
test.head(5)

train["hotel_cluster"].value_counts()
test_ids = set(test.user_id.unique())
train_ids = set(train.user_id.unique())
intersection_count = len(test_ids & train_ids)
intersection_count == len(test_ids)

train["date_time"] = pd.to_datetime(train["date_time"])
train["year"] = train["date_time"].dt.year
train["month"] = train["date_time"].dt.month

unique_users = train.user_id.unique()

sel_user_ids = [unique_users[i] for i in sorted(random.sample(range(len(unique_users)), 1000)) ]
sel_train = train[train.user_id.isin(sel_user_ids)]

# t1 = sel_train[((sel_train.year == 2013) | ((sel_train.year == 2014) & (sel_train.month < 8)))]
# t2 = sel_train[((sel_train.year == 2014) & (sel_train.month >= 8))]
# 
# t2 = t2[t2.is_booking == True]

t1 = train
t2 = test

most_common_clusters = list(train.hotel_cluster.value_counts().head().index)
# predictions = [most_common_clusters for i in range(t2.shape[0])]
# target = [[l] for l in t2["hotel_cluster"]]
# metrics.mapk(target, predictions, k=5)
# train.corr()["hotel_cluster"]

def make_key(items):
    return "_".join([str(i) for i in items])

match_cols = ["srch_destination_id"]
cluster_cols = match_cols + ['hotel_cluster']
groups = t1.groupby(cluster_cols)
top_clusters = {}
for name, group in groups:
    clicks = len(group.is_booking[group.is_booking == False])
    bookings = len(group.is_booking[group.is_booking == True])
    
    score = bookings + 0.15 * clicks
    
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
		
metrics.mapk([[l] for l in t2["hotel_cluster"]], preds, k=5)
match_cols = ['user_location_country', 'user_location_region', 'user_location_city', 'hotel_market', 'orig_destination_distance']

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
metrics.mapk([[l] for l in t2["hotel_cluster"]], full_preds, k=5)

write_p = [" ".join([str(l) for l in p]) for p in full_preds]
write_frame = ["{0},{1}".format(t2["srch_destination_id"].iloc[i], write_p[i]) for i in range(len(full_preds))]
write_frame = ["id,hotel_cluster"] + write_frame
#write_frame = ["{0}".format(write_p[i]) for i in range(len(full_preds))]
#write_frame = ["hotel_clusters"] + write_frame
with open("predictions.csv", "w+") as f:
    f.write("\n".join(write_frame))