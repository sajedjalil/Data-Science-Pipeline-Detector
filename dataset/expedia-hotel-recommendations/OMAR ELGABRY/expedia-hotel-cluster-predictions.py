# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy
import numpy as np

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# Define training and testing sets

train_df = pd.read_csv('../input/train.csv', usecols=['hotel_cluster'])

# Construct dest_clusters dictionary

from collections import defaultdict
from csv import DictReader

dest_clusters          = defaultdict(lambda: defaultdict(int))

for i, row in enumerate(DictReader(open("../input/train.csv"))):
    
    # for every destiantion, increment it's hotel_cluster by 1
    dest_clusters[row["srch_destination_id"]][row["hotel_cluster"]] += 1
    
    
# Create Submission

def get_top_clusters(d, l=5):
    return " ".join(str(x) for x in sorted(d, key=d.get, reverse=True)[:l])

# ferq_cluster_default: the most frequent hotel clusters
ferq_cluster_default = defaultdict(int, train_df['hotel_cluster'].value_counts()[:5])

with open("expedia.csv", "w") as outfile:
    outfile.write("id,hotel_cluster\n")
    
    for i, row in enumerate(DictReader(open("../input/test.csv"))):
        
        row_id = row["id"]
        
        # freq_cluster_d:    the frequency of all hotel clusters for a destiantion
        freq_cluster_d       = dest_clusters[row["srch_destination_id"]]

        if(len(freq_cluster_d) >= 5):
            outfile.write("{},{}\n".format(row_id, get_top_clusters(freq_cluster_d)))
        elif(len(freq_cluster_d) < 5 and len(freq_cluster_d) > 0):
            top_str = get_top_clusters(freq_cluster_d, len(freq_cluster_d))
            top_str = top_str + " " + get_top_clusters(ferq_cluster_default, 5 - len(freq_cluster_d))
            outfile.write("{},{}\n".format(row_id, top_str))
        else:
            outfile.write("{},{}\n".format(row_id, get_top_clusters(ferq_cluster_default)))