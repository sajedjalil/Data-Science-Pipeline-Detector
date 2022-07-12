# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale
from sklearn.decomposition import TruncatedSVD, SparsePCA
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#########################################################################################
#   App Events
#########################################################################################
print("# Read App Events")
app_ev = pd.read_csv("../input/app_events.csv")

# remove duplicates(app_id)
app_ev = app_ev.groupby("event_id")["app_id"].apply(
    lambda x: " ".join(set("app_id:" + str(s) for s in x)))

#########################################################################################
#     Events
#########################################################################################
print("# Read Events")
events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})
events["app_id"] = events["event_id"].map(app_ev)

events = events.dropna()

del app_ev

events = events[["device_id", "app_id"]]

# remove duplicates(app_id)
events = events.groupby("device_id")["app_id"].apply(
    lambda x: " ".join(set(str(" ".join(str(s) for s in x)).split(" "))))
events = events.reset_index(name="app_id")

# expand to multiple rows
events = pd.concat([pd.Series(row['device_id'], row['app_id'].split(' '))
                    for _, row in events.iterrows()]).reset_index()
events.columns = ['app_id', 'device_id']

################################################################################
#   Phone Brand
################################################################################
print("# Read Phone Brand")
pbd = pd.read_csv("../input/phone_brand_device_model.csv",
                  dtype={'device_id': np.str})
pbd.drop_duplicates('device_id', keep='first', inplace=True)


################################################################################
#  Train and Test
################################################################################
print("# Generate Train and Test")

train = pd.read_csv("../input/gender_age_train.csv",
                    dtype={'device_id': np.str})
train.drop(["age", "gender"], axis=1, inplace=True)

test = pd.read_csv("../input/gender_age_test.csv",
                   dtype={'device_id': np.str})
test["group"] = np.nan

split_len = len(train)

# Group Labels
Y = train["group"]
lable_group = LabelEncoder()
Y = lable_group.fit_transform(Y)
device_id = test["device_id"]

# Concat
Df = pd.concat((train, test), axis=0, ignore_index=True)

Df = pd.merge(Df, pbd, how="left", on="device_id")
Df["phone_brand"] = Df["phone_brand"].apply(lambda x: "phone_brand:" + str(x))
Df["device_model"] = Df["device_model"].apply(
    lambda x: "device_model:" + str(x))


################################################################################
#  Concat Feature
################################################################################

f1 = Df[["device_id", "phone_brand"]]   # phone_brand
f2 = Df[["device_id", "device_model"]]  # device_model
f3 = events[["device_id", "app_id"]]    # app_id

del Df

f1.columns.values[1] = "feature"
f2.columns.values[1] = "feature"
f3.columns.values[1] = "feature"

FLS = pd.concat((f1, f2, f3), axis=0, ignore_index=True)


################################################################################
# User-Item Feature
################################################################################
print("# User-Item-Feature")

device_ids = FLS["device_id"].unique()
feature_cs = FLS["feature"].unique()

data = np.ones(len(FLS))
dec = LabelEncoder().fit(FLS["device_id"])
row = dec.transform(FLS["device_id"])
col = LabelEncoder().fit_transform(FLS["feature"])
sparse_matrix = sparse.csr_matrix(
    (data, (row, col)), shape=(len(device_ids), len(feature_cs)))

sparse_matrix = sparse_matrix[:, sparse_matrix.getnnz(0) > 0]

################################################################################
# Data
################################################################################

train_row = dec.transform(train["device_id"])
train_sp = sparse_matrix[train_row, :]

test_row = dec.transform(test["device_id"])
test_sp = sparse_matrix[test_row, :]

X_train, X_val, y_train, y_val = train_test_split(
    train_sp, Y, train_size=.90, random_state=1234)

################################################################################
#   Feature Selection
################################################################################
print("# Feature Selection")
selector = SelectKBest(chi2, k=2)

selector.fit(X_train, y_train)

X_train = selector.transform(X_train)

X_val = selector.transform(X_val)

train_sp = selector.transform(train_sp)
test_sp = selector.transform(test_sp)

#================================================================================================================================
#Add your choice of machine learning algorithm here.......
#================================================================================================================================

#Cross validation step
print("Cross validation step...")

n_estimators=600

classifier = ExtraTreesClassifier(class_weight='auto', criterion='entropy',
            max_depth=2, max_features='sqrt', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,n_estimators=n_estimators, n_jobs=1)

classifier.fit(X_train, y_train)
y_predict = classifier.fit(X_train, y_train).predict_proba(X_val)

print("Classifier logloss=",log_loss(y_val, y_predict))

print("# Num of Features: ", X_train.shape[1])

#===================================================================================================================
#Random forest based predictions
#===================================================================================================================
print("Chosen classifier at work...")

classifier = ExtraTreesClassifier(class_weight='auto', criterion='entropy',
            max_depth=2, max_features='sqrt', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,n_estimators=n_estimators, n_jobs=1)

classifier.fit(train_sp, Y)

y_pre = classifier.fit(train_sp, Y).predict_proba(test_sp)


#========================================================================================================================
# Write and compress predictions
#========================================================================================================================

result = pd.DataFrame(y_pre, columns=lable_group.classes_)
result["device_id"] = device_id
result = result.set_index("device_id")
result.to_csv('fine_tune.gz', index=True,index_label='device_id', compression="gzip")

print("Done...thank you")




