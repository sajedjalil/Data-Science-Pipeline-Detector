import numpy as np
import pandas as pd
from scipy import sparse
import random
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

train_df = pd.read_json("../input/train.json")
test_df = pd.read_json("../input/test.json")

#basic features
#train_df.loc[train_df.loc[:,"bedrooms"] == 0, "bedrooms"] = 1
#train_df["price_t"] =train_df["price"]/train_df["bedrooms"]
#test_df.loc[test_df.loc[:,"bedrooms"] == 0, "bedrooms"] = 1
#test_df["price_t"] = test_df["price"]/test_df["bedrooms"] 

train_df["room_sum"] = train_df["bedrooms"]+train_df["bathrooms"] 
test_df["room_sum"] = test_df["bedrooms"]+test_df["bathrooms"] 

# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

# create time, interval time from since the list created
train_df["created"] = pd.to_datetime(train_df["created"])
train_df["passed"] = train_df["created"].max() - train_df["created"]
train_df["passed"] = train_df["passed"].dt.days

test_df["created"] = pd.to_datetime(test_df["created"])
test_df["passed"] = test_df["created"].max() - test_df["created"]
test_df["passed"] = test_df["passed"].dt.days

train_df["created_year"] = train_df["created"].dt.year
test_df["created_year"] = test_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
test_df["created_month"] = test_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day
test_df["created_day"] = test_df["created"].dt.day
train_df["created_hour"] = train_df["created"].dt.hour
test_df["created_hour"] = test_df["created"].dt.hour

features_to_use=["bathrooms", "bedrooms", "latitude", "longitude", "price",
"num_photos", "num_features", "num_description_words","listing_id", "created_year", "created_month", "created_day", "created_hour"]

#using cross valdation to compute the posterier prob (P(y = low/medium/high|x_manager))
#in the barreca's paper we know, thatcount(x_manager) maybe too small to give a credencial probability, thus we could combine the prior probability
#we may use that here
#and we could see if count(x_manager) is nan, the prob = 0 here, however we may use the prior here
index=list(range(train_df.shape[0]))
random.shuffle(index)
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)

for i in range(5):
    building_level={}
    for j in train_df['manager_id'].values:
        building_level[j]=[0,0,0]
    
    #select the fifth part as the validation set, and the other as the train set
    test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    
    #sum up the count of each level for a specific manager
    for j in train_index:
        temp=train_df.iloc[j]
        if temp['interest_level']=='low':
            building_level[temp['manager_id']][0]+=1
        if temp['interest_level']=='medium':
            building_level[temp['manager_id']][1]+=1
        if temp['interest_level']=='high':
            building_level[temp['manager_id']][2]+=1
            
    for j in test_index:
        temp=train_df.iloc[j]
        if sum(building_level[temp['manager_id']])!=0:
            a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])
            b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])
            c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])
            
train_df['manager_level_low']=a
train_df['manager_level_medium']=b
train_df['manager_level_high']=c


#here is too compute prior in the trainset as as estimate of posterier prob in the test set.
#if there is manager_id not found in the train_set, we use nan for the prob
a=[]
b=[]
c=[]
building_level={}
for j in train_df['manager_id'].values:
    building_level[j]=[0,0,0]

for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    if temp['interest_level']=='low':
        building_level[temp['manager_id']][0]+=1
    if temp['interest_level']=='medium':
        building_level[temp['manager_id']][1]+=1
    if temp['interest_level']=='high':
        building_level[temp['manager_id']][2]+=1

for i in test_df['manager_id'].values:
    if i not in building_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(building_level[i][0]*1.0/sum(building_level[i]))
        b.append(building_level[i][1]*1.0/sum(building_level[i]))
        c.append(building_level[i][2]*1.0/sum(building_level[i]))
test_df['manager_level_low']=a
test_df['manager_level_medium']=b
test_df['manager_level_high']=c

features_to_use.append('manager_level_low') 
features_to_use.append('manager_level_medium') 
features_to_use.append('manager_level_high')

# treat some missing values
train_df['manager_level_low'] = train_df['manager_level_low'].fillna(0)
train_df['manager_level_medium'] = train_df['manager_level_medium'].fillna(0)
train_df['manager_level_high'] = train_df['manager_level_high'].fillna(0)

test_df['manager_level_low'] = test_df['manager_level_low'].fillna(0)
test_df['manager_level_medium'] = test_df['manager_level_medium'].fillna(0)
test_df['manager_level_high'] = test_df['manager_level_high'].fillna(0)



#transfer the categorical varibles to label integer
categorical = ["display_address", "manager_id", "building_id"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)

#transfer features to bag of word and using tdidf to normalizing the word-count
#the tdidf transformation is what we haven't done in version 1, maybe that would improve performance
#and the tokens we chose here are the top 200, which is larger than version 1
train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
print(train_df["features"].head())
tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])


train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape)

# split data
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.33)

# AdaBoost Classifer
#ada = ensemble.AdaBoostClassifier()
rf = ensemble.RandomForestClassifier(n_estimators=1000)

# model training
#ada.fit(X_train, y_train)
rf.fit(X_train, y_train)

# get class probabilities
y_val_pred = rf.predict_proba(X_val)

# calculate log loss
print(log_loss(y_val, y_val_pred))


rf.fit(train_X, train_y)
preds = rf.predict_proba(test_X)

sub = pd.DataFrame(preds)
sub.columns = ["high", "medium", "low"]
sub["listing_id"] = test_df.listing_id.values
sub.to_csv("rf_sub.csv", index=False)







