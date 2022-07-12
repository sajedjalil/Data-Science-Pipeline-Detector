# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from __future__ import division
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

print('reading data...')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Subsample because currently can't run the whole thing
train['book_yr'] = train.date_time.map(lambda x: x[:4])
train = train[train.book_yr == '2014']
train = train[train.is_booking] # unfortunately


def featurize(df, cv, is_train):
    # df['has_children'] = map(lambda x: 'haschildren_'+str(x), df.srch_children_cnt > 0)
    # df['is_single'] = map(lambda x:'single_'+str(x), (df.srch_adults_cnt==1) & (df.srch_children_cnt==0))
    # df['is_couple'] = map(lambda x:'couple_'+str(x), (df.srch_adults_cnt==2) & (df.srch_children_cnt==0))
    df['main_feat'] = map(lambda x:str(x).replace(", ", "x"), zip(df.user_location_country, df.srch_destination_id, df.hotel_continent, df.hotel_country))
    df['main_feat2'] = map(lambda x:str(x).replace(", ", "x"), zip(df.user_location_country, df.srch_destination_id, df.hotel_continent, df.hotel_country, df.hotel_market))
    df['main_feat3'] = map(lambda x:str(x).replace(", ", "x"), zip(df.user_location_region, df.srch_destination_id, df.hotel_continent, df.hotel_country))
    df['main_feat4'] = map(lambda x:str(x).replace(", ", "x"), zip(df.user_location_city, df.srch_destination_id, df.hotel_continent, df.hotel_country))
    df['main_feat5'] = map(lambda x:str(x).replace(", ", "x"), zip(df.user_id, df.srch_destination_id))
    df['main_feat6'] = map(lambda x:str(x).replace(", ", "x"), zip(df.user_id, df.hotel_market))
    df['tokenized'] = map(lambda x:' '.join(map(str,x)), zip(df.main_feat, df.main_feat2, df.hotel_market, df.main_feat5, df.user_id, df.main_feat6))
    if is_train:
        feat = cv.fit_transform(df['tokenized'])
    else:
        feat = cv.transform(df['tokenized'])
    
    return feat #sparse.csr_matrix

print('featurizing...')
cv = CountVectorizer(max_features=10000000)
Xtrain = featurize(train, cv, is_train=True)
Ytrain = train.hotel_cluster
Xtest = featurize(test, cv, is_train=False)

print('fitting naive bayes')
clf = MultinomialNB(alpha = 0.07)
clf.fit(Xtrain, Ytrain, sample_weight = 0.1 + 0.5*train.is_booking)

# Predict
pred = clf.predict_proba(Xtest)
pred_rank = np.apply_along_axis(lambda x: np.argsort(-x)[:5], 1, pred)

# build data leak lkp
def build_id_lkp():
    user_event_clusters = defaultdict(set)
    test_id_lkp = defaultdict(str)
    for i, row in enumerate(DictReader(open("data/train.csv"))):
        if i%1000000 == 0:
            print(i)
        user_event_key = (row["user_location_country"], 
                          row["user_location_region"], 
                          row["user_location_city"],
                          row["hotel_market"],
                          row["orig_destination_distance"])
        user_event_clusters[user_event_key].add(row["hotel_cluster"])

    for i, row in enumerate(DictReader(open("data/test.csv"))):
        if i%1000000 == 0:
            print(i)
        user_event_key = (row["user_location_country"], 
                          row["user_location_region"], 
                          row["user_location_city"],
                          row["hotel_market"],
                          row["orig_destination_distance"])
        if user_event_key in user_event_clusters:
            pred_str = " ".join(list(user_event_clusters[user_event_key])[:5])
            if pred_str:
                test_id_lkp[i] = pred_str
    print(len(test_id_lkp))
    return test_id_lkp

test_id_lkp = build_id_lkp()


print('writing submission...')
lkp_ct = 0
with open('nb_submission2.csv', 'w') as f:
    f.write("id,hotel_cluster\n")
    for i,row in enumerate(pred_rank):
        if i%1000000==0:
            print(i)
        if i in test_id_lkp:
            lkp_ct += 1
            f.write("%d,%s\n"%(i, test_id_lkp[i]))
            # print test_id_lkp[i], pred_rank[i]
        else:
            f.write("%d,%s\n"%(i, ' '.join(map(str,pred_rank[i]))))
print(lkp_ct)







from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.