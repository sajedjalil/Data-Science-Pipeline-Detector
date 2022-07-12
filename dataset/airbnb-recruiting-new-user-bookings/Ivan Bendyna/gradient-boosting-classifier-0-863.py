import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from math import log

CV = False
cv_size = 0.1


def shuffle(df):
    # noinspection PyUnresolvedReferences
    return df.iloc[np.random.permutation(len(df))]


def ndcg(array, value):
    indices = np.where(array == value)[0]
    dcg = 0
    idcg = 0
    for k in range(len(indices)):
        dcg += log(2) / log(2 + indices[k])
        idcg += log(2) / log(2 + k)
    if dcg == 0:
        assert idcg == 0
        return 0
    return dcg / idcg


train_data = shuffle(pd.read_csv('../input/train_users_2.csv'))
if CV:
    train_data, test_data = train_test_split(train_data, test_size=cv_size)
    cv_answer = test_data['country_destination'].values
    test_data.drop(['country_destination'], axis=1, inplace=True)
else:
    test_data = pd.read_csv('../input/test_users.csv')
train_answer = train_data['country_destination'].values
train_data.drop(['country_destination'], axis=1, inplace=True)
test_id = test_data["id"]
test_id.reset_index(drop=True, inplace=True)

train_size = train_data.shape[0]
all_data = pd.concat((train_data, test_data), axis=0, ignore_index=True)
all_data = all_data.drop(['id', 'timestamp_first_active', 'date_first_booking', 'date_account_created'], axis=1)

label_encoder_features = ["gender", "signup_method", "signup_flow", "language", "affiliate_channel",
                          "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type",
                          "first_browser"]

for feature in label_encoder_features:
    all_data[feature] = LabelEncoder().fit_transform(all_data[feature])

le_answer = LabelEncoder()
le_answer.fit_transform(train_answer)

all_data.fillna(-1, inplace=True)

classifier = GradientBoostingClassifier(n_estimators=5, verbose=True)
classifier.fit(all_data[:train_size], train_answer)
prob_answer = classifier.predict_proba(all_data[train_size:])

answer = []
index = 0
for prob in prob_answer:
    one_answer = [test_id[index]]
    prob_list = le_answer.inverse_transform(np.argsort(prob)[::-1][:5])
    one_answer.append(prob_list)
    answer.append(one_answer)
    index += 1

if CV:
    # noinspection PyUnboundLocalVariable
    assert len(cv_answer) == len(answer)
    sum_ndcg = 0
    for i in range(len(answer)):
        sum_ndcg += ndcg(answer[i][1], cv_answer[i])
    print(sum_ndcg / len(answer))
else:
    ids = []
    countries = []
    for i in range(len(answer)):
        for j in range(5):
            ids.append(answer[i][0])
            countries.append(answer[i][1][j])
    submission = pd.DataFrame(np.column_stack((ids, countries)), columns=['id', 'country'])
    submission.to_csv('submission.csv', index=False)