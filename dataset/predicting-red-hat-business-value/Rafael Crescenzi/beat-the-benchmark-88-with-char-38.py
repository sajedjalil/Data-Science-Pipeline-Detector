import pandas as pd
from sklearn.metrics import roc_auc_score

data_path  = "../input/"

people = pd.read_csv(data_path + "people.csv",
                     index_col=0, usecols=["people_id", "char_38"])
train = pd.read_csv(data_path + "act_train.csv",
                    index_col=0, usecols=["people_id", "outcome"])
test = pd.read_csv(data_path + "act_test.csv",
                   index_col=0, usecols=["people_id", "activity_id"])

train = train.join(people)
print("AUC in train:", roc_auc_score(train.outcome, train.char_38))

test.join(people).set_index("activity_id").\
    rename(columns={"char_38": "outcome"}).\
    to_csv("BB_char_38.csv", header=True)