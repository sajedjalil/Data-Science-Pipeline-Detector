#TODO: create the train,dev and test set
import time
import numpy as np
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
labels = ["target"]
# for label in train_df.columns:
#     # added other_disability because of the lack of sufficient training data
#     # for that particular category
#     if label not in ["id", "comment_text", "other_disability",
#                      "created_date", "publication_id",
#                      "parent_id", "article_id", "rating",
#                      "funny", "wow", "sad", "likes",
#                      "disagree", "identity_annotator_count",
#                      "toxicity_annotator_count"]:
#         labels.append(label)

print(train_df[labels].max(axis=1))
# train_df['none'] = 1 - train_df[labels].max(axis=1)
# print(train_df['none'].unique())
print(train_df[labels].describe())

# now we find the empty comments
# perhaps we have no null values in the comment column
comment = "comment_text"
print(train_df[comment].isna().sum())

# compiling the regular expression for pattern matching
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


def tokenize(s): return re_tok.sub(r' \1 ', s).split()


start_time = time.time()
n = train_df.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
                      min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                      smooth_idf=1, sublinear_tf=1)
train_doc = vec.fit_transform(train_df[comment])
test_doc = vec.transform(test_df[comment])
print("---%s seconds ---" % (time.time() - start_time))

print(type(train_doc))
print(type(test_doc))

x = train_doc
test_x = test_doc


# the basic naive-bayes feature equation
def pr(y_i, y):
    p = x[y == y_i].sum(0)
    return (p+1) / ((y == y_i).sum()+1)


# fit a model for one independent at a time
def get_mdl(y):
    y = y.values
    r = np.log(pr(1, y) / pr(0, y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


# the variable to store predictions
pred = np.zeros((len(test_df), len(labels)))
start_time = time.time()
for i, j in enumerate(labels):
    print('fit', j)
    t = (train_df[j] >= 0.5)*1
    m, r = get_mdl(t)
    print(test_x.shape)
    print(r.shape)
    pred[:, i] = m.predict_proba(test_x.multiply(r))[:, 1]
print("---%s seconds ---" % (time.time() - start_time))

# creating the submission file
# not_in_sub = []
# for label in labels:
#     if label != "target":
#         not_in_sub.append(label)


submid = pd.DataFrame({'id': test_df["id"]})
out_put = pd.concat([submid, pd.DataFrame(pred, columns=labels)], axis=1)
#submission = out_put.drop(not_in_sub, axis=1)
submission = out_put.rename(columns={"target": "prediction"})
submission["prediction"] = (submission["prediction"] >= 0.5)*1
submission.to_csv('submission.csv', index=False)


print("success")
