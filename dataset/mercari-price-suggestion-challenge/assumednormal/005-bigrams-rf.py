import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer

# read train and test data
train_df = pd.read_csv("../input/train.tsv", sep="\t")
test_df = pd.read_csv("../input/test.tsv", sep="\t")

# build term frequency matrix
vectorizer = CountVectorizer(ngram_range=(1, 2))
name_corpus = train_df.name.tolist()
X = vectorizer.fit_transform(name_corpus)

# append condition and shipping
X_to_append = csr_matrix(train_df[["item_condition_id", "shipping"]].values)
X = hstack([X, X_to_append])

# build random forest
y = np.log(train_df["price"]+1)
rf = RandomForestRegressor(n_estimators=500,
                           criterion="mse",
                           max_features="sqrt",
                           max_depth=10,
                           n_jobs=4,
                           verbose=10)
rf.fit(X, y)

# score test data
submission_df = test_df[["test_id"]].copy()

X_test = vectorizer.transform(test_df.name.tolist())
X_test_to_append = csr_matrix(test_df[["item_condition_id", "shipping"]].values)
X_test = hstack([X_test, X_test_to_append])

submission_df["price"] = np.exp(rf.predict(X_test))-1

submission_df.to_csv("005_bigrams_rf.csv", index=False)