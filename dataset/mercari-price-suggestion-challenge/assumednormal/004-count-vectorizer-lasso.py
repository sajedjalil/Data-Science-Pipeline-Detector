import numpy as np
import pandas as pd

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("../input/train.tsv", sep="\t")
test_df = pd.read_csv("../input/test.tsv", sep="\t")

# build term frequency matrix
vectorizer = CountVectorizer()
name_corpus = train_df.name.tolist()
X = vectorizer.fit_transform(name_corpus)

# remove columns with <= 1% non-zero entries (arbitrary choice so that notebook doesn't crash)
n_docs = Counter(X.nonzero()[1])
cols_to_keep = [k for k, v in n_docs.items() if v > .01 * X.shape[0]]
X_filtered = X[:, cols_to_keep].todense()

# standardize columns
scaler = StandardScaler()
X_filtered_scaled = scaler.fit_transform(X_filtered)

# fit lasso
y = train_df["price"]
lasso = LassoCV(cv=10)
lasso.fit(X_filtered_scaled, y)

# score test data
submission_df = test_df[["test_id"]].copy()

X_test = vectorizer.transform(test_df.name.tolist())
X_test_filtered = X_test[:, cols_to_keep].todense()
X_test_filtered_scaled = scaler.transform(X_test_filtered)

submission_df["price"] = lasso.predict(X_test_filtered_scaled)
submission_df["price"] = submission_df["price"].apply(lambda x: x if x > 0 else 0)

submission_df.to_csv("004_count_vectorizer_lasso.csv", index=False)