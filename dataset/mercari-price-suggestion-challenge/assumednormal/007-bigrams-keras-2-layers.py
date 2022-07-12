import numpy as np
import pandas as pd

from collections import Counter
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer

# read train and test data
train_df = pd.read_csv("../input/train.tsv", sep="\t")
test_df = pd.read_csv("../input/test.tsv", sep="\t")

# build term frequency matrix
vectorizer = CountVectorizer(ngram_range=(1, 2))
name_corpus = train_df.name.tolist()
X = vectorizer.fit_transform(name_corpus)

# remove columns with <= 1% non-zero entries (arbitrary choice so that notebook doesn't crash)
n_docs = Counter(X.nonzero()[1])
cols_to_keep = [k for k, v in n_docs.items() if v > .01 * X.shape[0]]
X_filtered = X[:, cols_to_keep].todense()

# append condition and shipping
X_filtered = hstack([X_filtered, csr_matrix(train_df[["item_condition_id", "shipping"]].values)])

# define y
y = np.log(train_df["price"]+1)

# model
def baseline_model():
    model = Sequential()
    model.add(Dense(10, input_dim=59, kernel_initializer="normal", activation="relu"))
    model.add(Dense(5, kernel_initializer="normal", activation="relu"))
    model.add(Dense(1, kernel_initializer="normal"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model
    
estimator = KerasRegressor(build_fn=baseline_model, verbose=1)
estimator.fit(X_filtered.toarray(), y)

# score test data
submission_df = test_df[["test_id"]].copy()

X_test = vectorizer.transform(test_df.name.tolist())
X_test_filtered = X_test[:, cols_to_keep].todense()
X_test_filtered = hstack([X_test_filtered, csr_matrix(test_df[["item_condition_id", "shipping"]].values)])

submission_df["price"] = np.exp(estimator.predict(X_test_filtered.toarray()))-1

submission_df.to_csv("007_bigrams_keras_2_layers.csv", index=False)