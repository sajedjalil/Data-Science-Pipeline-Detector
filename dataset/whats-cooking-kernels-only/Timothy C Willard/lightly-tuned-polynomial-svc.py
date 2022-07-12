import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_json("../input/train.json").set_index("id")
test_df = pd.read_json("../input/test.json").set_index("id")

le = LabelEncoder()
train_df["cuisine"] = le.fit_transform(train_df["cuisine"])

tfidf_vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = lambda x: x, token_pattern = None, preprocessor = lambda x: x)
tfidf_train = tfidf_vectorizer.fit_transform(train_df["ingredients"])
tfidf_test = tfidf_vectorizer.transform(test_df["ingredients"])

svc_clf = SVC(C = 0.2933391008208308, kernel = "poly", degree = 3, gamma = 1, coef0 = 1, shrinking = True, tol = 0.0001, probability = False, cache_size = 200, class_weight = None, verbose = False, max_iter = -1, decision_function_shape = None, random_state = None)
svc_clf.fit(tfidf_train, train_df["cuisine"])

pred_test = svc_clf.predict(tfidf_test)
test_df["cuisine"] = le.inverse_transform(pred_test)
test_df.drop(columns = ["ingredients"], inplace = True)
test_df.to_csv("lightly_tuned_polynomial_svc_submission.csv")