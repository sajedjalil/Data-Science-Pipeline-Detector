"""
What's Cooking : Tf Idf with One Vs Rest Support Vector Machine (SVM) Model

Goal: Use recipe ingredients to categorize the cuisine

SVM script for multiclass classification 

Input : Text Data (Ingredients for a Cusine)
Output : Single Class (Cusine Class)

author = sban (https://www.kaggle.com/shivamb)
created date = 26 June, 2018

edited by Don Woodlock

"""

# Import the required libraries 

random_state = 20180709
import time
starttime = time.monotonic()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


import pandas as pd
import json
import pdb


# Dataset Preparation
print ("Read Dataset ... ")
def read_dataset(path):
	return json.load(open(path, encoding='utf-8')) 
train = read_dataset('../input/train.json')
submission = read_dataset('../input/test.json')

# prepare X and y
target = [doc['cuisine'] for doc in train]
print ("Label Encode the Target Variable ... ")
lb = LabelEncoder()
y = lb.fit_transform(target)

X = train

X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                 test_size=0.001,
                 random_state=random_state)


# Text Data Features
print ("Prepare text data  ... ")
def generate_text(data):
	text_data = [" ".join(doc['ingredients']).lower() for doc in data]
	return text_data 

train_text = generate_text(X_train)
valid_text = generate_text(X_valid)
submission_text = generate_text(submission)

# Feature Engineering 
print ("TF-IDF on text data ... ")
tfidf = TfidfVectorizer(binary=True)
def tfidf_features(txt, flag):
    if flag == "train":
    	x = tfidf.fit_transform(txt)
    else:
	    x = tfidf.transform(txt)
    x = x.astype('float16')
    # x.sort_indices()
    return x 

X_train = tfidf_features(train_text, flag="train")
X_valid = tfidf_features(valid_text, flag="valid")
X_submission = tfidf_features(submission_text, flag="submission")

# Label Encoding - Target 

# Model Training 
# print ("Train the model ... ")
# classifier = SVC(C=1000, # penalty parameter, setting it to a larger value 
# 	 			 kernel='rbf', # kernel type, rbf working fine here
# 	 			 degree=3, # default value, not tuned yet
# 	 			 gamma=1, # kernel coefficient, not tuned yet
# 	 			 coef0=1, # change to 1 from default value of 0.0
# 	 			 shrinking=True, # using shrinking heuristics
# 	 			 tol=0.001, # stopping criterion tolerance 
# 	 			 probability=False, # no need to enable probability estimates
# 	 			 cache_size=200, # 200 MB cache size
# 	 			 class_weight=None, # all classes are treated equally 
# 	 			 verbose=False, # print the logs 
# 	 			 max_iter=-1, # no limit, let it run
# 	 			 decision_function_shape=None, # will use one vs rest explicitly 
# 	 			 random_state=random_state)
# model = OneVsRestClassifier(classifier, n_jobs=4)

print("features: ", X_train.shape)

parameters = {'C':[10, 100, 1000], 'gamma': [0.1, 1, 10], 'kernel': ['rbf', 'poly', 'linear']}
svc = SVC(kernel='rbf', # kernel type, rbf working fine here
	 			 degree=3, # default value, not tuned yet
	 			 coef0=1, # change to 1 from default value of 0.0
	 			 shrinking=True, # using shrinking heuristics
	 			 tol=0.01, # stopping criterion tolerance 
	 			 probability=False, # no need to enable probability estimates
	 			 cache_size=2000, # 200 MB cache size
	 			 class_weight=None, # all classes are treated equally 
	 			 verbose=False, # print the logs 
	 			 max_iter=-1, # no limit, let it run
	 			 decision_function_shape=None, # will use one vs rest explicitly 
	 			 random_state=random_state)
clf = RandomizedSearchCV(svc, param_distributions=parameters, verbose=10, n_jobs=4)
clf.fit(X_train, y_train)
print("best params: ", clf.best_params_)
with open("bestparams.txt", "w") as data:
	data.write(str(clf.best_params_))

# K-fold validation
# from sklearn.model_selection import StratifiedKFold

# kfold = StratifiedKFold(n_splits=10, random_state=random_state).split(X_train, y_train)
# scores = []
# for k, (train, test) in enumerate(kfold):
# 	training_data = X_train[train]
# 	training_data.sort_indices()
# 	model.fit(training_data, y_train[train])

# 	score = model.score(X_train[test], y_train[test])
# 	scores.append(score)
# 	print('Fold: %2d, Acc: %.3f' % (k+1, score))

# model.fit(X_train, y_train)
# print("Test the model on the validation data")
# y_pred_valid = model.predict(X_valid)
# print('Accuracy: %.4f' % accuracy_score(y_pred_valid, y_valid))

# # Predictions 
# print ("Predict on submision data ... ")
# y_submission = model.predict(X_submission)
# y_pred = lb.inverse_transform(y_submission)

# # Submission
# print ("Generate Submission File ... ")
# submission_id = [doc['id'] for doc in submission]
# sub = pd.DataFrame({'id': submission_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
# sub.to_csv('svm_output_' + str(random_state) + '.csv', index=False)

print("that took ", (time.monotonic()-starttime)/60, " minutes")