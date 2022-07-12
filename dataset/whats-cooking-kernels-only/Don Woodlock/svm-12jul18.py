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

random_state = None
import time
starttime = time.monotonic()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit
from scipy.sparse import hstack, csr_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler


import pandas as pd
import json
import pdb


# Dataset Preparation
print ("Read Dataset ... ")

def read_dataset(path):
	return json.load(open(path, encoding='utf-8')) 

train = read_dataset('../input/train.json')
train_df = pd.read_json('../input/train.json')
train_df = train_df.set_index('id')
train_df.drop("cuisine", axis=1, inplace=True)
traindex = train_df.index
submission = read_dataset('../input/test.json')
submission_df = pd.read_json('../input/test.json')
submission_df = submission_df.set_index('id')
submissiondex = submission_df.index

print("Combine Train and Submission")
df = pd.concat([train_df,submission_df],axis=0)
del train_df, submission_df


print("adding features")
# df["num_ingredients"] = df['ingredients'].apply(lambda x: len(x))
# def last_word(str):
# 	array = str.split()
# 	l = len(array)
# 	return array[l-1]
# df["ingredient1"] = df['ingredients'].apply(lambda x: x[0])
# pdb.set_trace()
# df["ingredient1"] = df['ingredient1'].apply(last_word)
# df = pd.get_dummies(df['ingredient1'])






# prepare X and y
target = [doc['cuisine'] for doc in train]
print ("Label Encode the Target Variable ... ")
lb = LabelEncoder()
y = lb.fit_transform(target)

X = train

X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                 test_size=0.0001,
                 random_state=random_state)

X_train_features = df.loc[pd.DataFrame(X_train)['id']].drop(["ingredients"], axis=1)
X_valid_features = df.loc[pd.DataFrame(X_valid)['id']].drop(["ingredients"], axis=1)

submission_features = df.loc[submissiondex, :].drop(["ingredients"], axis=1)

print("scaling numeric features")
# scaler = StandardScaler()
# pdb.set_trace()
# scaler.fit(X_train_features[['num_ingredients']].values)
# df[['num_ingredients']] = scaler.transform(df[['num_ingredients']].values)


# Text Data Features
print ("Prepare text data  ... ")
def last_word(sentence):
	array = sentence.split()
	l = len(array)
	lw = array[l-1]
	return lw

def generate_text(data):
	def convert(doc):
		ingredients = doc['ingredients']
		# ingredients2 = map(last_word, ingredients)
		return " ".join(ingredients).lower()
	text_data = [convert(doc) for doc in data]
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

print("combine other features with text vectors")
X_train = tfidf_features(train_text, flag="train")
X_train = hstack((X_train,np.array(X_train_features.values)))

X_valid = tfidf_features(valid_text, flag="valid")
X_valid = hstack((X_valid,np.array(X_valid_features.values)))

X_submission = tfidf_features(submission_text, flag="submission")
X_submission = hstack((X_submission,np.array(submission_features.values)))

# Label Encoding - Target 

# Model Training 
print ("Train the model ... ")
classifier = SVC(C=50, # penalty parameter, setting it to a larger value 
	 			 kernel='rbf', # kernel type, rbf working fine here
	 			 degree=3, # default value, not tuned yet
	 			 gamma=1.4, # kernel coefficient, not tuned yet
	 			 coef0=1, # change to 1 from default value of 0.0
	 			 shrinking=True, # using shrinking heuristics
	 			 tol=0.001, # stopping criterion tolerance 
	 			 probability=False, # no need to enable probability estimates
	 			 cache_size=200, # 200 MB cache size
	 			 class_weight=None, # all classes are treated equally 
	 			 verbose=False, # print the logs 
	 			 max_iter=-1, # no limit, let it run
	 			 decision_function_shape=None, # will use one vs rest explicitly 
	 			 random_state=random_state)
# classifier = SVC(C=10, kernel='rbf', degree=3, gamma=1, tol=0.001, max_iter=-1, random_state=random_state)
model = OneVsRestClassifier(classifier, n_jobs=4)

print("features: ", X_train.shape)

# C_range = np.array([50, 100, 150, 250, 500])
# gamma_range = np.array([1.4])
# param_grid = dict(gamma=gamma_range, C=C_range)
# cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=random_state)
# grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv, verbose=10, n_jobs=8)
# grid.fit(X_train, y_train)

# print("The best parameters are %s with a score of %0.4f"
#       % (grid.best_params_, grid.best_score_))

# parameters = {'C':[10, 100, 1000], 'gamma': [0.1, 1, 10], 'kernel': ['rbf']}
# svc = SVC(kernel='rbf', # kernel type, rbf working fine here
# 	 			 degree=3, # default value, not tuned yet
# 	 			 coef0=1, # change to 1 from default value of 0.0
# 	 			 shrinking=True, # using shrinking heuristics
# 	 			 tol=0.01, # stopping criterion tolerance 
# 	 			 probability=False, # no need to enable probability estimates
# 	 			 cache_size=2000, # 200 MB cache size
# 	 			 class_weight=None, # all classes are treated equally 
# 	 			 verbose=False, # print the logs 
# 	 			 max_iter=-1, # no limit, let it run
# 	 			 decision_function_shape=None, # will use one vs rest explicitly 
# 	 			 random_state=random_state)
# clf = RandomizedSearchCV(svc, param_distributions=parameters, verbose=10, n_jobs=4, n_iter=1)
# clf.fit(X_train, y_train)
# print("best params: ", clf.best_params_)
# with open("bestparams.txt", "w") as data:
# 	data.write(str(clf.best_params_))

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

model.fit(X_train, y_train)
print("Test the model on the validation data")
y_pred_valid = model.predict(X_valid)
print('Accuracy: %.4f' % accuracy_score(y_pred_valid, y_valid))

# Predictions 
print ("Predict on submision data ... ")
y_submission = model.predict(X_submission)
y_pred = lb.inverse_transform(y_submission)

# Submission
print ("Generate Submission File ... ")
submission_id = [doc['id'] for doc in submission]
sub = pd.DataFrame({'id': submission_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('svm_output_' + str(random_state) + '.csv', index=False)

print("that took ", (time.monotonic()-starttime)/60, " minutes")