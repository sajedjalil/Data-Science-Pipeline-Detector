from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%

traindf = pd.read_json("../input/train.json")
print("read train")

traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]

traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]

#traindf.to_csv("traindf.csv", index=False,sep='\t', encoding='utf-8')

testdf = pd.read_json("../input/test.json")
print("read test")
testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]

#testdf.to_csv("testdf.csv", index=False)


corpustr = traindf['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word",
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
tfidftr = vectorizertr.fit_transform(corpustr).todense()
print("tfidft tran")
corpusts = testdf['ingredients_string']
vectorizerts = TfidfVectorizer(stop_words='english')
tfidfts = vectorizertr.transform(corpusts)
print("tfidft test")


predictors_tr = tfidftr

targets_tr = traindf['cuisine']

predictors_ts = tfidfts


#classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
parameters = {'C':[1, 10]}

lsvc = LinearSVC()
lr = LogisticRegression()

lsvcClassifier = grid_search.GridSearchCV(lsvc, parameters)
print("lsvc Classifier parameters")
lsvcClassifier = lsvcClassifier.fit(predictors_tr,targets_tr)
print("lsvc Classifier fit")
#predictions = lsvcClassifier.predict(predictors_ts)

lrClassifier = grid_search.GridSearchCV(lr, parameters)
print("lr Classifier parameters")
lrClassifier = lrClassifier.fit(predictors_tr,targets_tr)
print("lr Classifier fit")
#predictions = lrClassifier.predict(predictors_ts)

#clf = VotingClassifier(estimators = [('lsvc', lsvcClassifier), ('lr', lrClassifier)], voting='soft', weights=[0.78902,0.78862])
clf = VotingClassifier(estimators = [('lsvc', lsvcClassifier), ('lr', lrClassifier)], voting='hard')
print("Voting parameters")
clf.fit(predictors_tr,targets_tr)
print("Voting fit")
predictions = clf.predict(predictors_ts)
print("Voting predict")

testdf['cuisine'] = predictions
testdf = testdf.sort('id' , ascending=True)
print("sort by id")

#show the detail
#testdf[['id' , 'ingredients_clean_string' , 'cuisine' ]].to_csv("submission.csv")

#for submit, no index
testdf[['id' , 'cuisine' ]].to_csv("submission.csv", index=False)
print("done")
