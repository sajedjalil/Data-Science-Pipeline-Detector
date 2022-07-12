from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, VotingClassifier ,BaggingClassifier ,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import codecs
import os
from sklearn.svm import SVC
#from mlxtend.classifier import EnsembleClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import FeatureUnion
import json as js
import csv as csv
import scipy as scipy
import numpy as np
import pdb
from sklearn.neural_network import MLPClassifier



from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, VotingClassifier ,BaggingClassifier ,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import codecs
import os
from sklearn.svm import SVC

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import FeatureUnion
import xgboost as xgb

from collections import Counter
from operator import itemgetter


num_values = 9340

traindf = pd.read_json(codecs.open("../input/train.json",'r','utf-8') )#[:num_values]
print("read tran")
testdf = pd.read_json(codecs.open("../input/test.json",'r','utf-8') )
print("read test")

data = {'id':[],'cuisine':[],'ingredients':[]}
newTrain = pd.DataFrame(data)

trashhold = 2646
ingredients = []

for ingredient in traindf['ingredients']:
    ingredients.extend(ingredient)

c = Counter(ingredients)

unique_ingredients_counter = sorted(c.items(),key=itemgetter(1), reverse=False)

print("prepocess")
unique_cuisines_list = ['brazilian','russian','jamaican','irish','filipino','british','moroccan','vietnamese','korean','spanish','greek','japanese','thai','cajun_creole','french','chinese','indian','southern_us','mexican','italian']
cuisines_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

def getIndex(value):
    return unique_cuisines_list.index(value) - 1

def isInthelist(container ,value):
    boolList = []
    #print(container)
    #print(value)
    for item in container:
        #print(value)
        #print(item)
        if value in item:
            boolList.append(True)
        else:
            boolList.append(False)
    #print(boolList)
    return boolList
    
def getboolListTrueCount(l):
    count=0
    for item in l:
        if item == True:
            count+=1
    return count

def OrList(l1,l2):
    for i in range(0,len(l1),1):
        if l2[i] and (not l1[i]):
            index = getIndex(traindf["cuisine"].loc[i])
            #print(cuisines_count[index])
            if cuisines_count[index] < trashhold:
                l1[i] = l2[i]
                cuisines_count[index] +=1
    return l1;

boolList = isInthelist(traindf["ingredients"],"");

for z in unique_ingredients_counter:
    boolList = OrList(boolList ,isInthelist(traindf["ingredients"],z[0]))
    num = getboolListTrueCount(boolList)
    if num > (trashhold *20):
        break
newTrain = traindf[boolList]
traindf = newTrain
print("done")



traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]

traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

#traindf.to_csv("traindf.csv", index=False,sep='\t', encoding='utf-8')


testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       


corpustr = traindf['ingredients_string']


estimators = [("tfidf", TfidfVectorizer(stop_words='english', ngram_range = ( 1 , 1 ),analyzer="word", max_df = .57,
                                        binary=False ,max_features =6706, token_pattern=r'\w+' , sublinear_tf=False)),
              ("hash", HashingVectorizer (stop_words='english', ngram_range = ( 1 , 2 ),n_features  =6706, analyzer="word",token_pattern=r'\w+'
                                           , binary =False))]
print("tfidftr FeatureUnion")
tfidftr = FeatureUnion(estimators, transformer_weights={'tfidf': 4,'hash': 5}).fit_transform(corpustr).todense()

#tfidftr = tfidftr.todense();

corpusts = testdf['ingredients_string']

print("tfidfts FeatureUnion")
tfidfts = FeatureUnion(estimators, transformer_weights={'tfidf': 4,'hash': 5}).transform(corpusts)


predictors_tr = tfidftr

targets_tr = traindf['cuisine']

predictors_ts = tfidfts



"""
with open('../input/train.json') as json_data:
    data = js.load(json_data)
    json_data.close()
    
with open('../input/test.json') as json_data:
    test = js.load(json_data)
    json_data.close()
"""

"""
classes = [item['cuisine'] for item in data]
ingredients = [item['ingredients'] for item in data]
unique_ingredients = set(item for sublist in ingredients for item in sublist)
unique_cuisines = set(classes)

testIngredients = [item['ingredients'] for item in test]
"""

#big_data_matrix = scipy.sparse.dok_matrix((len(ingredients), len(unique_ingredients)), dtype=np.dtype(bool))
"""
for d,dish in enumerate(ingredients):
    for i,ingredient in enumerate(unique_ingredients):
        if ingredient in dish:
            big_data_matrix[d,i] = True
print("train")
#print(big_data_matrix)
"""

clf = MLPClassifier (algorithm = 'adam', alpha=0.001, hidden_layer_sizes=(100,100,100), random_state=1, activation='logistic' ,early_stopping = True);
#classMLP = clf.fit(big_data_matrix, classes)

"""
big_test_matrix = scipy.sparse.dok_matrix((len(testIngredients), len(unique_ingredients)), dtype=np.dtype(bool))
for d,dish in enumerate(testIngredients):
    for i,ingredient in enumerate(unique_ingredients):
        if ingredient in dish:
            big_test_matrix[d,i] = True
print("test")
print(big_test_matrix)
"""
#

#predictions = classMLP.predict(big_test_matrix) 

#print(predictions)


#print("predict finish")
"""
result = [(ref == res, ref, res) for (ref, res) in zip(classes, classMLP.predict(big_data_matrix))]
accuracy_learn = sum (r[0] for r in result) / float ( len(result) )

print('Accuracy on the learning set: ', accuracy_learn)
"""



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