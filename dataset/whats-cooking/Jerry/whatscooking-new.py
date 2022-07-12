import pandas as pd
import numpy as np
food = pd.read_json("../input/train.json")
foodTest = pd.read_json("../input/test.json")
##Doing stemming
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
print(stemmer.stem("tomatoes"))

def stemming(string):
    s = ""
    string = string.split() 
    for e in string:
        s = s + stemmer.stem(e)+ " "
    return s

def listtostr(listring):
    s = ""
    for e in listring:
        s = s + e + " , "
    return s
    
print(listtostr(['a','b','c']))
print(stemming('tomatoes potatoes , potatoes , lovely'))

food['ingredientText'] = food['ingredients'].apply(lambda x: listtostr(x))     
foodTest['ingredientText'] = foodTest['ingredients'].apply(lambda x: listtostr(x)) 

import re
food['ingredientTextClean'] = food['ingredientText'].apply(lambda x: re.sub('[^a-zA-Z,]',' ',x))     
foodTest['ingredientTextClean'] = foodTest['ingredientText'].apply(lambda x: re.sub('[^a-zA-Z,]',' ',x)) 

food['ingredientTextClean'] = food['ingredientText'].apply(lambda x: stemming(x))     
foodTest['ingredientTextClean'] = foodTest['ingredientText'].apply(lambda x: stemming(x)) 

#print(food.head())
##Doing stemming
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
print(stemmer.stem("tomatoes"))

msk = np.random.rand(len(food)) < 0.8

train_data = food[msk]
test_data = food[~msk]
print(len(train_data))
print(len(test_data))

## Selecting target variable 
y_train = np.array(train_data['cuisine'])
y_test = np.array(test_data['cuisine'])

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range = (0,4), stop_words = 'english', token_pattern = "\w+", norm = 'l2', )
vectorizer.fit(food['ingredientTextClean'])
x_train = vectorizer.transform(train_data['ingredientTextClean'])
x_test = vectorizer.transform(test_data['ingredientTextClean'])
x_foodTest = vectorizer.transform(foodTest['ingredientTextClean'])


#from sklearn.grid_search import GridSearchCV

# Tuning the parameters of all the campaigns
from scipy import stats

#from sklearn.grid_search import  RandomizedSearchCV
from sklearn.svm import LinearSVC
#parameters = {'C': stats.expon(scale=10)}
clf = LinearSVC()
#clf = RandomizedSearchCV(svr, parameters)
clf.fit(x_train,y_train)
print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))





from sklearn.cross_validation import cross_val_score
print(cross_val_score(clf,x_test,y_test,cv=5))
foodTest['cuisine']=clf.predict(x_foodTest)

#foodTest[['id','ingredients','cuisine']].to_csv("submission.csv",index = False)
foodTest[['id','cuisine']].to_csv("submission.csv",index = False)