from pandas import DataFrame
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

traindf = pd.read_json("../input/train.json")
testdf = pd.read_json("../input/test.json")
wnl = WordNetLemmatizer()
def lemmatize_each_row(x):
    y = []
    for each in x:
        y.append(wnl.lemmatize(each.lower()))
    return y

traindf['lemmatized_ingredients_list'] = traindf.apply(lambda row: lemmatize_each_row(row['ingredients']), axis=1)
all_ingredients_lemmatized = []
for ingredients_lists in traindf.ingredients:
    for ingredient in ingredients_lists:
        all_ingredients_lemmatized.append(wnl.lemmatize(ingredient.lower()))
all_ingredients_lemmatized = set(all_ingredients_lemmatized)
testdf['lemmatized_test_ingredients_list'] = testdf.apply(lambda row: lemmatize_each_row(row['ingredients']), axis=1)
all_ingredients_lemmatized_test = []
for ingredients_lists in testdf.ingredients:
    for ingredient in ingredients_lists:
        all_ingredients_lemmatized_test.append(wnl.lemmatize(ingredient.lower()))
all_ingredients_lemmatized_test = set(all_ingredients_lemmatized_test)

all_ingredients_union = all_ingredients_lemmatized | all_ingredients_lemmatized_test



vect = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, sublinear_tf=True, max_df=0.5, analyzer='word', stop_words='english', vocabulary=all_ingredients_union)
tfidf_matrix = vect.fit_transform(traindf['lemmatized_ingredients_list'])
predictor_matrix = tfidf_matrix
target_classes = traindf['cuisine']

vect_test = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, sublinear_tf=True, max_df=0.5, analyzer='word', stop_words='english', vocabulary=all_ingredients_union)
tfidf_matrix_test = vect_test.fit_transform(testdf['lemmatized_test_ingredients_list'])
predictor_matrix_test = tfidf_matrix_test

from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss="squared_hinge", penalty="l2", n_iter=10)
clf.fit(predictor_matrix, target_classes)

predicted_classes = clf.predict(predictor_matrix_test)

testdf['cuisine'] = predicted_classes
submission=testdf[['id' ,  'cuisine' ]]
submission.to_csv("SGDSubmission.csv",index=False)
