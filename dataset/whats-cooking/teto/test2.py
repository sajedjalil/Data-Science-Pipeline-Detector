
# -*- coding: utf-8 -*-
import os
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

def readData(filename):
	json_data = json.loads(open(os.path.join('../input/', filename)).read())
	return [' '.join(val['ingredients']) for val in json_data], [str(val['id']) for val in json_data],json_data
def output(y_pred, ids, fname):
    with open(fname, 'w') as f:
        f.write('id,cuisine\n')
        for i, y_class in zip(testIds,y_pred):
            f.write(','.join([i,y_class])+'\n')


ingredients, _,fullData = readData('train.json')
testData, testIds,_ = readData('test.json')
Countries = ([y['cuisine'] for y in fullData])


dic = CountVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english')
ingredients = dic.fit_transform(ingredients).toarray().astype(np.float32)
testData = dic.transform(testData).astype(np.float32)
lbl = LabelEncoder()
Countries = lbl.fit_transform([y['cuisine'] for y in fullData]).astype(np.int32)

logistic = linear_model.LogisticRegression()

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])


pca.fit(ingredients)
pipe.fit(ingredients, Countries)

pred = pipe.predict(testData)

output(pred, testIds,os.path.join('output.csv'))	
	