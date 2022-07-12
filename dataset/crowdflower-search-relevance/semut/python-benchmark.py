import nltk
import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


train = pd.read_csv("../input/train.csv").fillna("")
test  = pd.read_csv("../input/test.csv").fillna("")

class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for feature_name, column_name, extractor in self.features:
            extractor.fit(X[column_name], y)

    def transform(self, X):
        extracted = []
        for feature_name, column_name, extractor in self.features:
            fea = extractor.transform(X[column_name])
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

    def fit_transform(self, X, y=None):
        extracted = []
        for feature_name, column_name, extractor in self.features:
            fea = extractor.fit_transform(X[column_name], y)
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

def identity(x):
    return x

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T

#                          Feature Set Name            Data Frame Column              Transformer
features = FeatureMapper([('QueryBagOfWords',          'query',                       CountVectorizer(max_features=200)),
                          ('TitleBagOfWords',          'product_title',               CountVectorizer(max_features=200)),
                          ('DescriptionBagOfWords',    'product_description',         CountVectorizer(max_features=200)),
                          ('QueryTokensInTitle',       'query_tokens_in_title',       SimpleTransform()),
                          ('QueryTokensInDescription', 'query_tokens_in_description', SimpleTransform()),
                          ('TitleTokensInQuery',       'title_tokens_in_query',       SimpleTransform()),
                          ('DescriptionTokensInQuery', 'description_tokens_in_query', SimpleTransform())])

def extract_features(data):
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    data["query_tokens_in_title"] = 0.0
    data["query_tokens_in_description"] = 0.0
    data["title_tokens_in_query"] = 0.0
    data["description_tokens_in_query"] = 0.0
    for i, row in data.iterrows():
        query = set(x.lower() for x in token_pattern.findall(row["query"]))
        title = set(x.lower() for x in token_pattern.findall(row["product_title"]))
        description = set(x.lower() for x in token_pattern.findall(row["product_description"]))
        if len(title) > 0:
            data.set_value(i, "query_tokens_in_title", len(query.intersection(title))/len(title))
        if len(description) > 0:
            data.set_value(i, "query_tokens_in_description", len(query.intersection(description))/len(description))
        
        if len(query) > 0:
            data.set_value(i, "title_tokens_in_query", len(title.intersection(query))/len(query))
            data.set_value(i, "description_tokens_in_query", len(description.intersection(query))/len(query))
            
extract_features(train)
extract_features(test)

c_name = ['Nearest Neighbors', 'Linear SVM', 'RBF SVM', 'Random Forest', 'AdaBoost', 'Naive Bayes']
c_classifier = [KNeighborsClassifier(), SVC(kernel='linear', C=0.025), SVC(gamma=2, C=1),\
                RandomForestClassifier(n_estimators=200, min_samples_split=2, n_jobs=1, random_state=1),\
                AdaBoostClassifier(), GaussianNB()]
# pca = decomposition.PCA()
result_out=[]
ik= 2
name, classifier = c_name[ik], c_classifier[ik]
pipeline = Pipeline([("extract_features", features), (name, classifier)])
pipeline.fit(train, train["median_relevance"])
predictions = pipeline.predict(test)
tmp_out = [name, pipeline.score(train, train["median_relevance"])]
result_out.append(tmp_out)
submission = pd.DataFrame({"id": test["id"], "prediction": predictions})
submission.to_csv("python_benchmark"+name+".csv", index=False)

with open('result_out.txt', 'w') as f:
    for a, b in result_out:
        f.write("%s\t%s\n" %(a, b))
