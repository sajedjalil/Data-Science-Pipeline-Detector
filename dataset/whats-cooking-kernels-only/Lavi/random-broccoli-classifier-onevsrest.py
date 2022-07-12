import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


# Load data
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')

X = train.ingredients.map(" ".join)
y = train.cuisine

X_test = test.ingredients.map(" ".join)

# Create pipeline for using Count Vectorizer followed by a Random Tree Forest Classifier
rf_pipe = Pipeline([('count', TfidfVectorizer(max_features=1000)), 
                    ('rfc', OneVsRestClassifier(RandomForestClassifier(n_estimators=200, random_state=11, class_weight='balanced')))
                    ])


# Paramenters set using a little bit of gridcv testing
# rf_pipe.set_params(count__max_features=1000, rfc__n_estimators=200)

# Fit the data
rf_pipe.fit(X, y)

# Make predictions
y_pred = rf_pipe.predict(X_test)

# Format and output predictions
y_out = pd.DataFrame({'id': test.id, 'cuisine': y_pred})
y_out.to_csv('rfc_sub.csv', index=False)