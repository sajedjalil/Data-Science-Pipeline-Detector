import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
from sklearn.neighbors import KNeighborsClassifier

# Load and describe data
# print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_json(open("../input/train.json", "r"))
df['n_photos'] = df['photos'].apply(len)
df['n_features'] = df['features'].apply(len)
df['ilevel_categ'] = df['interest_level'].map({'low': 1, 'medium': 2, 'high': 3})
df["n_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
# clean outliers
df = df[df['longitude']!=0]
# df = df[df['price']>100]
# df = df[df['price']<]

# extract numerical features
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
dfX = df.select_dtypes(include=numerics)
del dfX['listing_id']
del dfX['ilevel_categ']
# extract target variable (interest categories)
dfY = df['ilevel_categ'].copy()


# Load and wrangle test data
test = pd.read_json(open("../input/test.json", "r"))
test['n_photos'] = test['photos'].apply(len)
test['n_features'] = test['features'].apply(len)
test["n_description_words"] = test["description"].apply(lambda x: len(x.split(" ")))
# numerics only
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
testX = test.select_dtypes(include=numerics)
listing_id = testX['listing_id'].copy()
del testX['listing_id']

# Train model
clf = KNeighborsClassifier(n_neighbors=600, weights='distance')
clf.fit(dfX,dfY)

print(testX.head(5))
print(dfX.head(5))

# Make predictions
predictions = clf.predict_proba(testX)
submission = pd.DataFrame(index=listing_id)
submission['high'] = predictions[:,2]
submission['medium'] = predictions[:,1]
submission['low'] = predictions[:,0]
print(submission)
submission.to_csv('submission2.csv')


