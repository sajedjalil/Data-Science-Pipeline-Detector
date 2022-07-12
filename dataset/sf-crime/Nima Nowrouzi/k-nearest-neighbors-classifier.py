import math
import zipfile
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_hour(date):
    return date.hour


def get_decimal(number):
    dec, int = math.modf(number)
    return dec


z = zipfile.ZipFile('../input/train.csv.zip')
df = pd.read_csv(z.open('train.csv'))
z = zipfile.ZipFile('../input/test.csv.zip')
test_df = pd.read_csv(z.open('test.csv'))
df['Dates'] = pd.to_datetime(df['Dates'])

df['Dates'] = df['Dates'].apply(get_hour)
df['X'] = df['X'].apply(get_decimal)
df['Y'] = df['Y'].apply(get_decimal)
df['PdDistrict'] = df['PdDistrict'].astype('category')
cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
df['Category'] = df['Category'].astype('category')

test_df['Dates'] = pd.to_datetime(test_df['Dates'])

test_df['Dates'] = test_df['Dates'].apply(get_hour)
test_df['X'] = test_df['X'].apply(get_decimal)
test_df['Y'] = test_df['Y'].apply(get_decimal)
test_df['PdDistrict'] = test_df['PdDistrict'].astype('category')
cat_columns = test_df.select_dtypes(['category']).columns
test_df[cat_columns] = test_df[cat_columns].apply(lambda x: x.cat.codes)


for category in df['Category'].unique():
    expression = df['Category'] == category
    std = df['Dates'][expression].std()
    if std > 6.9:
        df['Dates'][expression] = 0
    remove_outlier = (df['Dates'][expression] < (2 * std)) & (df['Dates'][expression] > (-2 * std))
    df['Dates'][expression][~remove_outlier] = 0

labels = df["Category"].values
features = df[['Dates', u'PdDistrict', u'X', u'Y']].values
classifier = KNeighborsClassifier(n_neighbors=40)
classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])
print('fit data')
classifier.fit(features, labels)
print('end fit')
test_features = test_df[['Dates', u'PdDistrict', u'X', u'Y']].values
result = classifier.predict(test_features)
print('end prediction')
submit = pd.DataFrame({'Id': test_df.Id.tolist()})
for category in df['Category'].unique():
    print(category)
    submit[category] = np.where(result == category, 1, 0)

submit.to_csv('nearest_neigbour.csv', index=False)