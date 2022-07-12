import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import numpy
import pickle
import os.path

# Helper function to check if a string is in ascii
def is_ascii(s):
    return all(ord(c) < 128 for c in s)


# Helper function to remove anomolies in ingredients
all_ingredients = []
def preprocess(ingredients):
    ingredients = [(ingredient.replace('-', ' ')).lower() for ingredient in ingredients if is_ascii(ingredient)]
    for i in ingredients:
        all_ingredients.append(i)
    return ingredients

# Helper function to convert ingredients to encoding
def convert_ingredient_list_to_encoding(ingredients):
    ingredients = [ingredient_to_encoding[ingredient] for ingredient in ingredients]
    return ingredients

# Helper function to convert ingredients to encoding
def convert_cuisine_to_encoding(cuisine):
    return cuisine_to_encoding[cuisine]


def remove_not_found_ingredients(ingredients):
    filtered_ingredients = []
    for i in ingredients:
        if i in all_ingredients:
            filtered_ingredients.append(i)
    return filtered_ingredients

def vectorize_ingredients(ingredients_encoded):
    vectorized_list = []
    for i in range(7003):
        vectorized_list.append(0)
    for i in ingredients_encoded:
        vectorized_list[i] = 1
    return vectorized_list

print("STARTED")

# Reats the json data files
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')

print("FILES FOUND")

# Gets rid of reciples that only have one ingredient from train
train['num_ingredients'] = train['ingredients'].apply(len)
train = train[train['num_ingredients'] > 1]

# Filters out non ASCII ingredients from train and test json
train['ingredients'] = train['ingredients'].apply(preprocess)
test['ingredients'] = test['ingredients'].apply(preprocess)

# Encodes dependent variable and creates a dictionary as a reference
cuisine_encoder = LabelEncoder()
cuisine_encoder.fit_transform(train['cuisine'].values)
cuisine_to_encoding = dict(zip(cuisine_encoder.classes_, cuisine_encoder.transform(cuisine_encoder.classes_)))
encoding_to_cuisine = dict(zip(cuisine_encoder.transform(cuisine_encoder.classes_), cuisine_encoder.classes_))

# Converts the list of ingredients into a set so that each only appears once
all_ingredients = list(set(all_ingredients))

# Encodes all ingredients and creates a dictionary as a reference
ingredient_encoder = LabelEncoder()
ingredient_encoder.fit_transform(all_ingredients)
ingredient_to_encoding = dict(zip(ingredient_encoder.classes_, ingredient_encoder.transform(ingredient_encoder.classes_)))
encoding_to_ingredient = dict(zip(ingredient_encoder.transform(ingredient_encoder.classes_), ingredient_encoder.classes_))

# Converts ingredients and cuisine to their encodings for training
train['ingredients_encoded'] = train['ingredients'].apply(convert_ingredient_list_to_encoding)
train['cuisine_encoded'] = train['cuisine'].apply(convert_cuisine_to_encoding)

# Removes ingredients not found from training
# Converts ingredients and cuisine to their encodings for testing
test['ingredients'] = test['ingredients'].apply(remove_not_found_ingredients)
test['ingredients_encoded'] = test['ingredients'].apply(convert_ingredient_list_to_encoding)

print("DONE ENCODING")

# 6595 is the size of all unique ingredient list
train['vectorized_ingredients'] = train['ingredients_encoded'].apply(vectorize_ingredients)
test['vectorized_ingredients'] = test['ingredients_encoded'].apply(vectorize_ingredients)

print("DONE VECTORIZING")

X = []
for i in train['vectorized_ingredients']:
    X.append(i)

Y = []
for i in train['cuisine_encoded']:
    Y.append(i)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

X_test = []
for i in test['vectorized_ingredients']:
    X_test.append(i)

print("STARTED TRAINING")
results = clf.predict(X_test)

test = pd.read_json('../input/test.json')
translated_results = []
for i in results:
    translated_results.append(encoding_to_cuisine[i])

final_results = []
for i in range(len(translated_results)):
    element = [test['id'][i], translated_results[i]]
    final_results.append(element)

df = pd.DataFrame(final_results, columns=['id', 'cuisine'])
df.reset_index(drop=True)
df.to_csv('sample_submission.csv', index=False)
print("DONE")
