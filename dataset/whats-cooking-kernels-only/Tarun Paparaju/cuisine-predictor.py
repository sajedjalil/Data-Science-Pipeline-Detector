import json
import numpy as np

with open('../input/train.json') as json_file:  
    train_data = json.load(json_file)
    
with open('../input/test.json') as json_file:  
    test_data = json.load(json_file)
    
cuisines = []

for i in range(0, len(train_data)):
    cuisines.append(train_data[i]['cuisine'])
    
cuisines = list(set(cuisines))

ingredients_train = []

for i in range(0, len(train_data)):
    for j in range(0, len(train_data[i]['ingredients'])):
        ingredients_train.append(train_data[i]['ingredients'][j])
        
ingredients_train = list(set(ingredients_train))

ingredients_test = []

for i in range(0, len(test_data)):
    for j in range(0, len(test_data[i]['ingredients'])):
        ingredients_test.append(test_data[i]['ingredients'][j])
        
ingredients_test = list(set(ingredients_test))

ingredients = list(set(ingredients_train) | set(ingredients_test))

for i in range(0, len(train_data)):
    train_data[i]['cuisine'] = list.index(cuisines, train_data[i]['cuisine']) + 1
    
    for j in range(0, len(train_data[i]['ingredients'])):
        train_data[i]['ingredients'][j] = list.index(ingredients, train_data[i]['ingredients'][j]) + 1
        
features = []
targets = []

for i in range(0, len(train_data)):
    features.append(train_data[i]['ingredients'])
    targets.append([train_data[i]['cuisine']])

oneHotFeatures = np.zeros((len(features), len(ingredients)))

for i in range(0, len(features)):
    for j in range(0, len(ingredients)):
        if j + 1 in train_data[i]['ingredients']:
            oneHotFeatures[i][j] = 1
        else:
            oneHotFeatures[i][j] = 0
            
oneHotTargets = np.zeros((len(targets), len(cuisines)))
            
for i in range(0, len(targets)):
    for j in range(0, len(cuisines)):
        if j + 1 == train_data[i]['cuisine']:
            oneHotTargets[i][j] = 1
        else:
            oneHotTargets[i][j] = 0
            
oneHotFeatures = np.int32(oneHotFeatures)
features = oneHotFeatures

oneHotTargets = np.int32(oneHotTargets) 
targets = oneHotTargets

import json
import numpy as np

with open('../input/train.json') as json_file:  
    train_data = json.load(json_file)
    
with open('../input/test.json') as json_file:  
    test_data = json.load(json_file)
    
cuisines = []

for i in range(0, len(train_data)):
    cuisines.append(train_data[i]['cuisine'])
    
cuisines = list(set(cuisines))

ingredients_train = []

for i in range(0, len(train_data)):
    for j in range(0, len(train_data[i]['ingredients'])):
        ingredients_train.append(train_data[i]['ingredients'][j])
        
ingredients_train = list(set(ingredients_train))

ingredients_test = []

for i in range(0, len(test_data)):
    for j in range(0, len(test_data[i]['ingredients'])):
        ingredients_test.append(test_data[i]['ingredients'][j])
        
ingredients_test = list(set(ingredients_test))

ingredients = list(set(ingredients_train) | set(ingredients_test))

for i in range(0, len(train_data)):
    train_data[i]['cuisine'] = list.index(cuisines, train_data[i]['cuisine']) + 1
    
    for j in range(0, len(train_data[i]['ingredients'])):
        train_data[i]['ingredients'][j] = list.index(ingredients, train_data[i]['ingredients'][j]) + 1
        
features = []
targets = []

for i in range(0, len(train_data)):
    features.append(train_data[i]['ingredients'])
    targets.append([train_data[i]['cuisine']])

oneHotFeatures = np.zeros((len(features), len(ingredients)))

for i in range(0, len(features)):
    for j in range(0, len(ingredients)):
        if j + 1 in train_data[i]['ingredients']:
            oneHotFeatures[i][j] = 1
        else:
            oneHotFeatures[i][j] = 0
            
oneHotTargets = np.zeros((len(targets), len(cuisines)))
            
for i in range(0, len(targets)):
    for j in range(0, len(cuisines)):
        if j + 1 == train_data[i]['cuisine']:
            oneHotTargets[i][j] = 1
        else:
            oneHotTargets[i][j] = 0
            
oneHotFeatures = np.int32(oneHotFeatures)
features = oneHotFeatures

oneHotTargets = np.int32(oneHotTargets) 
targets = oneHotTargets

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(16))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(48))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(20, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(features, targets, epochs=50)

# import sklearn
# from sklearn.ensemble import RandomForestClassifier

# classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 42)
# classifier.fit(features, targets)
        
features = []

for i in range(0, len(test_data)):
    features.append(test_data[i]['ingredients'])
    
oneHotFeatures = np.zeros((len(features), len(ingredients)))

for i in range(0, len(features)):
    for j in range(0, len(ingredients)):
        if ingredients[j] in test_data[i]['ingredients']:
            oneHotFeatures[i][j] = 1
        else:
            oneHotFeatures[i][j] = 0
            
oneHotFeatures = np.int32(oneHotFeatures)
features = oneHotFeatures

predictions = model.predict(features)

cuisinePredictions = []

for i in range(0, len(predictions)):
    cuisinePredictions.append(cuisines[list.index(list(predictions[i]), max(predictions[i]))])
    
ids = np.int32(np.array([test_data[i]['id'] for i in range(0, len(test_data))]))

import pandas as pd

submission = pd.DataFrame(ids, columns={"id"})
submission['cuisine'] = cuisinePredictions
submission = submission[["id", "cuisine"]]
submission.to_csv('sample_submission.csv', sep=',', index=False)