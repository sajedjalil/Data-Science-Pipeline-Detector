import json as js
import csv as csv
import scipy as scipy
import numpy as np
import pdb
from sklearn.neural_network import MLPClassifier

with open('../input/train.json') as json_data:
    data = js.load(json_data)
    json_data.close()

classes = [item['cuisine'] for item in data]
ingredients = [item['ingredients'] for item in data]
unique_ingredients = set(item for sublist in ingredients for item in sublist)
unique_cuisines = set(classes)

# print len(data)
# print (classes)
print ( len (ingredients) )
print ( len ( unique_ingredients ) )
print ( len ( unique_cuisines ) )

big_data_matrix = scipy.sparse.dok_matrix((len(ingredients), len(unique_ingredients)), dtype=np.dtype(bool))

for d,dish in enumerate(ingredients):
    for i,ingredient in enumerate(unique_ingredients):
        if ingredient in dish:
            big_data_matrix[d,i] = True
            
clf2 = MLPClassifier (algorithm = 'sgd', alpha=0.001, hidden_layer_sizes=(100, 100, 100), random_state=1, activation='logistic' );
f = clf2.fit(big_data_matrix, classes)
result = [(ref == res, ref, res) for (ref, res) in zip(classes, clf2.predict(big_data_matrix))]
accuracy_learn = sum (r[0] for r in result) / float ( len(result) )

print('Accuracy on the learning set: ', accuracy_learn)