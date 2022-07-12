import json
import numpy as np
import sys
from sklearn.naive_bayes import MultinomialNB as nb
import pickle


# Load training and test data json objects
train_file = open("../input/train.json")
training_data = json.load(train_file)
train_file.close()
test_file = open("../input/test.json")
test_data = json.load(test_file)
test_file.close()

# Initialize variables
ingredients = set()  # ingredient vocabulary
cuisines = set()  # labels
recipe_ids_train = list()  # maps natural numbers to ids
recipe_ids_test = list()  # maps natural numbers to ids

print ('loading training data...')

# Gather recipe ids, and cuisine (label) set and ingredient set
for example in training_data:
    recipe_ids_train.append(example['id'])
    cuisines.add(example['cuisine'])
    for item in example['ingredients']:
        ingredients.add(item)

print ('loading test data...')

# Augment ingredients set with vocabulary in test data (some ingredients may not appear in training data)
for example in test_data:
    for item in example['ingredients']:
        ingredients.add(item)

# Map each ingredient to a natural number that represents a dimension in feature vector
ingredients = list(ingredients)
ingredients = dict([(ingredient, i) for i, ingredient in enumerate(ingredients)])
# Store size of feature vector
vec_size = len(ingredients)
# Map each cuisine to a natural number
cuisines = list(cuisines)
cuisines = dict([(cuisine, i) for i, cuisine in enumerate(cuisines)])
# Reverse mapping (natural  number to cuisine label) will be used to generate submission file
num_to_cuisine = {val: key for key, val in cuisines.items()}

# Initialize lists that will be used to create training and test data arrays.
X_list = []
X_test_list = []
y_list = []

# Build training data matrix
# For each recipe (example), initialize a feature vector (all zeros) with a dimenation for each ingredient
# For each ingredient set the corresponding component to 1; leave all others as 0
# Store the cuisine label (in natural number form) in the answer vector (y)
for example in training_data:
    ingredients_vector_list = [0 for x in range(vec_size)]
    for ingredient in example['ingredients']:
        dimension = ingredients[ingredient]
        ingredients_vector_list[dimension] = 1.0
    X_list.append(ingredients_vector_list)

    cuisine_int = cuisines[example['cuisine']]
    y_list.append(cuisine_int)

# create numpy arrays for training data
X = np.array(X_list)
y = np.array(y_list)

# Build test data matrix (as above)
for example in test_data:
    recipe_ids_test.append(example['id'])
    ingredients_vector_list = [0 for x in range(vec_size)]
    for ingredient in example['ingredients']:
        dimension = ingredients[ingredient]
        ingredients_vector_list[dimension] = 1.0
    X_test_list.append(ingredients_vector_list)

# Create numpy array for test data
X_test = np.array(X_test_list)

# Train naive bayes classifier
print ('train naive bayes classifier')

classifier = nb()
classifier.fit(X, y)

# store or load
#out_pickle = open('./naive_bayes_model1.pickle', 'w')
#pickle.dump(classifier, out_pickle)
#out_pickle.close()
#classifier = pickle.load(open('naive_bayes_model1.pickle'))

# generate predictions for test data
y_pred_test_nb = classifier.predict(X_test)
# store output as list of cuisine labels
y_pred_test_nb = [num_to_cuisine[x] for x in y_pred_test_nb]

# write results to file
out_nb = open('nb1.csv','w')
out_nb.write('id,cuisine\n')

for x in range(len(recipe_ids_test)):
    out_nb.write(str(recipe_ids_test[x]) + ',' + y_pred_test_nb[x] + '\n')
out_nb.close()