import json as js
import operator
import time
start = time.time()

def similarity(testIngredients, trainingIngredients):
    testSet = set(testIngredients)
    trainingSet = set(trainingIngredients)
    score = float(len(testSet&trainingSet))/float(len(testSet|trainingSet))
    return score

with open('../input/train.json') as training_data_file:
    training_data = js.load(training_data_file)
with open('../input/test.json') as test_data_file:
    test_data = js.load(test_data_file)
    
n_training = len(training_data)
n_test = len(test_data)

print ("n_training:", n_training)
print ("n_test:", n_test)

print("runtime:", time.time()-start, "seconds")