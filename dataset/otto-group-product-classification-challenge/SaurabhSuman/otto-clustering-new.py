# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 12:35:05 2016

@author: saurabh.s1
"""
import scipy as sp
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

#returns most occuring element from an array
def most_common_element(arr):
    most_occuring = Counter(arr).most_common(1)
    return most_occuring[0][0]


training_dataset = np.genfromtxt("../input/train.csv", delimiter=',', dtype = str)
training_dataset_data = training_dataset[1:,1:-1]
training_dataset_data = training_dataset_data.astype(float)
training_dataset_target = training_dataset[1:,-1]

testing_dataset = np.genfromtxt("../input/test.csv", delimiter=",")


num_clusters = 10
km = KMeans(n_clusters = num_clusters, n_init = 3, init = 'k-means++', verbose = 0)
km.fit(training_dataset_data)
#print km.labels_

print (type(km.labels_))

#error = 0
#for i in xrange(num_clusters):
#    similar_indices = (km.labels_ == i).nonzero()[0]
##    print similar_indices.shape
#    predicted_target = most_common_element(training_dataset_target[similar_indices])
##    print predicted_target
#    error += sum(training_dataset_target[similar_indices] != predicted_target)
##    print error
#print error

output_file = "output.csv"
f = open(output_file, 'w')
f.flush()
f.write("id")
for j in range(1,10):
    f.write(",Class_{}".format(j))
f.write("\n")
#print testing_dataset.shape[0]
for i in range(1, 55):
    test_data_vec = testing_dataset[i,1:]
    test_data_vec = test_data_vec.reshape(1,-1)
    predicted_label_for_test_data = km.predict(test_data_vec)[0]
    similar_indices = (km.labels_ == predicted_label_for_test_data).nonzero()[0]
    predicted_target = most_common_element(training_dataset_target[similar_indices])
    f.write("{}".format(i))
    for j in range(1,10):
        if(predicted_target == ("Class_" + str(j))):
            f.write(",1")
        else:
            f.write(",0")
    f.write("\n")
f.close()       


