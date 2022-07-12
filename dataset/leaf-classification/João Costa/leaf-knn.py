# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

with open("../input/train.csv", "r") as train:
    training_set = pd.read_csv(train)
training_set.fillna(0, inplace = True)
labels = pd.unique(training_set["species"])

with open("../input/test.csv", "r") as test:
    test_set = pd.read_csv(test)
test_set.fillna(0, inplace = True)

def extract_training_attributes(line):
    return line[2:].values
    
def extract_test_attributes(line):
    return line[1:].values
    
def euclidean_distance(attr1, attr2):
    return math.sqrt(sum(
        map(lambda attrs: (attrs[0] - attrs[1])**2, zip(attr1, attr2))))

def manhattan_distance(attr1, attr2):
    return sum(
        map(lambda attrs: abs(attrs[0] - attrs[1]), zip(attr1, attr2)))
        
distance = euclidean_distance

# From https://arxiv.org/pdf/1101.5783.pdf
def weight(d, k, n, i):
    return (1.0 + d / 2.0 - d / (2.0 * k**(2/d)) * (i**(1+2/d) - (i - 1)**(1+2/d)))
    
def normalized(attrs, avgs, stds):
    centered = map(lambda x: x[0] - x[1], zip(attrs, avgs))
    return list(map(lambda x: x[0] / x[1], zip(centered, stds)))

def classify(line, training_data, labels, k, avgs, stds):
    attrs = normalized(extract_test_attributes(line), avgs, stds)
    distances = training_data.apply(
        lambda x: distance(normalized(extract_training_attributes(x), avgs, stds), attrs),
        1)
    distance_frame = pd.DataFrame(data={"dist": distances, "idx": distances.index})
    indices = distance_frame.nsmallest(k, "dist")["idx"]
    predictions = list(map(lambda i: training_data.ix[i]["species"], indices))
    predict_dict = {}
    print(int(line["id"]))
    for p in labels:
        predict_dict[p] = 0
    i = 1;
    d = len(attrs)
    n = training_data.size
    for p in predictions:
        predict_dict[p] += weight(d, k, n, i)
        i += 1
    for p in predict_dict:
        predict_dict[p] = predict_dict[p] / k
    predict_dict["id"] = int(line["id"])
    return predict_dict

attributes_avg = training_set.mean(numeric_only=True)[1:].values
attributes_std = training_set.std(numeric_only=True)[1:].values
classifications = test_set.apply(
    lambda x: classify(x, training_set, labels, 7, attributes_avg, attributes_std)
    ,1)
output = pd.DataFrame(classifications.tolist())
output.to_csv("predictions.csv", index = False)
print("Done")
# Any results you write to the current directory are saved as output.