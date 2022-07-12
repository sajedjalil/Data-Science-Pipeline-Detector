import numpy as np # linear algebra
import csv

from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

f = csv.reader(open('../input/test.csv', 'r'))
for i,  row in enumerate(f):
    if i > 0:
        raw_data = [int(k) for k in row[1].split(',')]
        if len(raw_data) > 6:
            print(raw_data)
            data = np.zeros((len(raw_data)-6, 6))
            net = buildNetwork(6, 8, 8, 1, bias=True)
            for k in range(6, len(data)):
                data[k-6, :] = np.array(raw_data[k-6:k])
            print(data)
            #trainer = BackpropTrainer(net, dataset=data)
            
    if i > 10:
        break