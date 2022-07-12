# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as sparse

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

data_path = "../input/"

def loadData(fname):
    data = pd.read_csv(data_path+fname)
    names = data.columns.get_values()
    ids = data.values[:, 0]
    target = data.values[:, 1:2]
    values = data.values[:, 2:]
    return (names, ids, np.array(target, dtype=np.double), np.array(values, dtype=np.double))
    
def loadDataTest(fname):
    with open(data_path + fname, 'r') as f:
        header = f.readline().strip().split(',')

        ids = []
        table = []
        tres = None
        for line in iter(f.readline, ""):
            if len(ids) % 1000 == 0:
                print(len(ids))
                #                 if len(ids)>0:
                #                     break

                if len(ids) > 0:
                    if tres is None:
                        tres = sparse.csr_matrix(table)
                        table = []
                    else:
                        tres = sparse.vstack((tres, sparse.csr_matrix(table)))
                        table = []

            spl = line.split(',')

            ids.append(spl[0])
            vals = list(map(np.double, spl[1:]))
            table.append(vals)

        tres = sparse.vstack((tres, sparse.csr_matrix(table)))

    return (header, ids, tres)
    
(namesT, idsT, valuesT) = loadDataTest('test.csv')


from six.moves import cPickle as pickle
import bz2
import numpy as np
import os

def loadPickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        loadedData = pickle.load(f)
        return loadedData

def savePickle(pickle_file, data):
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return

def loadPickleBZ(pickle_file):
    with bz2.BZ2File(pickle_file, 'r') as f:
        loadedData = pickle.load(f)
        return loadedData

def savePickleBZ(pickle_file, data):
    with bz2.BZ2File(pickle_file, 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return

import numpy as np
import scipy as sp
import scipy.sparse as sparse
import lightgbm as lgb
from sklearn.model_selection import KFold
from time import time
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
import collections


if True:
    values = valuesT
    values = sparse.csr_matrix(values, dtype=np.uint64)
else:
    values = np.array(values, dtype=np.uint64)
    target = np.array(target, dtype=np.uint64)


try:
    valmap = loadPickleBZ('./valmap.pbz')
except:
    valmap = {}

    for i, row in enumerate(values):
        if i % 100 == 0:
            print('i', i, values.shape[0])
        # if i > 300:
        #     break

        valsInRow = row[row.nonzero()]
        valsInRow = np.array(valsInRow).reshape((-1,))
        for j in range(len(valsInRow)):
            v = valsInRow[j]
            v = int(v - v % 10)
            if v not in valmap:
                valmap[v] = set()

            valmap[v].add(i)

    savePickleBZ('./valmap.pbz', valmap)


try:
    row2rows = loadPickleBZ('./row2rows.pbz')
except:
    row2rows = [[] for i in range(values.shape[0])]

    for i, (val, rowids) in enumerate(valmap.items()):
        if i % 100 == 0 or len(rowids) > 100:
            print('i', i, len(valmap), '\t\t', val)

        for row in rowids:
            row2rows[row].append(rowids)

    savePickleBZ('./row2rows.pbz', row2rows)

del valmap


try:
    rowNumMatches = loadPickleBZ('./rowNumMatches.pbz')
except:

    print('----- Matches')
    rowNumMatches = []
    for i in range(values.shape[0]):
        col = collections.Counter()
        for rs in row2rows[i]:
            col += collections.Counter(rs)

        count = np.array(list(col.values()))
        count.sort()
        count = count[::-1]

        numMatches = []
        for perc in [0.8, 0.7, 0.5, 0.4, 0.3]:
            nm = (count >= (count[0]*perc)).sum()
            numMatches.append(nm)
        rowNumMatches.append(numMatches)

        print(i, '\t', numMatches, '\t', np.array(numMatches)/len(count), '\t', len(count))

    rowNumMatches = np.array(rowNumMatches)

    rowNumMatches = np.sort(rowNumMatches, axis=0)[::-1, :]

    savePickleBZ('./rowNumMatches.pbz', rowNumMatches)


matches30 = rowNumMatches[:, 4]
matches50 = rowNumMatches[:, 2]
matches80 = rowNumMatches[:, 0]
for lim in [1, 2, 3, 5, 10, 20, 50]:
    print('less than : more than', '\t ', lim,
          '\t', (matches30 <= lim).sum(), ':', (matches30 > lim).sum(),
          '\t', (matches50 <= lim).sum(), ':', (matches50 > lim).sum(),
          '\t', (matches80 <= lim).sum(), ':', (matches80 > lim).sum(),
          )

fig = plt.figure()
ax = plt.subplot(111)
ax.set_yscale('log')
plt.plot(rowNumMatches)
plt.show()


print()



















