# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# number of records
print("Number of records")
c = 0
for line in open("../input/test.csv", "r"):
    c = c + 1
    #print(line, end = '')
print( c - 1)

# nullertekek szama oszloponkent
print("Number of null values by columns")
cr = 0
nrowcn = 0
for line in open("../input/test.csv", "r"):
    row = []
    if nrowcn != 0:
        row.append(line.split(','))
        for i in range(0, len(row)):
            if row[i] == "":
                cr += 1
    nrowcn = nrowcn + 1
print( cr )

# converting datas into appropiate form for the functions
datas = []
rowc = 0    
for line in open("../input/test.csv", "r"):
    if rowc != 0:
        datas.append([])
        datas[rowc - 1] = line.split(',')
    rowc = rowc + 1

# numerikus oszop maximum, minimum, kovepertekeki(atlag, median), es szorasa
print("Numeric column maximum, minimum, middle values(average, median), and dispersion")
#oszlop maximum, minimum, atlag
print("max result in column order:")
for i in range(0, len(datas[0])):
    print("column " + str(i))
    array = []
    for j in range(0, len(datas)):
            if datas[j][i] == "":
                datas[j][i] = 0
            array.append(float(datas[j][i]))
    print("maximum")
    print(np.amax(array, axis=None, out=None, keepdims=False))
    print("minimum")
    print(np.amin(array, axis=None, out=None, keepdims=False))
    print("average")
    print(np.average(array, axis=None, weights=None, returned=False))
    print("median")
    print(np.median(array, axis=None, out=None, overwrite_input=False, keepdims=False))
    print("dispersion, standard deviation")
    print(np.std(array, axis=None, dtype=None, out=None, ddof=0, keepdims=False))
    # Historgram about dispersions
    np.histogram(array, bins=10, range=None, normed=False, weights=None, density=None)
# Any results you write to the current directory are saved as output.