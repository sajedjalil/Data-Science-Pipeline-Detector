# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree
import csv
import random
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train_data = []
test_data = []
test_target = []
train_target = []
n = []
m = []
l = 0

with open('../input/train.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
	    l = l + 1
	    
	    n = [row['Semana'],row['Agencia_ID'], row['Canal_ID'],row['Ruta_SAK'],row['Cliente_ID'],row['Producto_ID']]
	    train_data.append(n)
	    m = [row['Demanda_uni_equil']]
	    train_target.append(m)
	    if(l > 700000):
	        break
	   
l = 0
val = 300000
num = 0
clf = tree.ExtraTreeClassifier()
clf = clf.fit(train_data,train_target)
out = open("answer.csv", "w")
out.write("id,Demanda_uni_equil\n")
for i in range(0,24):
    test_data = []
    predict_data = [] 
    with open('../input/test.csv') as csvfile:
	    reader = csv.DictReader(csvfile)
	    for row in reader:
	        l = l + 1
	        if(l <= val * i):
	            continue
	        n = [row['Semana'],row['Agencia_ID'], row['Canal_ID'],row['Ruta_SAK'],row['Cliente_ID'],row['Producto_ID']]
	        test_data.append(n)
	        if(l > (val * (i+1))):
	            break

    predict_data = clf.predict(test_data)

    count = len(predict_data)
    temp = 0
    while (count > 0):
        value = predict_data[temp]
        out.write(str(num) + ','+''.join(value)+"\n")
        if(num == 6999250):
            break
        count = count - 1
        num = num + 1
        temp = temp + 1
        if(num%100000 == 0):
            print(num)
    
    if(num == 6999250):
            break
out.close()
