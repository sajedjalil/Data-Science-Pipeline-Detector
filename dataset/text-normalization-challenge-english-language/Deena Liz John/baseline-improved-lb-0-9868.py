# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import operator

# Any results you write to the current directory are saved as output.

open('../input/en_test.csv')

print('Train start...')
train = open('../input/en_train.csv', "r")
line = train.readline()
res = dict()
total = 0
not_same = 0
while 1:
    line = train.readline().strip()
    if line == '':
        break
    total += 1
    pos = line.find('","')
    text = line[pos + 2:]
    if text[:3] == '","':
        continue
    text = text[1:-1]
    arr = text.split('","')
    if arr[0] != arr[1]:
        not_same += 1
    if arr[0] not in res:
        res[arr[0]] = dict()
        res[arr[0]][arr[1]] = 1
    else:
        if arr[1] in res[arr[0]]:
            res[arr[0]][arr[1]] += 1
        else:
            res[arr[0]][arr[1]] = 1


train.close()
print('Total: {} Have diff value: {}'.format(total, not_same))

total = 0
changes = 0
out = open('./submission.csv', "w")
out.write('"id","after"\n')
test = open('../input/en_test_2.csv')
line = test.readline().strip()

symbols = ['km', 'km2', 'km²', 'mm', 'Hz', 'mi', 'cm', 'ft', 'm', 'kg', 'm3', 'MB', 'm2', 'mg', 'yd', 'ha']
while 1:
    line = test.readline().strip()
    if line == '':
        break

    pos = line.find(',')
    i1 = line[:pos]
    line = line[pos + 1:]

    pos = line.find(',')
    i2 = line[:pos]
    line = line[pos + 1:]

    line = line[1:-1]
    out.write('"' + i1 + '_' + i2 + '",')
    if line in res:
        srtd = sorted(res[line].items(), key=operator.itemgetter(1), reverse=True)
        out.write('"' + srtd[0][0] + '"')
        changes += 1
    elif any(str.isdigit(c) for c in line) and any(s in line for s in symbols):
        l = line.split(' ')
        if l[0] in res:
            srtd = sorted(res[l[0]].items(), key=operator.itemgetter(1), reverse=True)
            num = srtd[0][0]
            if l[1] == 'km': sy = 'kilometers'
            elif l[1] == 'km2' or l[1] == 'km²': sy = 'square kilometers'
            elif l[1] == 'mm': sy = 'millimeters'
            elif l[1] == 'Hz': sy = 'hertz'
            elif l[1] == 'mi': sy = 'miles'
            elif l[1] == 'cm': sy = 'centimeters'
            elif l[1] == 'm': sy = 'meters'
            elif l[1] == 'ft': sy = 'feet'
            elif l[1] == 'kg': sy = 'kilograms'
            elif l[1] == 'm3': sy = 'cubic meters'
            elif l[1] == 'MB': sy = 'centimeters'
            elif l[1] == 'm2': sy = 'square meters'
            elif l[1] == 'mg': sy = 'milligrams'
            elif l[1] == 'yd': sy = 'yards'
            elif l[1] == 'ha': sy = 'hectares'
            else: sy = ''
            out.write('"' + num + " " + sy + '"')
            changes += 1
        else:
            out.write('"' + line + '"')
    else:
        out.write('"' + line + '"')
        #if any(str.isdigit(c) for c in line) : print line

    out.write('\n')
    total += 1

print('Total: {} Changed: {}'.format(total, changes))
test.close()
out.close()

