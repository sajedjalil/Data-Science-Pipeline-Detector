# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
'''
This kernel is made by taking reference from other kernels and using external data 
available at https://github.com/rwsproat/text-normalization-data (the link is also mentioned
under discussion tab).

You can add other functions along with this to improve your score.

Note: The external dataset is huge and it will take a little more time to run.
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import operator
import glob
import os
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




INPUT_PATH = "../input"
SUBM_PATH = INPUT_PATH
DATA_INPUT_PATH = "../input/en_with_types"
print(glob.glob(INPUT_PATH + '*'))
symbols = ['km', 'km2', 'km²', 'mm', 'Hz', 'mi', 'cm', 'ft', 'm', 'kg', 'm3', 'MB', 'm2', 'mg', 'yd', 'ha']


print('Train start...')
train = open(INPUT_PATH + "/en_train.csv")
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


files = os.listdir(DATA_INPUT_PATH)
for file in files:
    train = open(os.path.join(DATA_INPUT_PATH, file))
    while 1:
        line = train.readline().strip()
        if line == '':
            break
        total += 1
        pos = line.find('\t')
        text = line[pos + 1:]
        if text[:3] == '':
            continue
        arr = text.split('\t')
        if arr[0] == '<eos>':
            continue
        if arr[1] != '<self>':
            not_same += 1

        if arr[1] == '<self>' or arr[1] == 'sil':
            arr[1] = arr[0]

        if arr[1] == '<self>' or arr[1] == 'sil':
            arr[1] = arr[0]

        if arr[0] not in res:
            res[arr[0]] = dict()
            res[arr[0]][arr[1]] = 1
        else:
            if arr[1] in res[arr[0]]:
                res[arr[0]][arr[1]] += 1
            else:
                res[arr[0]][arr[1]] = 1
    train.close()
    print(file + ':\tTotal: {} Have diff value: {}'.format(total, not_same))
    gc.collect()


total = 0
changes = 0
out = open(SUBM_PATH + '/sub_text_v1.csv', "w")
out.write('"id","after"\n')
test = open(INPUT_PATH + "/en_test_2.csv")
line = test.readline().strip()

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

    out.write('\n')
    total += 1

print('Total: {} Changed: {}'.format(total, changes))
test.close()
out.close()


