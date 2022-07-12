########################################################
'''
READ MY COMMENT BELOW
'''
########################################################

import os
import operator
from num2words import num2words #i'm not sure that it works with ru-version
import gc


INPUT_PATH = r'../input'
DATA_INPUT_PATH = r'../input/ru_with_types'
SUBM_PATH = INPUT_PATH

SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
OTH = str.maketrans("፬", "4")

print('Train start...')

file = "ru_train.csv"
train = open(os.path.join(INPUT_PATH, "ru_train.csv"), encoding='UTF8')
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
print(file + ':\tTotal: {} Have diff value: {}'.format(total, not_same))

files = os.listdir(DATA_INPUT_PATH)
for file in files:
    train = open(os.path.join(DATA_INPUT_PATH, file), encoding='UTF8')
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

#looks useless for ru-version, but...

sdict = {}
sdict['km2'] = 'square kilometers'
sdict['km'] = 'kilometers'
sdict['kg'] = 'kilograms'
sdict['lb'] = 'pounds'
sdict['dr'] = 'doctor'
sdict['m²'] = 'square meters'

total = 0
changes = 0
out = open(os.path.join(SUBM_PATH, 'baseline_ext_ru.csv'), "w", encoding='UTF8')
out.write('"id","after"\n')
test = open(os.path.join(INPUT_PATH, "ru_test.csv"), encoding='UTF8')
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
    else:
        if len(line) > 1:
            val = line.split(',')
            if len(val) == 2 and val[0].isdigit and val[1].isdigit:
                line = ''.join(val)

        if line.isdigit():
            srtd = line.translate(SUB)
            srtd = srtd.translate(SUP)
            srtd = srtd.translate(OTH)
            out.write('"' + num2words(float(srtd)) + '"')
            changes += 1
        elif len(line.split(' ')) > 1:
            val = line.split(' ')
            for i, v in enumerate(val):
                if v.isdigit():
                    srtd = v.translate(SUB)
                    srtd = srtd.translate(SUP)
                    srtd = srtd.translate(OTH)
                    val[i] = num2words(float(srtd))
                elif v in sdict:
                    val[i] = sdict[v]

            out.write('"' + ' '.join(val) + '"')
            changes += 1
        else:
            out.write('"' + line + '"')

    out.write('\n')
    total += 1

print('Total: {} Changed: {}'.format(total, changes))
test.close()
out.close()