__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
import os
from collections import defaultdict
from operator import itemgetter
import operator
import random
import itertools
import heapq
import math
random.seed(999)


def apk(actual, predicted, k=7):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


hashes_important_indexes = list(range(2, 24))
hashes_important_indexes.remove(6) # date type (fecha_alta)
hashes_important_indexes.remove(22) # float type (renta)
all = itertools.combinations(hashes_important_indexes, 5)
hashes_indexes = random.sample(list(all), 2)
# print('Current set of hash indexes: {}'.format(hashes_indexes))


distrib = defaultdict(int)

def get_hashes(arr):
    global hashes_indexes, distrib

    (fecha_dato, ncodpers, ind_empleado,
    pais_residencia, sexo, age,
    fecha_alta, ind_nuevo, antiguedad,
    indrel, ult_fec_cli_1t, indrel_1mes,
    tiprel_1mes, indresi, indext,
    conyuemp, canal_entrada, indfall,
    tipodom, cod_prov, nomprov,
    ind_actividad_cliente, renta, segmento) = arr[:24]
    if renta == '' or renta == 'NA':
        renta1 = '-100'
    elif float(renta) < 45542.97:
        renta1 = '200'
    elif float(renta) < 57629.67:
        renta1 = '400'
    elif float(renta) < 68211.78:
        renta1 = '600'
    elif float(renta) < 78852.39:
        renta1 = '800'
    elif float(renta) < 90461.97:
        renta1 = '1000'
    elif float(renta) < 103855.23:
        renta1 = '1200'
    elif float(renta) < 120063.00:
        renta1 = '1400'
    elif float(renta) < 141347.49:
        renta1 = '1600'
    elif float(renta) < 173418.36:
        renta1 = '1800'
    elif float(renta) < 234687.12:
        renta1 = '2000'
    else:
        renta1 = '2000'

    distrib[renta1] += 1

    sub = []
    if 1:
        # Fixed set
        sub.append((1, pais_residencia, sexo, age, ind_nuevo, segmento, ind_empleado, ind_actividad_cliente, indresi, renta1))
        # sub.append((2, segmento, nomprov))
        sub.append((3, ncodpers))
        sub.append((4, ind_empleado,ind_actividad_cliente,ind_nuevo,canal_entrada))
        sub.append((5, pais_residencia, sexo, renta1, age, segmento))
        sub.append((6, pais_residencia, sexo, antiguedad, segmento, ind_empleado))
    else:
        # Random set
        sub = [itemgetter(*h)(arr) for h in hashes_indexes]

    return sub


def date_is_important(date):
    possible_dates = ['2015-01-28', '2015-02-28', '2015-03-28', '2015-04-28', '2015-05-28',
    '2015-06-28', '2015-07-28', '2015-08-28', '2015-09-28', '2015-10-28',
    '2015-11-28', '2015-12-28', '2016-01-28', '2016-02-28', '2016-03-28',
    '2016-04-28', '2016-05-28']

    koeff = 1
    score = 8 + koeff*possible_dates.index(date)
    return score


def add_data_to_main_arrays(arr, best, overallbest, customer):
    date = arr[0]
    ncodpers = arr[1]
    hashes = get_hashes(arr)
    importance = date_is_important(date)

    part = arr[24:]
    num_prod = 0
    for i in range(24):
        if part[i] == '1':
            num_prod +=1
    num_prod = num_prod % 4        
    for i in range(24):
        if part[i] == '1':
            if ncodpers in customer:
                if customer[ncodpers][i] == '0':
                    for h in hashes:
                        best[h][i] += (importance+num_prod)
                    overallbest[i] += (importance+num_prod)
            else:
                for h in hashes:
                    best[h][i] += (importance+num_prod)
                overallbest[i] += (importance+num_prod)
    customer[ncodpers] = part


def sort_main_arrays(best, overallbest):
    out = dict()
    for b in best:
        arr = best[b]
        srtd = sorted(arr.items(), key=operator.itemgetter(1), reverse=True)
        out[b] = srtd
    best = out
    overallbest = sorted(overallbest.items(), key=operator.itemgetter(1), reverse=True)
    return best, overallbest


def get_next_best_prediction(best, hashes, predicted, cst):

    score = [0] * 24

    for h in hashes:
        if h in best:
            for i in range(len(best[h])):
                sc = 24-i + len(h)
                index = best[h][i][0]
                if cst is not None:
                    if cst[index] == '1':
                        continue
                if index not in predicted:
                    score[index] += sc

    final = []
    pred = heapq.nlargest(24, range(len(score)), score.__getitem__)
    for i in range(len(pred)):
        if score[pred[i]] > 0:
            final.append(pred[i])
        if len(final) >= 7:
            break

    return final


def get_predictions(arr1, best, overallbest, customer):

    predicted = []
    hashes = get_hashes(arr1)
    ncodpers = arr1[1]

    customer_data = None
    if ncodpers in customer:
        customer_data = customer[ncodpers]

    predicted = get_next_best_prediction(best, hashes, predicted, customer_data)

    # overall
    if len(predicted) < 7:
        for a in overallbest:
            # If user is not new
            if ncodpers in customer:
                if customer[ncodpers][a[0]] == '1':
                    continue
            if a[0] not in predicted:
                predicted.append(a[0])
                if len(predicted) == 7:
                    break

    return predicted


def get_real_values(arr1, customer):
    real = []
    ncodpers = arr1[1]
    arr2 = arr1[24:]

    for i in range(len(arr2)):
        if arr2[i] == '1':
            if ncodpers in customer:
                if customer[ncodpers][i] == '0':
                    real.append(i)
            else:
                real.append(i)
    return real


def run_solution():

    print('Preparing arrays...')
    f = open("../input/train_ver2.csv", "r")
    first_line = f.readline().strip()
    first_line = first_line.replace("\"", "")
    map_names = first_line.split(",")[24:]

    # Normal variables
    customer = dict()
    best = defaultdict(lambda: defaultdict(int))
    overallbest = defaultdict(int)

    # Validation variables
    customer_valid = dict()
    best_valid = defaultdict(lambda: defaultdict(int))
    overallbest_valid = defaultdict(int)

    valid_part = []
    # Calc counts
    total = 0
    while 1:
        line = f.readline()[:-1]
        total += 1

        if line == '':
            break

        tmp1 = line.split("\"")
        arr = tmp1[0][:-1].split(",") + [tmp1[1]] + tmp1[2][1:].split(',')
        arr = [a.strip() for a in arr]

        # Normal part
        add_data_to_main_arrays(arr, best, overallbest, customer)

        # Valid part
        if arr[0] != '2016-04-28':
            add_data_to_main_arrays(arr, best_valid, overallbest_valid, customer_valid)
        else:
            valid_part.append(arr)

        if total % 1000000 == 0:
            print('Process {} lines ...'.format(total))
            # break

    f.close()

    print(distrib)

    print('Sort best arrays...')
    print('Hashes num: ', len(best))
    print('Valid part: ', len(valid_part))

    # Normal
    best, overallbest = sort_main_arrays(best, overallbest)
    # print(best)

    # Valid
    best_valid, overallbest_valid = sort_main_arrays(best_valid, overallbest_valid)

    map7 = 0.0
    print('Validation...')
    for arr1 in valid_part:
        predicted = get_predictions(arr1, best_valid, overallbest_valid, customer_valid)
        real = get_real_values(arr1, customer_valid)

        score = apk(real, predicted)
        map7 += score

    if len(valid_part) > 0:
        map7 /= len(valid_part)
    print('Predicted score: {}'.format(map7))

    print('Generate submission...')
   
    sub_file = os.path.join('submission_' + str(round(map7, 8)) + '.csv')
    out = open(sub_file, "w")
    f = open("../input/test_ver2.csv", "r")
    f.readline()
    total = 0
    out.write("ncodpers,added_products\n")

    while 1:
        line = f.readline()[:-1]
        total += 1

        if line == '':
            break

        tmp1 = line.split("\"")
        arr = tmp1[0][:-1].split(",") + [tmp1[1]] + tmp1[2][1:].split(',')
        arr = [a.strip() for a in arr]
        ncodpers = arr[1]
        out.write(ncodpers + ',')

        predicted = get_predictions(arr, best, overallbest, customer)

        for p in predicted:
            out.write(map_names[p] + ' ')

        if total % 1000000 == 0:
            print('Read {} lines ...'.format(total))
            # break

        out.write("\n")

    print('Total cases:', str(total))
    out.close()
    f.close()


if __name__ == "__main__":
    run_solution()
