# coding: utf-8
__author__ = 'Ravi: https://kaggle.com/company'

import datetime
import time
from collections import defaultdict
import gc

def run_solution():
    print('Preparing arrays...')
    f = open("../input/train.csv", "r")
    f.readline()
    total = 0

    client_product_arr = defaultdict(int)
    client_product_arr_count = defaultdict(int)

    # Calc counts
    avg_target = 0.0
    while 1:
        line = f.readline().strip()
        total += 1

        if total % 10000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        week = int(arr[0])
        agency = arr[1]
        canal_id = arr[2]
        ruta_sak = arr[3]
        cliente_id = int(arr[4])
        producto_id = int(arr[5])
        vuh = arr[6]
        vh = arr[7]
        dup = arr[8]
        dp = arr[9]
        target = int(arr[10])

        client_product_arr[(cliente_id, producto_id)] += target
        client_product_arr_count[(cliente_id, producto_id)] += 1

    f.close()
    gc.collect()
    
    print('Generate submission...')
    now = datetime.datetime.now()
    path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    f = open("../input/test.csv", "r")
    f.readline()
    total = 0
    out.write("id,Demanda_uni_equil\n")

    index_both = 0
    index_empty = 0

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 10000000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        id = arr[0]
        week = int(arr[1])
        agency = arr[2]
        canal_id = arr[3]
        ruta_sak = arr[4]
        cliente_id = int(arr[5])
        producto_id = int(arr[6])

        out.write(str(id) + ',')
        if (cliente_id, producto_id) in client_product_arr:
            val = client_product_arr[(cliente_id, producto_id)]/client_product_arr_count[(cliente_id, producto_id)]
            out.write(str(round(val)))
            index_both += 1
        else:
            avg_target = 4.00
            out.write(str(avg_target))
            index_empty += 1
        out.write("\n")

    print('Both: {}'.format(index_both))
    print('Empty: {}'.format(index_empty))

    out.close()
    f.close()

start_time = time.time()
run_solution()
print("Elapsed time overall: %s seconds" % (time.time() - start_time))
