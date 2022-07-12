# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
from heapq import nlargest
from operator import itemgetter
import os
import time
import math
from collections import defaultdict


def prep_xy(x, y):
    range_x = 500
    ix = math.floor(range_x*x/10)
    if ix < 0:
        ix = 0
    if ix >= range_x:
        ix = range_x-1
    
    range_y = 1000
    iy = math.floor(range_y*y/10)
    if iy < 0:
        iy = 0
    if iy >= range_y:
        iy = range_y-1

    return ix, iy


def run_solution():
    print('Preparing data...')
    f = open("../input/train.csv", "r")
    f.readline()
    total = 0

    grid = defaultdict(lambda: defaultdict(int))
    grid_sorted = dict()

    # Calc counts
    while 1:
        line = f.readline().strip()
        total += 1

        if line == '':
            break

        arr = line.split(",")
        row_id = arr[0]
        x = float(arr[1])
        y = float(arr[2])
        accuracy = arr[3]
        time = arr[4]
        place_id = arr[5]

        ix, iy = prep_xy(x, y)

        grid[(ix, iy)][place_id] += 1

    f.close()

    # Sort array
    for el in grid:
        grid_sorted[el] = nlargest(3, sorted(grid[el].items()), key=itemgetter(1))

    print('Generate submission...')
    sub_file = os.path.join('submission_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
    out = open(sub_file, "w")
    f = open("../input/test.csv", "r")
    f.readline()
    total = 0
    out.write("row_id,place_id\n")

    while 1:
        line = f.readline().strip()
        total += 1

        if line == '':
            break

        arr = line.split(",")
        row_id = arr[0]
        x = float(arr[1])
        y = float(arr[2])
        accuracy = arr[3]
        time = arr[4]

        out.write(str(row_id) + ',')

        ix, iy = prep_xy(x, y)

        s1 = (ix, iy)
        if s1 in grid_sorted:
            topitems = grid_sorted[s1]
            for i in range(len(topitems)):
                out.write(' ' + topitems[i][0])

        out.write("\n")

    out.close()
    f.close()


start_time = time.time()
run_solution()
print("Elapsed time overall: %s seconds" % (time.time() - start_time))
