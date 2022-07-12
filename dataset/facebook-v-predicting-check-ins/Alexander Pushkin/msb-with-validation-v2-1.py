# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
from heapq import nlargest
from operator import itemgetter
import os
import time
import math
from collections import defaultdict


def apk(actual, predicted, k=3):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def prep_xy(x, y, range_x, range_y):
    ix = math.floor(range_x*x/10)
    if ix < 0:
        ix = 0
    if ix >= range_x:
        ix = range_x-1

    iy = math.floor(range_y*y/10)
    if iy < 0:
        iy = 0
    if iy >= range_y:
        iy = range_y-1

    return ix, iy


def run_solution():
    print('Preparing arrays...')
    f = open("../input/train.csv", "r")
    f.readline()
    total = 0
    grid_size_x = 500
    grid_size_y = 1000
    grid_size_x2 = 100
    grid_size_y2 = 200
    # Maximum T = 786239. Take -10% of it
    split_t = 707616
    test_arr = []

    grid = defaultdict(lambda: defaultdict(int))
    grid_valid = defaultdict(lambda: defaultdict(int))
    grid_sorted = dict()
    grid_sorted_valid = dict()

    # Calc counts
    train_samples = 0
    test_samples = 0
    while 1:
        line = f.readline().strip()
        total += 1

        if line == '':
            break

        arr = line.split(",")
        #row_id = arr[0]
        x = float(arr[1])
        y = float(arr[2])
        #accuracy = int(arr[3])
        time1 = int(arr[4])
        place_id = arr[5]
        quarter_period_of_day = math.floor((time1 - 60) / (6*60)) % 4
        #day_of_week = math.floor((time1 % 10080) / 1440) 
        #if (day_of_week == 0 or day_of_week == 1):
        #    weekend = 1
        #else:
        #    weekend = 0
        #hour_of_day = math.floor(time_of_week / 60) % 24

        ix, iy = prep_xy(x, y, grid_size_x, grid_size_y)
        ix2, iy2 = prep_xy(x, y, grid_size_x2, grid_size_y2)
        #grid[(ix, iy, quarter_period_of_day, weekend)][place_id] += 1
        grid[(ix, iy, quarter_period_of_day)][place_id] += 1
        grid[(ix, iy)][place_id] += 1
        grid[((ix2+10000), (iy2+10000))][place_id] += 1
        
        if time1 < split_t:
            #grid_valid[(ix, iy, quarter_period_of_day,weekend)][place_id] += 1
            grid_valid[(ix, iy, quarter_period_of_day)][place_id] += 1
            grid_valid[(ix, iy)][place_id] += 1
            grid_valid[((ix2+10000), (iy2+10000))][place_id] += 1
            train_samples += 1
        else:
            test_arr.append(arr)
            test_samples += 1

    f.close()

    print('Sorting arrays...')
    for el in grid:
        grid_sorted[el] = nlargest(3, sorted(grid[el].items()), key=itemgetter(1))
    for el in grid_valid:
        grid_sorted_valid[el] = nlargest(3, sorted(grid_valid[el].items()), key=itemgetter(1))

    print('Run validation...')
    total = 0
    score = 0.0
    score_num = 0
    for arr in test_arr:
        total += 1

        #row_id = arr[0]
        x = float(arr[1])
        y = float(arr[2])
        #accuracy = int(arr[3])
        time1 = int(arr[4])
        place_id = arr[5]
        quarter_period_of_day = math.floor((time1 - 60) / (6*60)) % 4
        #day_of_week = math.floor((time1 % 10080) / 1440) 
        #if (day_of_week == 0 or day_of_week == 1):
        #    weekend = 1
        #else:
        #    weekend = 0

        filled = []

        ix, iy = prep_xy(x, y, grid_size_x, grid_size_y)
        ix2, iy2 = prep_xy(x, y, grid_size_x2, grid_size_y2)

        #s0 = (ix, iy, quarter_period_of_day, weekend)
        s1 = (ix, iy, quarter_period_of_day)
        s2 = (ix, iy)
        s3 = ((ix2+10000), (iy2+10000))
        #if s0 in grid_sorted_valid:
        #    topitems = grid_sorted_valid[s0]
        #    for i in range(len(topitems)):
        #        if topitems[i][0] in filled:
        #            continue
        #        if len(filled) == 3:
        #            break
        #        filled.append(topitems[i][0])
        if len(filled) < 3 and s1 in grid_sorted_valid:
            topitems = grid_sorted_valid[s1]
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 3:
                    break
                filled.append(topitems[i][0])
        if len(filled) < 3 and s2 in grid_sorted_valid:
            topitems = grid_sorted_valid[s2]
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 3:
                    break
                filled.append(topitems[i][0])
        if len(filled) < 3 and s3 in grid_sorted_valid:
            topitems = grid_sorted_valid[s3]
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 3:
                    break
                filled.append(topitems[i][0])

        score += apk([place_id], filled, 3)
        score_num += 1

    f.close()
    score /= score_num
    print('Predicted score: {}'.format(score))
    print('Train samples: ', train_samples)
    print('Test samples: ', test_samples)

    print('Generate submission...')
    sub_file = os.path.join('submission_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
    out = open(sub_file, "w")
    f = open("../input/test.csv", "r")
    f.readline()
    total = 0
    count_empty0 = 0
    count_empty1 = 0
    count_empty2 = 0
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
        #accuracy = int(arr[3])
        time1 = int(arr[4])
        quarter_period_of_day = math.floor((time1 - 60) / (6*60)) % 4
        #day_of_week = math.floor((time1 % 10080) / 1440) 
        #weekend = 0
        #if (day_of_week == 0 or day_of_week == 1):
        #    weekend = 1

        out.write(str(row_id) + ',')
        filled = []

        ix, iy = prep_xy(x, y, grid_size_x, grid_size_y)
        ix2, iy2 = prep_xy(x, y, grid_size_x2, grid_size_y2)

        #s0 = (ix, iy, quarter_period_of_day, weekend)
        s1 = (ix, iy, quarter_period_of_day)
        s2 = (ix, iy)
        s3 = ((ix2+10000), (iy2+10000))
        #if s0 in grid_sorted:
        #    topitems = grid_sorted[s0]
        #    for i in range(len(topitems)):
        #        if topitems[i][0] in filled:
        #            continue
        #        if len(filled) == 3:
        #            break
        #        out.write(' ' + topitems[i][0])
        #        filled.append(topitems[i][0])
        if len(filled) < 3 and s1 in grid_sorted:
            topitems = grid_sorted[s1]
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 3:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
        if len(filled) < 3 and s2 in grid_sorted:
            topitems = grid_sorted[s2]
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 3:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
        if len(filled) < 3 and s3 in grid_sorted_valid:
            topitems = grid_sorted_valid[s3]
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 3:
                    break
                filled.append(topitems[i][0])
        
        if len(filled) == 0:
            count_empty0 += 1
        if len(filled) == 1:
            count_empty1 += 1
        if len(filled) == 2:
            count_empty2 += 1
        out.write("\n")

    print('Empty0 cases:', str(count_empty0))
    print('Empty1 cases:', str(count_empty1))
    print('Empty2 cases:', str(count_empty2))
    out.close()
    f.close()


start_time = time.time()
run_solution()
print("Elapsed time overall: %s seconds" % (time.time() - start_time))