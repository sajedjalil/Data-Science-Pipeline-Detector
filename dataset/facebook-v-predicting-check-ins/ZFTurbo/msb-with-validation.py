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
    grid_size_x = 285
    grid_size_y = 725
    # Maximum T = 786239. Take -10% of it
    split_t = math.floor((1.0 - 0.1) * 786239)
    out_of_business_time = 0.1
    split_out_of_business = math.floor((1.0 - out_of_business_time) * 786239)
    split_out_of_business_valid = math.floor((1.0 - out_of_business_time) * split_t)
    test_arr = []

    grid = defaultdict(lambda: defaultdict(int))
    grid_valid = defaultdict(lambda: defaultdict(int))
    working_place = dict()
    working_place_valid = dict()
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
        row_id = arr[0]
        x = float(arr[1])
        y = float(arr[2])
        accuracy = int(arr[3])
        time1 = int(arr[4])
        place_id = arr[5]
        time_of_week = time1 % 10080
        quarter_period_of_day = math.floor((time1 + 120) / (6*60)) % 4
        # day_of_week = math.floor(time_of_week / 1440)
        # hour_of_day = math.floor(time_of_week / 60) % 24
        log_month = math.log10(3+((time1 + 120.0) / (60 * 24 * 30)))
        if accuracy > 100:
            log_month = log_month*0.5
        
        ix, iy = prep_xy(x, y, grid_size_x, grid_size_y)
        grid[(ix, iy, quarter_period_of_day)][place_id] += log_month
        grid[(ix, iy)][place_id] += log_month
        
        if time1 < split_t:
            grid_valid[(ix, iy, quarter_period_of_day)][place_id] += log_month
            grid_valid[(ix, iy)][place_id] += log_month
            train_samples += 1
            if time1 >= split_out_of_business_valid:
                working_place_valid[place_id] = 1
        else:
            test_arr.append(arr)
            test_samples += 1

        if time1 >= split_out_of_business:
            working_place[place_id] = 1

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
    accuracy_test = defaultdict(int)
    accuracy_test_count = defaultdict(int)
    for arr in test_arr:
        total += 1

        row_id = arr[0]
        x = float(arr[1])
        y = float(arr[2])
        accuracy = int(arr[3])
        time1 = int(arr[4])
        place_id = arr[5]
        time_of_week = time1 % 10080
        quarter_period_of_day = math.floor((time1 + 120) / (6*60)) % 4

        filled = []

        ix, iy = prep_xy(x, y, grid_size_x, grid_size_y)

        s1 = (ix, iy, quarter_period_of_day)
        s2 = (ix, iy)
        
        if len(filled) < 3 and s1 in grid_sorted_valid:
            topitems = grid_sorted_valid[s1]
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 3:
                    break
                if topitems[i][0] in working_place_valid:
                    filled.append(topitems[i][0])
        if len(filled) < 3 and s2 in grid_sorted_valid:
            topitems = grid_sorted_valid[s2]
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 3:
                    break
                if topitems[i][0] in working_place_valid:
                    filled.append(topitems[i][0])

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
        
        scr = apk([place_id], filled, 3)
        accuracy_test[scr] += accuracy
        accuracy_test_count[scr] += 1
        score += scr
        score_num += 1

    f.close()
    score /= score_num
    for el in accuracy_test:
        print('Avg accuracy {}: {}'.format(el, accuracy_test[el]/accuracy_test_count[el]))
    
    print('Predicted score: {}'.format(score))
    print('Train samples: ', train_samples)
    print('Test samples: ', test_samples)

    print('Generate submission...')
    sub_file = os.path.join('submission_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
    out = open(sub_file, "w")
    f = open("../input/test.csv", "r")
    f.readline()
    total = 0
    count_empty = 0
    second_pass_count = 0
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
        time1 = int(arr[4])
        quarter_period_of_day = math.floor((time1 + 120) / (6*60)) % 4

        out.write(str(row_id) + ',')
        filled = []

        ix, iy = prep_xy(x, y, grid_size_x, grid_size_y)

        s1 = (ix, iy, quarter_period_of_day)
        s2 = (ix, iy)
        if len(filled) < 3 and s1 in grid_sorted:
            topitems = grid_sorted[s1]
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 3:
                    break
                if topitems[i][0] in working_place:
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])
                    
        if len(filled) < 3 and s2 in grid_sorted:
            topitems = grid_sorted[s2]
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 3:
                    break
                if topitems[i][0] in working_place:
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])
        
        if len(filled) < 3 and s1 in grid_sorted:
            topitems = grid_sorted[s1]
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 3:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                second_pass_count += 1
                
        if len(filled) < 3 and s2 in grid_sorted:
            topitems = grid_sorted[s2]
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 3:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                second_pass_count += 1
        
        if len(filled) < 3:
            count_empty += 1
        out.write("\n")

    print('Empty cases:', str(count_empty))
    print('Is second pass nesessary?:', str(second_pass_count))
    out.close()
    f.close()


start_time = time.time()
run_solution()
print("Elapsed time overall: %s seconds" % (time.time() - start_time))

# Best: 0.4781861319422314