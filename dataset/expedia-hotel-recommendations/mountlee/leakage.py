from __future__ import division
import string
import numpy as np
from numpy.random import randn
from pandas import Series, DataFrame
import pandas as pd
import csv
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import time
import random
import datetime
from sklearn.ensemble import GradientBoostingClassifier
#%matplotlib inline

        
# file = open("../input/train.csv")
# fout = open('subset_datatrain.csv','w')
# n = 0
# fout.write(file.readline())
# for line in file:
#     arr = line.strip().split(',')
#     is_book = int(arr[-6])
#     if is_book == 1:
#         fout.write(line)
# fout.close()
# file.close()



# dategroup = ['2013Jan','2013May','2013Sep','2014Jan','2014May','2014Sep','2015Jan','2015May','2015Sep','2016Jan','2016May','2016Sep']
# chingroup = ['2013Jan','2013May','2013Sep','2014Jan','2014May','2014Sep','2015Jan','2015May','2015Sep','2016Jan','2016May','2016Sep']
# dateix = [[] for i in range(12)]
# chinix = [[] for i in range(12)]

def datedeal(date):
    n = len(date)
    for i in range(n):
        a = date[i]
        if type(a) == type(0.1):
            a = '2015-01-01'
        if int(a[1])>0 or int(a[2])>1 or int(a[2])<1 or int(a[3])>6 or int(a[3])<3:   #大于2016或小于2013的年份全部换成2015
            date[i] = '2015-01-01'
    return pd.to_datetime(date)

def frameDateDeal(frame, datename):
    frame[datename]=frame[datename].fillna('2015-01-01')
    dateix = [[] for i in range(12)]
    datevalue = datedeal(frame[datename].values)
    datevalue = Series(np.arange(len(datevalue)),index = datevalue)
    for i in range(48):
        y = divmod(i,12)[0]
        r = divmod(i,12)[1]
        n = divmod(i,4)[0]
        if r<9:
            dateix[n].extend(datevalue['201'+str(3+y)+'-0'+str(r+1)].values)
        else:
            dateix[n].extend(datevalue['201'+str(3+y)+'-'+str(r+1)].values)
    for i in range(12):
        frame[datename].values[dateix[i]] = i
    return frame

# coding: utf-8

import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict


def run_solution():
    print('Preparing arrays...')
    f = open("../input/train.csv", "r")
    f.readline()
    best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest1 = defaultdict(lambda: defaultdict(int))
    best_hotel_country = defaultdict(lambda: defaultdict(int))
    popular_hotel_cluster = defaultdict(int)
    total = 0

    # Calc counts
    while 1:
        line = f.readline().strip()
        total += 1

        if total % 10000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        book_year = int(arr[0][:4])
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        srch_destination_id = arr[16]
        is_booking = int(arr[18])
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = arr[23]

        append_1 = 3 + 17*is_booking
        append_2 = 1 + 5*is_booking

        if user_location_city != '' and orig_destination_distance != '':
            best_hotels_od_ulc[(user_location_city, orig_destination_distance)][hotel_cluster] += 1

        if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year == 2014:
            best_hotels_search_dest[(srch_destination_id, hotel_country, hotel_market)][hotel_cluster] += append_1
        
        if srch_destination_id != '':
            best_hotels_search_dest1[srch_destination_id][hotel_cluster] += append_1
        
        if hotel_country != '':
            best_hotel_country[hotel_country][hotel_cluster] += append_2
        
        popular_hotel_cluster[hotel_cluster] += 1
    
    f.close()

    print('Generate submission...')
    now = datetime.datetime.now()
    #path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    path = 'predict_test.csv'
    out = open(path, "w")
    out.write('id,hotel_cluster\n')
    f = open("../input/test.csv", "r")
    f.readline()
    total = 0
    #out.write("id,hotel_cluster\n")
    topclasters = nlargest(10, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 1000000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        id = arr[0]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]
        # id = arr[0]
        # user_location_city = arr[5]
        # orig_destination_distance = arr[6]
        # srch_destination_id = arr[16]
        # hotel_country = arr[21]
        # hotel_market = arr[22]


        out.write(str(id) + ',')
        filled = []

        s1 = (user_location_city, orig_destination_distance)
        if s1 in best_hotels_od_ulc:
            d = best_hotels_od_ulc[s1]
            topitems = nlargest(6, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])

        s2 = (srch_destination_id, hotel_country, hotel_market)
        if s2 in best_hotels_search_dest:
            d = best_hotels_search_dest[s2]
            topitems = nlargest(6, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
        # elif srch_destination_id in best_hotels_search_dest1:
        #     d = best_hotels_search_dest1[srch_destination_id]
        #     topitems = nlargest(6, d.items(), key=itemgetter(1))
        #     for i in range(len(topitems)):
        #         if topitems[i][0] in filled:
        #             continue
        #         if len(filled) == 5:
        #             break
        #         out.write(' ' + topitems[i][0])
        #         filled.append(topitems[i][0])

        # if hotel_country in best_hotel_country:
        #     d = best_hotel_country[hotel_country]
        #     topitems = nlargest(6, d.items(), key=itemgetter(1))
        #     for i in range(len(topitems)):
        #         if topitems[i][0] in filled:
        #             continue
        #         if len(filled) == 5:
        #             break
        #         out.write(' ' + topitems[i][0])
        #         filled.append(topitems[i][0])

        # for i in range(len(topclasters)):
        #     if topclasters[i][0] in filled:
        #         continue
        #     if len(filled) == 5:
        #         break
        #     out.write(' ' + topclasters[i][0])
        #     filled.append(topclasters[i][0])

        out.write("\n")
    out.close()
    print('Completed!')


run_solution()

