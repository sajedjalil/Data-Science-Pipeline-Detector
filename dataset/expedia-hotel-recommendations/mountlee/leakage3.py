from __future__ import division
import string
import numpy as np
from numpy.random import randn
from pandas import Series, DataFrame
import pandas as pd
import csv
from scipy import stats
import time
import random
import datetime
#%matplotlib inline

chingroup = ['2013-01','2013-05','2013-09','2014-01','2014-05','2014-09','2015-01','2015-05','2015-09','2016-01','2016-05','2016-09']
chingroup1 = pd.to_datetime(['2013-01','2013-05','2013-09','2014-01','2014-05','2014-09','2015-01','2015-05','2015-09','2016-01','2016-05','2016-09'])
 

import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict




# Calc counts
def run_solution():
    print('Preparing arrays...')
    f = open("../input/train.csv", "r")
    f.readline()
    best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))
    best_hotels_ulc_hlc = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest1 = defaultdict(lambda: defaultdict(int))
    best_hotel_country = defaultdict(lambda: defaultdict(int))
    popular_hotel_cluster = defaultdict(int)
    total = 0    
    while 1:
        line = f.readline().strip()
        total += 1
        if total%1000000 == 0:
            print('Read {} lines...'.format(total))
        #if total == 25000000:
            #break

        if line == '':
            break

        arr = line.split(",")
        book_year = int(arr[0][:4])
        date = (arr[11])
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        srch_destination_id = arr[16]
        is_booking = int(arr[18])
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = arr[23]
        
        if type(date) == type(0.1) or date == '':
            date = '2015-03-15'
        date = int(int(date[5:7])/4)
        
        append_1 = 3 + 20*is_booking
        append_2 = 1 + 5*is_booking
        # chingroup = ['2013Jan','2013May','2013Sep','2014Jan','2014May','2014Sep','2015Jan','2015May','2015Sep','2016Jan','2016May','2016Sep']
        chingroup = ['2013-01','2013-05','2013-09','2014-01','2014-05','2014-09','2015-01','2015-05','2015-09','2016-01','2016-05','2016-09']

        if user_location_city != '' and orig_destination_distance != '':
            best_hotels_od_ulc[(user_location_city, orig_destination_distance,date)][hotel_cluster] += 1     

        #if user_location_city != '' and hotel_market != '': 
            #best_hotels_ulc_hlc[(user_location_city, hotel_market,date)][hotel_cluster] += 1  

        if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year == 2014:
            best_hotels_search_dest[(srch_destination_id, hotel_country, hotel_market,date)][hotel_cluster] += append_1

        if srch_destination_id != '':
            best_hotels_search_dest1[(srch_destination_id,date)][hotel_cluster] += append_1

        if hotel_country != '':
            best_hotel_country[(hotel_country,date)][hotel_cluster] += append_2

        popular_hotel_cluster[hotel_cluster] += 1

    f.close()
    print('train completed!')
    print('Generate submission...')
    now = datetime.datetime.now()
    path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    f = open("../input/test.csv", "r")
    f.readline()
    total = 0
    out.write("id,hotel_cluster\n")
    topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 5000000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        id = arr[0]
        user_location_city = arr[6]
        date = arr[12]
        if type(date) == type(0.1) or date == '':
            date = '2015-03-15'
        date = int(int(date[5:7])/4)
        orig_destination_distance = arr[7]
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]

        out.write(str(id) + ',')
        filled = []

        s1 = (user_location_city, orig_destination_distance,date)
        if s1 in best_hotels_od_ulc:
            d = best_hotels_od_ulc[s1]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])

        # s2 = (user_location_city, hotel_market,date)
        # if s2 in best_hotels_ulc_hlc:
        #     d = best_hotels_ulc_hlc[s2]
        #     topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
        #     for i in range(len(topitems)):
        #         if topitems[i][0] in filled:
        #             continue
        #         if len(filled) == 5:
        #             break
        #         out.write(' ' + topitems[i][0])
        #         filled.append(topitems[i][0])

        s3 = (srch_destination_id, hotel_country, hotel_market,date)
        if s3 in best_hotels_search_dest:
            d = best_hotels_search_dest[s3]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
        elif (srch_destination_id,date) in best_hotels_search_dest1:
            d = best_hotels_search_dest1[(srch_destination_id,date)]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])

        if (hotel_country,date) in best_hotel_country:
            d = best_hotel_country[(hotel_country,date)]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])

        for i in range(len(topclasters)):
            if topclasters[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            out.write(' ' + topclasters[i][0])
            filled.append(topclasters[i][0])

        out.write("\n")
    out.close()
    print('Completed!')

run_solution()