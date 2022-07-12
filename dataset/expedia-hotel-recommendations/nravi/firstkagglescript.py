# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd 

# Input data files are available in the "../input/" directory.

import ml_metrics #to calculate map@5
import random

import datetime
from heapq import nlargest
from operator import itemgetter
import math


#train = open("../input/train.csv", "r")
#train.readline()
#unique_users = train.user_id.unique()

#sample_user_ids = [unique_users[i] for i in sorted(random.sample(range(len(unique_users)), 10000)) ]
#sample_train = train[train.user_id.isin(sample_user_ids)]

#f = sample_train[((sample_train.year == 2013) | ((sample_train.year == 2014) & (sample_train.month < 7)))]
#t = sample_train[((sample_train.year == 2014) & (sample_train.month >= 7))]



def prepare_arrays_match():
    f = open("../input/train.csv", "r")
    f.readline()
    
    best_hotels_od_ulc = dict()
    best_hotels_uid_miss = dict()
    best_hotels_search_dest = dict()
    best_hotels_country = dict()
    popular_hotel_cluster = dict()
    best_s00 = dict()
    best_s01 = dict()

    # Calculate scale
    while 1:
        line = f.readline().strip()

        if line == '':
            break

        arr = line.split(",")
        
        if arr[11] != '':
            book_year = int(arr[11][:4])
            book_month = int(arr[11][5:7])
        else:
            book_year = int(arr[0][:4])
            book_month = int(arr[0][5:7])
            
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        user_id = arr[7]
        is_package = arr[9]
        srch_destination_id = arr[16]
        hotel_country = arr[21]
        hotel_market = arr[22]
        is_booking = float(arr[18])
        hotel_cluster = arr[23]

        append_0 = ((book_year - 2012)*12 + (book_month - 12))
        #if not (append_0>0 and append_0<=36):
            #print(book_year)
            #print(book_month)
            #print(line)
            #print(append_0)
            #continue
        if not (append_0>0 and append_0<=36):
            append_0=0
            continue
        append_1 = pow(math.log(append_0), 1.76) * (-0.1+0.95*pow(append_0, 1.36)) * (3.5*((book_year - 2012)/2) + 21.56*is_booking)
        append_2 = 3 + 5.66*is_booking

        if user_location_city != '' and orig_destination_distance != '' and user_id !='' and srch_destination_id != '' and hotel_country != '' and is_booking==1:
            s00 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
            if s00 in best_s00:
                if hotel_cluster in best_s00[s00]:
                    best_s00[s00][hotel_cluster] += append_0
                else:
                    best_s00[s00][hotel_cluster] = append_0
            else:
                best_s00[s00] = dict()
                best_s00[s00][hotel_cluster] = append_0

        if user_location_city != '' and orig_destination_distance != '' and user_id !='' and srch_destination_id != '' and is_booking==1:
            s01 = (user_id, srch_destination_id, hotel_country, hotel_market)
            if s01 in best_s01:
                if hotel_cluster in best_s01[s01]:
                    best_s01[s01][hotel_cluster] += append_0
                else:
                    best_s01[s01][hotel_cluster] = append_0
            else:
                best_s01[s01] = dict()
                best_s01[s01][hotel_cluster] = append_0


        if user_location_city != '' and orig_destination_distance == '' and user_id !='' and srch_destination_id != '' and hotel_country != '' and is_booking==1:
            s0 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
            if s0 in best_hotels_uid_miss:
                if hotel_cluster in best_hotels_uid_miss[s0]:
                    best_hotels_uid_miss[s0][hotel_cluster] += append_0
                else:
                    best_hotels_uid_miss[s0][hotel_cluster] = append_0
            else:
                best_hotels_uid_miss[s0] = dict()
                best_hotels_uid_miss[s0][hotel_cluster] = append_0

        if user_location_city != '' and orig_destination_distance != '':
            s1 = (user_location_city, orig_destination_distance)

            if s1 in best_hotels_od_ulc:
                if hotel_cluster in best_hotels_od_ulc[s1]:
                    best_hotels_od_ulc[s1][hotel_cluster] += append_0
                else:
                    best_hotels_od_ulc[s1][hotel_cluster] = append_0
            else:
                best_hotels_od_ulc[s1] = dict()
                best_hotels_od_ulc[s1][hotel_cluster] = append_0

        if srch_destination_id != '' and hotel_country != '' and hotel_market != '':
            s2 = (srch_destination_id,hotel_country,hotel_market,is_package)
            if s2 in best_hotels_search_dest:
                if hotel_cluster in best_hotels_search_dest[s2]:
                    best_hotels_search_dest[s2][hotel_cluster] += append_1
                else:
                    best_hotels_search_dest[s2][hotel_cluster] = append_1
            else:
                best_hotels_search_dest[s2] = dict()
                best_hotels_search_dest[s2][hotel_cluster] = append_1

        if hotel_market != '':
            s3 = (hotel_market)
            if s3 in best_hotels_country:
                if hotel_cluster in best_hotels_country[s3]:
                    best_hotels_country[s3][hotel_cluster] += append_2
                else:
                    best_hotels_country[s3][hotel_cluster] = append_2
            else:
                best_hotels_country[s3] = dict()
                best_hotels_country[s3][hotel_cluster] = append_2

        if hotel_cluster in popular_hotel_cluster:
            popular_hotel_cluster[hotel_cluster] += append_0
        else:
            popular_hotel_cluster[hotel_cluster] = append_0

    f.close()
    return best_s00, best_s01, best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster


def gen_sub(best_s00, best_s01,best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc, best_hotels_uid_miss, popular_hotel_cluster):
    now = datetime.datetime.now()
    path = 'submission_' + str(now.strftime("%Y-%m-%d-%H")) + '.csv'
    out = open(path, "w")
    out.write("id,hotel_cluster\n")
    
    t = open("../input/test.csv", "r")
    t.readline()

    topclusters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    while 1:
        line = t.readline().strip()

        if line == '':
            break

        arr = line.split(",")
        id = arr[0]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        user_id = arr[8]
        is_package = arr[10]
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]

        out.write(str(id) + ',')
        filled = []

        s1 = (user_location_city, orig_destination_distance)
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

        if orig_destination_distance == '':
            s0 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
            if s0 in best_hotels_uid_miss:
                d = best_hotels_uid_miss[s0]
                topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

        s00 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
        s01 = (user_id, srch_destination_id, hotel_country, hotel_market)
        if s01 in best_s01 and s00 not in best_s00:
            d = best_s01[s01]
            topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])


        s2 = (srch_destination_id,hotel_country,hotel_market,is_package)
        if s2 in best_hotels_search_dest:
            d = best_hotels_search_dest[s2]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])

        s3 = (hotel_market)
        if s3 in best_hotels_country:
            d = best_hotels_country[s3]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])

        for i in range(len(topclusters)):
            if topclusters[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            out.write(' ' + topclusters[i][0])
            filled.append(topclusters[i][0])

        out.write("\n")
    out.close()
    t.close()


best_s00, best_s01, best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster = prepare_arrays_match()

gen_sub(best_s00, best_s01, best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc, best_hotels_uid_miss, popular_hotel_cluster)

#now = datetime.datetime.now()
#path = 'submission_' + str(now.strftime("%Y-%m-%d-%H")) + '.csv'
#pred = open(path, "r")
#predictions = [[l] for l in pred["hotel_cluster"]]
#target = [[l] for l in t["hotel_cluster"]]
#print(ml_metrics.mapk(target, predictions, k=5))

#predictions.close()
#train.close()


# Any results you write to the current directory are saved as output.