# coding: utf-8
__author__ = 'SeanStorey: https://kaggle.com/company'

import datetime
from heapq import nlargest
from operator import itemgetter
import math

input_file_name = "../input/train.csv"


def add_weighting(key, cluster_mapping, h_cluster, weighting):
    if key in cluster_mapping:
        if h_cluster in cluster_mapping[key]:
            cluster_mapping[key][h_cluster] += weighting
        else:
            cluster_mapping[key][h_cluster] = weighting
    else:
        cluster_mapping[key] = dict()
        cluster_mapping[key][h_cluster] = weighting


def populate_top_helper(filled, inputDict,out):
    added = 0
    d = inputDict
    topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
    for i in range(len(topitems)):
        if topitems[i][0] in filled:
            continue
        if len(filled) == 5:
            break
        out.write(' ' + topitems[i][0])
        filled.append(topitems[i][0])
        added += 1

    return added

def prepare_arrays_match():
    f = open(input_file_name, "r")
    f.readline()

    best_hotels_od_ulc = dict()
    best_hotels_uid_miss = dict()
    best_hotels_search_dest = dict()
    best_hotels_country = dict()
    popular_hotel_cluster = dict()
    best_s01 = dict()
    total = 0

    u_region_missing = 0
    u_country_missing = 0
    u_city_missing = 0
    distance_missing = 0
    u_id_missing = 0
    srch_dest_id_missing = 0
    h_country_missing = 0
    h_market_missing = 0
    is_booking_count = 0
    h_cluster_missing = 0


    # Calc counts
    while True:
    
        line = f.readline().strip()
        total += 1

        
        #break early while testing
        #if total > 100:
        #    break
        
        if total % 2000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            print('Read {} lines...'.format(total))
            break

        arr = line.split(",")

        if arr[11] != '':
            book_year = int(arr[11][:4])
            book_month = int(arr[11][5:7])
        else:
            book_year = int(arr[0][:4])
            book_month = int(arr[0][5:7])

        if book_month < 1 or book_month > 12 or book_year < 2012 or book_year > 2015:
            continue

        u_country       =  arr[3]
        u_region        = arr[4]
        u_city          = arr[5]
        distance        = arr[6]
        u_id            = arr[7]
        srch_dest_id    = arr[16]
        is_booking      = float(arr[18])
        h_country       = arr[21]
        h_market        = arr[22]
        h_cluster       = arr[23]

        if u_country == '': 
            u_country_missing += 1

        if u_region == '' :
            u_region_missing += 1

        if u_city == '' :
            u_city_missing += 1

        if u_city == '' :
            u_city_missing += 1

        if distance == '': 
            distance_missing += 1

        if u_id == '' :
            u_id_missing += 1
        
        if srch_dest_id == '' :
            srch_dest_id_missing += 1
        
        if h_country == '' :
            h_country_missing += 1
        
        if h_market == '' :
            h_market_missing += 1
        
        if is_booking == 1. :
            is_booking_count += 1
        
        if h_cluster == '' :
            h_cluster_missing += 1

        append_0 = ((book_year - 2012) * 12 + (book_month - 12))
    
        if not (0 < append_0 <= 36):
            continue

        append_1 = pow(append_0, 0.45) * append_0 * (3.5 + 17.60 * is_booking)
        append_2 = 3 * math.floor(((book_month + 1) % 12) / 4) + 5.56 * is_booking


        if distance != '':
            key = (u_city, distance)
            add_weighting(key, best_hotels_od_ulc, h_cluster, append_0)
        else:
            if is_booking == 1.:
                key = (u_id, srch_dest_id, h_country, h_market)
                add_weighting(key, best_s01, h_cluster, append_0)
    
                key = (u_id, u_city, srch_dest_id, h_country, h_market)
                add_weighting(key, best_hotels_uid_miss, h_cluster, append_0)

        key = (srch_dest_id, h_country, h_market)
        add_weighting(key, best_hotels_search_dest, h_cluster, append_1)

        add_weighting(h_market, best_hotels_country, h_cluster, append_2)


        if h_cluster in popular_hotel_cluster:
            popular_hotel_cluster[h_cluster] += append_0
        else:
            popular_hotel_cluster[h_cluster] = append_0

    f.close()

    print('u_country_missing:    {}'.format(u_country_missing))
    print('u_region_missing:     {}'.format(u_region_missing))
    print('u_city_missing:       {}'.format(u_city_missing))
    print('distance_missing:     {}'.format(distance_missing))
    print('u_id_missing:         {}'.format(u_id_missing))
    print('srch_dest_id_missing: {}'.format(srch_dest_id_missing))
    print('h_country_missing:    {}'.format(h_country_missing))
    print('h_market_missing:     {}'.format(h_market_missing))
    print('is_booking:           {}'.format(is_booking_count))
    print('h_cluster_missing:    {}'.format(h_cluster_missing))

    print('Len of best_hotels_od_ulc: {} ...'.format(len(best_hotels_od_ulc)))
    print('Len of best_hotels_uid_miss: {} ...'.format(len(best_hotels_uid_miss)))
    print('Len of best_hotels_search_dest: {} ...'.format(len(best_hotels_search_dest)))
    print('Len of best_hotels_country: {} ...'.format(len(best_hotels_country)))
    print('Len of popular_hotel_cluster: {} ...'.format(len(popular_hotel_cluster)))
    print('Len of best_s01: {} ...'.format(len(best_s01)))

    return best_s01, best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster


def gen_submission(best_s01, best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc,
                   best_hotels_uid_miss, popular_hotel_cluster):
    now = datetime.datetime.now()
    path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    f = open("../input/test.csv", "r")
    f.readline()
    total = 0
    total0 = 0
    total00 = 0
    total1 = 0
    total2 = 0
    total3 = 0
    total4 = 0
    out.write("id,hotel_cluster\n")
    topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    u_region_missing = 0
    u_country_missing = 0
    u_city_missing = 0
    distance_missing = 0
    u_id_missing = 0
    srch_dest_id_missing = 0
    h_country_missing = 0
    h_market_missing = 0
    h_cluster_missing = 0


    while 1:
        line = f.readline().strip()
        total += 1

        #break early while testing
        #if total > 10000:
        #    break


        if total % 100000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        id = arr[0]

        u_country =  arr[4]
        u_region =  arr[5]
        u_city = arr[6]
        distance = arr[7]
        u_id = arr[8]
        srch_dest_id = arr[17]
        h_country = arr[20]
        h_market = arr[21]

        if u_country == '' :
            u_country_missing += 1

        if u_region == '' :
            u_region_missing += 1
        
        if u_city == '' :
            u_city_missing += 1

        if distance == '': 
            distance_missing += 1

        if u_id == '' :
            u_id_missing += 1
        
        if srch_dest_id == '' :
            srch_dest_id_missing += 1
        
        if h_country == '' :
            h_country_missing += 1
        
        if h_market == '' :
            h_market_missing += 1

        out.write(str(id) + ',')
        filled = []

        s1 = (u_city, distance)
        if s1 in best_hotels_od_ulc:
            total1 += populate_top_helper(filled, best_hotels_od_ulc[s1],out)
        
        if distance == '':
            s1 = (u_id, u_city, srch_dest_id, h_country, h_market)
            if s1 in best_hotels_uid_miss:
                total0 += populate_top_helper(filled, best_hotels_uid_miss[s1],out)

        s00 = (u_id, u_city, srch_dest_id, h_country, h_market)
        s01 = (u_id, srch_dest_id, h_country, h_market)
        if s01 in best_s01 and s00 not in best_hotels_uid_miss:
            total00 += populate_top_helper(filled, best_s01[s01],out)

        s2 = (srch_dest_id, h_country, h_market)
        if s2 in best_hotels_search_dest:
            total2 += populate_top_helper(filled, best_hotels_search_dest[s2],out)

        if h_market in best_hotels_country:
            total3 += populate_top_helper(filled, best_hotels_country[h_market],out)

        for i in range(len(topclasters)):
            if topclasters[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            out.write(' ' + topclasters[i][0])
            filled.append(topclasters[i][0])
            total4 += 1

        out.write("\n")
    out.close()
    print('Total 1: {} ...'.format(total1))
    print('Total 0: {} ...'.format(total0))
    print('Total 00: {} ...'.format(total00))
    print('Total 2: {} ...'.format(total2))
    print('Total 3: {} ...'.format(total3))
    print('Total 4: {} ...'.format(total4))

    print('u_country_missing:    {}'.format(u_country_missing))
    print('u_region_missing:     {}'.format(u_region_missing))
    print('u_city_missing:       {}'.format(u_city_missing))
    print('distance_missing:     {}'.format(distance_missing))
    print('u_id_missing:         {}'.format(u_id_missing))
    print('srch_dest_id_missing: {}'.format(srch_dest_id_missing))
    print('h_country_missing:    {}'.format(h_country_missing))
    print('h_market_missing:     {}'.format(h_market_missing))
    print('h_cluster_missing:    {}'.format(h_cluster_missing))


best_s01, best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster = prepare_arrays_match()
#gen_submission(best_s01, best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc,
#               best_hotels_uid_miss, popular_hotel_cluster)

