# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

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
    
    k_best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))
    k_best_hotels_search_dest = defaultdict(lambda: defaultdict(int))
    k_best_hotels_search_dest1 = defaultdict(lambda: defaultdict(int))
    k_best_hotel_country = defaultdict(lambda: defaultdict(int))
    
    popular_hotel_cluster = defaultdict(int)
    total = 0

    # Calc counts
    while 1:
        line = f.readline().strip()
        total += 1

        if total % 1000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        book_year = int(arr[0][:4])
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        
        if(len(arr[11]) <5):
            book_month = -1
        else:
            book_month = int(arr[11][5:7])
        srch_kids = int(arr[14])
        srch_destination_id = arr[16]
        is_booking = int(arr[18])
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = arr[23]

        append_1 = 3 + 17*is_booking
        append_2 = 1 + 5*is_booking


        if book_month > 6:
            if user_location_city != '' and orig_destination_distance != '':
                k_best_hotels_od_ulc[(user_location_city, orig_destination_distance)][hotel_cluster] += 1
    
            if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year == 2014:
                k_best_hotels_search_dest[(srch_destination_id, hotel_country, hotel_market)][hotel_cluster] += append_1
            
            if srch_destination_id != '':
                k_best_hotels_search_dest1[srch_destination_id][hotel_cluster] += append_1
            
            if hotel_country != '':
                k_best_hotel_country[hotel_country][hotel_cluster] += append_2
        elif book_month > -1:
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

        if total % 1000000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        id = arr[0]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        srch_kids = int(arr[15])
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]
        if(len(arr[12]) <5):
            print("empty")
            book_month = -1
        else:
            book_month = int(arr[12][5:7])
            
        out.write(str(id) + ',')
        filled = []

        if book_month > 6:
            s1 = (user_location_city, orig_destination_distance)
            if s1 in k_best_hotels_od_ulc:
                d = k_best_hotels_od_ulc[s1]
                topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])
    
            s2 = (srch_destination_id, hotel_country, hotel_market)
            if s2 in k_best_hotels_search_dest:
                d = k_best_hotels_search_dest[s2]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])
            elif srch_destination_id in k_best_hotels_search_dest1:
                d = k_best_hotels_search_dest1[srch_destination_id]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])
    
            if hotel_country in k_best_hotel_country:
                d = k_best_hotel_country[hotel_country]
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
        elif book_month > -1:
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
    
            s2 = (srch_destination_id, hotel_country, hotel_market)
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
            elif srch_destination_id in best_hotels_search_dest1:
                d = best_hotels_search_dest1[srch_destination_id]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])
    
            if hotel_country in best_hotel_country:
                d = best_hotel_country[hotel_country]
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
        else:
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