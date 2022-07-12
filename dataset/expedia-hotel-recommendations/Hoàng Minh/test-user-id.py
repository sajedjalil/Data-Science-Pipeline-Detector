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
    best_hotels_m_ulc = defaultdict(lambda: defaultdict(int))
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
        #book_month = int(arr[0][5:7])
        #book_year = int(arr[0][:4])
        user_id = arr[7]
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        srch_destination_id = arr[16]
        is_booking = int(arr[18])
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = arr[23]

        append_1 = 3 + 17*is_booking
        append_2 = 1 + 5*is_booking

        if user_id != '' and srch_destination_id != '':
            hsh = (hash('user_id_'+str(user_id) + '_srch_destination_id_' + str(srch_destination_id)))
            best_hotels_m_ulc[hsh][hotel_cluster] += append_2
            
        if user_location_city != '' and orig_destination_distance != '':
            hsh = (hash('user_location_city_'+str(user_location_city) + '_orig_destination_distance_' + str(orig_destination_distance)))
            best_hotels_od_ulc[hsh][hotel_cluster] += 1

        if srch_destination_id != '' and hotel_country != '' and hotel_market != '':
            hsh = (hash('srch_destination_id_' + str(srch_destination_id) + "_hotel_country_" + str(hotel_country) + "_hotel_market_"+str(hotel_market)))
            best_hotels_search_dest[hsh][hotel_cluster] += append_1
        
        if srch_destination_id != '':
            hsh = hash('srch_destination_id_'+str(srch_destination_id))
            best_hotels_search_dest1[hsh][hotel_cluster] += append_1
        
        if hotel_country != '':
            hsh = hash('hotel_country_'+str(hotel_country))
            best_hotel_country[hsh][hotel_cluster] += append_2
        
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
        #book_month = int(arr[1][5:7])
        #book_year = int(arr[1][:4])
        user_id = arr[8]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]

        out.write(str(id) + ',')
        filled = []

        hsh = (hash('user_id_'+str(user_id) + '_srch_destination_id_' + str(srch_destination_id)))
        if hsh in best_hotels_m_ulc:
            d = best_hotels_m_ulc[hsh]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                
        hsh = (hash('user_location_city_'+str(user_location_city) + '_orig_destination_distance_' + str(orig_destination_distance)))
        if (len(filled) < 5) and (hsh in best_hotels_od_ulc):
            d = best_hotels_od_ulc[hsh]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])

        hsh = (hash('srch_destination_id_' + str(srch_destination_id) + "_hotel_country_" + str(hotel_country) + "_hotel_market_"+str(hotel_market)))
        hsh1 = hash('srch_destination_id_'+str(srch_destination_id))
        if (len(filled) < 5) and (hsh in best_hotels_search_dest):
            d = best_hotels_search_dest[hsh]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
        elif (len(filled) < 5) and (hsh1 in best_hotels_search_dest1):
            d = best_hotels_search_dest1[hsh1]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])

        hsh = hash('hotel_country_'+str(hotel_country))
        if (len(filled) < 5) and (hsh in best_hotel_country):
            d = best_hotel_country[hsh]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])

        if (len(filled) < 5):
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

if __name__ == "__main__":
    run_solution()