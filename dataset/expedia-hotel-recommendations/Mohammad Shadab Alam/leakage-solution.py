# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
from heapq import nlargest
from operator import itemgetter


def prepare_arrays_match():
    f = open("../input/train.csv", "r")
    f.readline()
    best_hotels_od_ulc = dict()
    best_hotels_search_dest = dict()
    popular_hotel_cluster = dict()
    total = 0

    # Calc counts
    while 1:
        line = f.readline().strip()
        total += 1

        if total % 2000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        srch_destination_id = arr[16]
        hotel_country = arr[21]
        hotel_market = arr[22]
        is_booking = float(arr[18])
        hotel_cluster = arr[23]

        
        if srch_destination_id != '' and hotel_country != '' and hotel_market != '':
            s2 = (srch_destination_id,hotel_country,hotel_market)
            if s2 in best_hotels_search_dest:
                if hotel_cluster in best_hotels_search_dest[s2]:
                    best_hotels_search_dest[s2][hotel_cluster] += is_booking*1 + (1-is_booking)*0.15
                else:
                    best_hotels_search_dest[s2][hotel_cluster] = is_booking*1 + (1-is_booking)*0.15
            else:
                best_hotels_search_dest[s2] = dict()
                best_hotels_search_dest[s2][hotel_cluster] = is_booking*1 + (1-is_booking)*0.15



    f.close()
    return  best_hotels_search_dest


def gen_submission(best_hotels_search_dest):
    now = datetime.datetime.now()
    path = 'lolz_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    f = open("../input/test.csv", "r")
    f.readline()
    total = 0
    out.write("id,hotel_cluster\n")
    #topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 100000 == 0:
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

        out.write(str(id) + ',')
        filled = []



        s2 = (srch_destination_id,hotel_country,hotel_market)
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
        out.write("\n")
    out.close()


best_hotels_search_dest = prepare_arrays_match()
gen_submission(best_hotels_search_dest)