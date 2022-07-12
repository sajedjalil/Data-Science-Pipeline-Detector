# coding: utf-8
__author__ = 'Ravi: https://kaggle.com/company'

import datetime
from heapq import nlargest
from operator import itemgetter


def prepare_arrays_match():
    f = open("../input/train.csv", "r")
    f.readline()
    best_hotels_od_ulc = dict()
    best_hotels_search_dest = dict()
    best_hotels_country = dict()
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

        if user_location_city != '' and orig_destination_distance != '':
            s1 = (user_location_city, orig_destination_distance)

            if s1 in best_hotels_od_ulc:
                if hotel_cluster in best_hotels_od_ulc[s1]:
                    best_hotels_od_ulc[s1][hotel_cluster] += 1
                else:
                    best_hotels_od_ulc[s1][hotel_cluster] = 1
            else:
                best_hotels_od_ulc[s1] = dict()
                best_hotels_od_ulc[s1][hotel_cluster] = 1

        if srch_destination_id != '' and hotel_country != '' and hotel_market != '':
            s2 = (srch_destination_id,hotel_country,hotel_market)
            if s2 in best_hotels_search_dest:
                if hotel_cluster in best_hotels_search_dest[s2]:
                    best_hotels_search_dest[s2][hotel_cluster] += is_booking*1 + (1-is_booking)*0.15443
                else:
                    best_hotels_search_dest[s2][hotel_cluster] = is_booking*1 + (1-is_booking)*0.15443
            else:
                best_hotels_search_dest[s2] = dict()
                best_hotels_search_dest[s2][hotel_cluster] = is_booking*1 + (1-is_booking)*0.15443

        if hotel_country != '':
            s3 = (hotel_country)
            if s3 in best_hotels_country:
                if hotel_cluster in best_hotels_country[s3]:
                    best_hotels_country[s3][hotel_cluster] += is_booking*1 + (1-is_booking)*0.0834
                else:
                    best_hotels_country[s3][hotel_cluster] = is_booking*1 + (1-is_booking)*0.0834
            else:
                best_hotels_country[s3] = dict()
                best_hotels_country[s3][hotel_cluster] = is_booking*1 + (1-is_booking)*0.0834

        if hotel_cluster in popular_hotel_cluster:
            popular_hotel_cluster[hotel_cluster] += 1
        else:
            popular_hotel_cluster[hotel_cluster] = 1

    f.close()
    return best_hotels_country, best_hotels_od_ulc, best_hotels_search_dest, popular_hotel_cluster


def gen_submission(best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc, popular_hotel_cluster):
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

        s3 = (hotel_country)
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
                
        for i in range(len(topclasters)):
            if topclasters[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            out.write(' ' + topclasters[i][0])
            filled.append(topclasters[i][0])

        out.write("\n")
    out.close()


best_hotels_country, best_hotels_od_ulc, best_hotels_search_dest, popular_hotel_cluster = prepare_arrays_match()
gen_submission(best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc, popular_hotel_cluster)