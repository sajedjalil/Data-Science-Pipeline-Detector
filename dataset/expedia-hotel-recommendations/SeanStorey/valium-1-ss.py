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


def populate_top_helper(filled, inputDict, output_file):
    added = 0
    d = inputDict
    topItems = nlargest(5, sorted(d.items()), key=itemgetter(1))
    for i in range(len(topItems)):
        if topItems[i][0] in filled:
            continue
        if len(filled) == 5:
            break
        output_file.write(' ' + topItems[i][0])
        filled.append(topItems[i][0])
        added += 1

    return added


def get_stats_train():
    f = open(input_file_name, "r")
    f.readline()

    u_city_zero = 0
    distance_zero = 0
    u_id_zero = 0
    sdi_zero = 0
    sdt_zero = 0
    h_country_zero = 0
    h_market_zero = 0
    h_cluster_zero = 0

    # Calc counts
    while 1:
        line = f.readline().strip()

        if line == '':
            break

        arr = line.split(",")

        u_city = arr[5]
        distance = arr[6]
        u_id = arr[7]
        sdi = arr[16]  # search destination id
        sdt = arr[17]  # search destination id
        h_country = arr[21]
        h_market = arr[22]
        h_cluster = arr[23]


        if u_city == '':
            u_city_zero += 1

        if distance == '':
            distance_zero += 1

        if u_id == '':
            u_id_zero += 1

        if sdi == '':
            sdi_zero += 1

        if sdt == '':
            sdt_zero += 1

        if h_country == '':
            h_country_zero += 1

        if h_market == '':
            h_market_zero += 1

        if h_cluster == '':
            h_cluster_zero += 1

    print('u_city zero ...{}'.format(u_city_zero))
    print('distance zero ...{}'.format(distance_zero))
    print('u_id zero ...{}'.format(u_id_zero))
    print('sdi zero ...{}'.format(sdi_zero))
    print('sdt zero ...{}'.format(sdt_zero))
    print('h_country zero ...{}'.format(h_country_zero))
    print('h_market zero ...{}'.format(h_market_zero))
    print('h_cluster zero ...{}'.format(h_cluster_zero))

def get_stats_test():
    f = open("D:/Data/ML/Kaggle/Expedia/test.csv", "r")
    f.readline()

    u_city_zero = 0
    distance_zero = 0
    u_id_zero = 0
    sdi_zero = 0
    sdt_zero = 0
    h_country_zero = 0
    h_market_zero = 0
    h_cluster_zero = 0

    # Calc counts
    while 1:
        line = f.readline().strip()

        if line == '':
            break

        arr = line.split(",")

        u_city = arr[6]
        distance = arr[7]
        u_id = arr[8]
        sdi = arr[17]  # search destination id
        sdt = arr[18]  # search destination id
        h_country = arr[20]
        h_market = arr[21]

        if u_city == '':
            u_city_zero += 1

        if distance == '':
            distance_zero += 1

        if u_id == '':
            u_id_zero += 1

        if sdi == '':
            sdi_zero += 1

        if sdt == '':
            sdt_zero += 1

        if h_country == '':
            h_country_zero += 1

        if h_market == '':
            h_market_zero += 1


    print('u_city zero ...{}'.format(u_city_zero))
    print('distance zero ...{}'.format(distance_zero))
    print('u_id zero ...{}'.format(u_id_zero))
    print('sdi zero ...{}'.format(sdi_zero))
    print('sdt zero ...{}'.format(sdt_zero))
    print('h_country zero ...{}'.format(h_country_zero))
    print('h_market zero ...{}'.format(h_market_zero))



def prepare_arrays_match():
    f = open(input_file_name, "r")
    f.readline()

    by_distance_u_city = dict()
    best_hotels_uid_miss = dict()
    best_hotels_search_dest = dict()
    best_hotels_country = dict()
    popular_hotel_cluster = dict()
    best_s01 = dict()
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

        if arr[11] != '':
            book_year = int(arr[11][:4])
            book_month = int(arr[11][5:7])
        else:
            book_year = int(arr[0][:4])
            book_month = int(arr[0][5:7])

        if book_month < 1 or book_month > 12 or book_year < 2012 or book_year > 2015:
            continue

        u_city = arr[5]
        distance = arr[6]
        u_id = arr[7]
        sdi = arr[16]  # search destination id
        sdt = arr[17]  # search destination type
        h_country = arr[21]
        h_market = arr[22]
        is_booking = float(arr[18])
        h_cluster = arr[23]

        append_0 = ((book_year - 2012) * 12 + (book_month - 12))
        if not (0 < append_0 <= 36):
            continue

        append_1 = pow(math.log(append_0), 1.35) * (-0.09+0.96*pow(append_0, 1.46)) * (3.5 + 17.55*is_booking)
        append_2 = 3 + 5.56*is_booking

        if u_city != '' and distance != '':
            s1 = (u_city, distance)
            add_weighting(s1, by_distance_u_city, h_cluster, append_0)

        if sdi != '' and sdt != '' and h_country != '' and h_market != '':
            s2 = (sdi, sdt, h_country, h_market)
            add_weighting(s2, best_hotels_search_dest, h_cluster, append_1)

        if h_market != '':
            s3 = h_market
            add_weighting(s3, best_hotels_country, h_cluster, append_2)

        if u_city != '' and distance != '' and u_id != '' and sdi != '' and is_booking == 1:
            s01 = (u_id, sdi, h_country, h_market)
            add_weighting(s01, best_s01, h_cluster, append_0)

        if u_city != '' and distance == '' and u_id != '' and sdi != '' and h_country != '' and is_booking == 1:
            s0 = (u_id, u_city, sdi, h_country, h_market)
            add_weighting(s0, best_hotels_uid_miss, h_cluster, append_0)

        if h_cluster in popular_hotel_cluster:
            popular_hotel_cluster[h_cluster] += append_0
        else:
            popular_hotel_cluster[h_cluster] = append_0

    f.close()
    return best_s01, best_hotels_country, by_distance_u_city, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster




def gen_submission(best_s01, best_hotels_country, best_hotels_search_dest, by_distance_u_city,
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
    out.write("id,h_cluster\n")
    topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 100000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        test_id = arr[0]
        u_city = arr[6]
        distance = arr[7]
        u_id = arr[8]
        sdi = arr[17]
        sdt = arr[18]
        h_country = arr[20]
        h_market = arr[21]

        out.write(str(test_id) + ',')
        filled = []

        if distance != '':
            s1 = (u_city, distance)
            if s1 in by_distance_u_city:
                total1 += populate_top_helper(filled, by_distance_u_city[s1], out)
        else:
            s1 = (u_id, u_city, sdi, h_country, h_market)
            if s1 in best_hotels_uid_miss:
                total0 += populate_top_helper(filled, best_hotels_uid_miss[s1], out)

        s00 = (u_id, u_city, sdi, h_country, h_market)
        s01 = (u_id, sdi, h_country, h_market)
        if s01 in best_s01 and s00 not in best_hotels_uid_miss:
            total00 += populate_top_helper(filled, best_s01[s01], out)

        s2 = (sdi, sdt, h_country, h_market)
        if s2 in best_hotels_search_dest:
            total2 += populate_top_helper(filled, best_hotels_search_dest[s2], out)

        if h_market in best_hotels_country:
            total3 += populate_top_helper(filled, best_hotels_country[h_market], out)

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


best_s01, best_hotels_country, by_distance_u_city, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster = prepare_arrays_match()
gen_submission(best_s01, best_hotels_country, best_hotels_search_dest, by_distance_u_city, best_hotels_uid_miss, popular_hotel_cluster)
get_stats_train()
get_stats_test()

