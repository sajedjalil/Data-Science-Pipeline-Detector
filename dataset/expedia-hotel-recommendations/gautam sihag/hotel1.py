# coding: utf-8
__author__ = 'Ravi: https://kaggle.com/company'

import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict

def prepare_arrays_match():
    f = open("../input/train.csv", "r")
    f.readline()
    
    best_hotels_od_ulc = defaultdict(lambda:defaultdict(int))
    best_hotels_uid_miss = defaultdict(lambda:defaultdict(int))
    best_hotels_search_dest = defaultdict(lambda:defaultdict(int))
    best_hotels_country = defaultdict(lambda:defaultdict(int))
    popular_hotel_cluster = defaultdict(int)
    best_s00 = defaultdict(int)
    best_s01 = defaultdict(int)
    total = 0

    # Calc counts

    while total < 32000000:
        line = f.readline().strip()
        total += 1

        if total % 2000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        #book_year = int(arr[0][:4])
        #book_month = int(arr[0][5:7])
        #book_hour = int(arr[0][11:13])
        #srch_adults_cnt = int(arr[13])
        #srch_children_cnt = int(arr[14])
        #srch_rm_cnt = int(arr[15])
        individuals = int(arr[13]) + int(arr[14])
        #user_location_city = arr[5]
        #orig_destination_distance = arr[6]
        #user_id = arr[7]
        #srch_destination_id = arr[16]
        #hotel_country = arr[21]
        #hotel_market = arr[22]
        #srch_destination_type_id = arr[17]
        is_booking = float(arr[18])
        hotel_cluster = arr[23]


        #if int(arr[0][11:13]) >= 10 and int(arr[0][11:13]) < 18:
        append_0 = ((int(arr[0][:4]) - 2012)*12 + (int(arr[0][5:7]) - 12) + (int(arr[0][5:7]) % 4))# + 1)
        #elif int(arr[0][11:13]) >= 18 and int(arr[0][11:13]) < 22:
        #    append_0 = ((int(arr[0][:4]) - 2012)*12 + (int(arr[0][5:7]) - 12) + (int(arr[0][5:7]) % 4) + 2)
        #elif int(arr[0][11:13]) >= 22 and int(arr[0][11:13]) < 24:
        #    append_0 = ((int(arr[0][:4]) - 2012)*12 + (int(arr[0][5:7]) - 12) + (int(arr[0][5:7]) % 4) + 3)
        #elif int(arr[0][11:13]) >= 1 and int(arr[0][11:13]) < 10:
        #    append_0 = ((int(arr[0][:4]) - 2012)*12 + (int(arr[0][5:7]) - 12) + (int(arr[0][5:7]) % 4) + 3)

        append_1 = append_0 * append_0 * (3 + 17.60*is_booking)
        append_2 = 3 + 5.56*is_booking

        if arr[21] != '' and arr[5] != '' and individuals !='' and arr[16] != '' and arr[17] != '' and is_booking==1:
            s00 = (individuals, arr[5], arr[16], arr[21], arr[17])
            if s00 in best_s00:
                if hotel_cluster in best_s00[s00]:
                    best_s00[s00][hotel_cluster] += append_0
                else:
                    best_s00[s00][hotel_cluster] = append_0
            else:
                best_s00[s00] = dict()
                best_s00[s00][hotel_cluster] = append_0

        if arr[21] != '' and individuals !='' and arr[16] != '' and arr[17] != '' and is_booking==1:
            s01 = (individuals, arr[16], arr[21], arr[17])
            if s01 in best_s01:
                if hotel_cluster in best_s01[s01]:
                    best_s01[s01][hotel_cluster] += append_0
                else:
                    best_s01[s01][hotel_cluster] = append_0
            else:
                best_s01[s01] = dict()
                best_s01[s01][hotel_cluster] = append_0


        if arr[5] != '' and individuals !='' and arr[16] != '' and arr[17] != '' and arr[21] != '' and is_booking==1:
            s0 = (individuals, arr[5], arr[16], arr[17], arr[21])
            if s0 in best_hotels_uid_miss:
                if hotel_cluster in best_hotels_uid_miss[s0]:
                    best_hotels_uid_miss[s0][hotel_cluster] += append_0
                else:
                    best_hotels_uid_miss[s0][hotel_cluster] = append_0
            else:
                best_hotels_uid_miss[s0] = dict()
                best_hotels_uid_miss[s0][hotel_cluster] = append_0

        if arr[5] != '' and arr[6] != ''and individuals !='' and arr[16] != '' and arr[16] != '':
            s1 = (arr[5], arr[6], individuals)

            if s1 in best_hotels_od_ulc:
                if hotel_cluster in best_hotels_od_ulc[s1]:
                    best_hotels_od_ulc[s1][hotel_cluster] += append_0
                else:
                    best_hotels_od_ulc[s1][hotel_cluster] = append_0
            else:
                best_hotels_od_ulc[s1] = dict()
                best_hotels_od_ulc[s1][hotel_cluster] = append_0

        if arr[16] != '' and arr[21] != '' and arr[16] != ''and individuals !='':
            s2 = (arr[16],arr[21],arr[17], individuals)
            if s2 in best_hotels_search_dest:
                if hotel_cluster in best_hotels_search_dest[s2]:
                    best_hotels_search_dest[s2][hotel_cluster] += append_1
                else:
                    best_hotels_search_dest[s2][hotel_cluster] = append_1
            else:
                best_hotels_search_dest[s2] = dict()
                best_hotels_search_dest[s2][hotel_cluster] = append_1

        if arr[21] != '':
            s3 = (arr[21])
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
    return best_s00,best_s01, best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster


def gen_submission(best_s00, best_s01,best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc, best_hotels_uid_miss, popular_hotel_cluster):
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
        #user_id = arr[8]
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]
        srch_adults_cnt = int(arr[14])
        srch_children_cnt = int(arr[15])
        #srch_rm_cnt = int(arr[16])
        individuals = srch_adults_cnt + srch_children_cnt
        srch_destination_type_id = arr[18]
        
        

        out.write(str(id) + ',')
        filled = []

        s1 = (user_location_city, orig_destination_distance, individuals)
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
                total1 += 1

        if orig_destination_distance == '':
            s0 = (individuals, user_location_city, srch_destination_type_id, srch_destination_id, hotel_country)
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
                    total0 += 1

        s00 = (individuals, user_location_city, srch_destination_id, hotel_country, srch_destination_type_id)
        s01 = (individuals, srch_destination_id, hotel_country, srch_destination_type_id)
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
                total00 += 1


        s2 = (srch_destination_id,hotel_country,srch_destination_type_id, individuals)
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
                total2 += 1

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
                total3 += 1
                
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


best_s00,best_s01,best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster = prepare_arrays_match()
gen_submission(best_s00, best_s01,best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc, best_hotels_uid_miss, popular_hotel_cluster)