
import datetime
from heapq import nlargest
from operator import itemgetter


def prepare_arrays_match():
    f = open("../input/train.csv", "r")
    f.readline()
    
    best_hotels_od_ulc = dict()
    best_hotels_uid_miss = dict()
    best_hotels_search_dest0 = dict()
    best_hotels_search_dest1 = dict()
    best_hotels_country = dict()
    popular_hotel_cluster = dict()
    best_s00 = dict()
    best_s01 = dict()
    known_cities = dict()
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
        user_location_country = arr[3]
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        user_id = arr[7]
        srch_destination_id = arr[16]
        hotel_country = arr[21]
        hotel_market = arr[22]
        is_booking = float(arr[18])
        hotel_cluster = arr[23]




        if user_location_city != '' and orig_destination_distance != '':
            s1 = (user_location_country,user_location_city, orig_destination_distance, hotel_market)

            if s1 in best_hotels_od_ulc:
                if hotel_cluster in best_hotels_od_ulc[s1]:
                    best_hotels_od_ulc[s1][hotel_cluster] += 1
                else:
                    best_hotels_od_ulc[s1][hotel_cluster] = 1
            else:
                best_hotels_od_ulc[s1] = dict()
                best_hotels_od_ulc[s1][hotel_cluster] = 1
                
            s11 = (user_location_country,user_location_city)

            if s11 in known_cities:
                if hotel_cluster in known_cities[s11]:
                    known_cities[s11][hotel_cluster] += 1
                else:
                    known_cities[s11][hotel_cluster]  = 1
            else:
                known_cities[s11] = dict()
                known_cities[s11][hotel_cluster] = 1
            	    

		
    f.close()
    return known_cities,best_s00,best_s01, best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest0,best_hotels_search_dest1, popular_hotel_cluster


def gen_submission(known_cities,best_s00, best_s01,best_hotels_country, best_hotels_search_dest0,best_hotels_search_dest1, best_hotels_od_ulc, best_hotels_uid_miss, popular_hotel_cluster):
    now = datetime.datetime.now()
    path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    f = open("../input/test.csv", "r")
    f.readline()
    total = 0
    total0 = 0
    total00 = 0
    total1 = 0
    total20 = 0
    total21 = 0
    total3 = 0
    total4 = 0
    total_known = 0
    total_unknown = 0
    
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
        user_location_country = arr[4]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        user_id = arr[8]
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]

        out.write(str(id) + ',')
        filled = []

        if orig_destination_distance != '':
            s1 = (user_location_country,user_location_city, orig_destination_distance, hotel_market)
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
        else:	
            s11 = (user_location_country,user_location_city)

            if s11 in known_cities:
                total_known += 1
            else:
                total_unknown += 1

		 

    out.write("\n")
    out.close()
    print('Total 1: {} ...'.format(total1))
    print('Total 0: {} ...'.format(total0))
    print('Total 00: {} ...'.format(total00))
    print('Total 20: {} ...'.format(total20))
    print('Total 21: {} ...'.format(total21))
    print('Total 3: {} ...'.format(total3))
    print('Total 4: {} ...'.format(total4))
    print('Total known: {} ...'.format(total_known))
    print('Total unknown: {} ...'.format(total_unknown))


known_cities,best_s00,best_s01,best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest0,best_hotels_search_dest1, popular_hotel_cluster = prepare_arrays_match()
gen_submission(known_cities,best_s00, best_s01,best_hotels_country, best_hotels_search_dest0,best_hotels_search_dest1, best_hotels_od_ulc, best_hotels_uid_miss, popular_hotel_cluster)


