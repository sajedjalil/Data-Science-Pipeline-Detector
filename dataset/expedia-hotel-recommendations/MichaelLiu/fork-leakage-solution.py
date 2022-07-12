# coding: utf-8

import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict

p_train = "../input/train.csv"
p_test = "../input/test.csv"

best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))
best_hotels_search_dest = defaultdict(lambda: defaultdict(int))
popular_hotel_cluster = defaultdict(int)

def prepare_arrays_match():
    print('Preparing arrays...')
    total = 0

    with open(p_train) as f:
      f.readline()
      for line in f:
        total += 1
        if total % 1000000 == 0:
            print('Read {} lines...'.format(total))

        arr = line.strip().split(",")
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        is_package = arr[9]
        srch_destination_id = arr[16]
        srch_destination_type = arr[17]
        is_booking = int(arr[18])
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = arr[23]
        append_1 = 3 + 17*is_booking
        '''
        if user_location_city and orig_destination_distance:
            orig_destination_distance = str(int(float(orig_destination_distance)))
            best_hotels_od_ulc[(user_location_city, orig_destination_distance)][hotel_cluster] += append_1
        if srch_destination_id and hotel_country and hotel_market:
            best_hotels_search_dest[(srch_destination_id, hotel_country, hotel_market)][hotel_cluster] += append_1
        '''
        if srch_destination_id and is_package and hotel_market:
            s2 = (srch_destination_id, is_package, hotel_market)
            best_hotels_od_ulc[s2][hotel_cluster] += append_1
        popular_hotel_cluster[hotel_cluster] += append_1


def gen_submission():
    print('Generate submission...')
    now = datetime.datetime.now()
    path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    total = 0
    topclusters = [k for k, v in sorted(popular_hotel_cluster.items(), key=lambda d:-d[1])[:5]]
    
    with open(path, "w") as out:
     out.write("id,hotel_cluster\n")

     with open(p_test) as f:
      f.readline()
      for line in f:
        total += 1
        if total % 100000 == 0:
            print('Write {} lines...'.format(total))

        arr = line.strip().split(",")
        id = arr[0]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        is_package = arr[10]
        srch_destination_id = arr[17]
        srch_destination_type = arr[18]
        hotel_country = arr[20]
        hotel_market = arr[21]

        filled = []
        '''
        if user_location_city and orig_destination_distance:
          orig_destination_distance = str(int(float(orig_destination_distance)))
          s1 = (user_location_city, orig_destination_distance)
          if s1 in best_hotels_od_ulc:
            d = best_hotels_od_ulc[s1]
            topitems = [k for k, v in sorted(d.items(), key=lambda d:-d[1])[:5]]
            for i, v in enumerate(topitems):
                if v not in filled:
                    filled.append(v)
        if srch_destination_id and hotel_country and hotel_market:
          s2 = (srch_destination_id, hotel_country, hotel_market)
          if s2 in best_hotels_search_dest:
            d = best_hotels_search_dest[s2]
            topitems = [k for k, v in sorted(d.items(), key=lambda d:-d[1])[:5]]
            for i, v in enumerate(topitems):
                if v not in filled:
                    filled.append(v)
        '''
        if srch_destination_id and is_package and hotel_market:
          s2 = (srch_destination_id, is_package, hotel_market)
          if s2 in best_hotels_od_ulc:
            d = best_hotels_od_ulc[s2]
            topitems = [k for k, v in sorted(d.items(), key=lambda d:-d[1])[:5]]
            for i, v in enumerate(topitems):
                if v not in filled:
                    filled.append(v)

        if len(filled) < 5:
            for i, v in enumerate(topclusters[:5]):
                if v not in filled:
                    filled.append(v)

        out.write("%s,%s\n" % (id, ' '.join([str(v) for v in filled[:5]])))

prepare_arrays_match()
gen_submission()
