# coding: utf-8
__author__   = 'Ravi: https://kaggle.com/company'
__reviewer__ = 'Bruno: https://kaggle.com/bguberfain'

import datetime
from heapq import nlargest
from operator import itemgetter

#
# FIRST PHASE: Collect statistics of hotel_cluster for some groups
#

# Read each line of train.csv storing data onto dictionaries
f = open("../input/train.csv", "r")
f.readline()

# Dictionaries holding statistics about every group
best_hotels_od_ulc = dict()
best_hotels_uid_miss = dict()
best_hotels_search_dest = dict()
best_hotels_country = dict()
popular_hotel_cluster = dict()
best_s00 = dict()
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

    # Parse line
    arr = line.split(",")
    book_year = int(arr[0][:4])
    book_month = int(arr[0][5:7])
    user_location_city = arr[5]
    orig_destination_distance = arr[6]
    user_id = arr[7]
    srch_destination_id = arr[16]
    hotel_country = arr[21]
    hotel_market = arr[22]
    is_booking = float(arr[18])
    hotel_cluster = arr[23]

    # Number of months since 2011 of booking (will vary between 1 and 24)
    append_0 = ((book_year - 2012) * 12 + (book_month - 12))
    # 'append_1' will vary between 1 and 11865
    # Note: the overall append_0 mean is 15.57. I suspect that the magical 17.60 below has some relation to it
    append_1 = append_0 * append_0 * (3 + 17.60 * is_booking)
    # 'append_2' is only related to is_booking
    append_2 = 3 + 5.56 * is_booking

    # Increase score of a hotel cluster, giving the key (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
    if user_location_city != '' and orig_destination_distance != '' and user_id != '' and srch_destination_id != '' and hotel_country != '' and is_booking == 1:
        s00 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
        if s00 in best_s00:
            if hotel_cluster in best_s00[s00]:
                best_s00[s00][hotel_cluster] += append_0
            else:
                best_s00[s00][hotel_cluster] = append_0
        else:
            best_s00[s00] = dict()
            best_s00[s00][hotel_cluster] = append_0

    # Increase score of a hotel cluster, giving the key (user_id, srch_destination_id, hotel_country, hotel_market)
    if user_location_city != '' and orig_destination_distance != '' and user_id != '' and srch_destination_id != '' and is_booking == 1:
        s01 = (user_id, srch_destination_id, hotel_country, hotel_market)
        if s01 in best_s01:
            if hotel_cluster in best_s01[s01]:
                best_s01[s01][hotel_cluster] += append_0
            else:
                best_s01[s01][hotel_cluster] = append_0
        else:
            best_s01[s01] = dict()
            best_s01[s01][hotel_cluster] = append_0

    # Increase score of a hotel cluster, giving the key (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
    if user_location_city != '' and orig_destination_distance == '' and user_id != '' and srch_destination_id != '' and hotel_country != '' and is_booking == 1:
        s0 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
        if s0 in best_hotels_uid_miss:
            if hotel_cluster in best_hotels_uid_miss[s0]:
                best_hotels_uid_miss[s0][hotel_cluster] += append_0
            else:
                best_hotels_uid_miss[s0][hotel_cluster] = append_0
        else:
            best_hotels_uid_miss[s0] = dict()
            best_hotels_uid_miss[s0][hotel_cluster] = append_0

    # Increase score of a hotel cluster, giving the key (user_location_city, orig_destination_distance)
    # Note: this will take advantage of the leak
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

    # Increase score of a hotel cluster, giving the key (srch_destination_id,hotel_country,hotel_market)
    if srch_destination_id != '' and hotel_country != '' and hotel_market != '':
        s2 = (srch_destination_id, hotel_country, hotel_market)
        if s2 in best_hotels_search_dest:
            if hotel_cluster in best_hotels_search_dest[s2]:
                best_hotels_search_dest[s2][hotel_cluster] += append_1
            else:
                best_hotels_search_dest[s2][hotel_cluster] = append_1
        else:
            best_hotels_search_dest[s2] = dict()
            best_hotels_search_dest[s2][hotel_cluster] = append_1

    # Increase score of a hotel cluster, giving the key (hotel_country)
    if hotel_country != '':
        s3 = (hotel_country)
        if s3 in best_hotels_country:
            if hotel_cluster in best_hotels_country[s3]:
                best_hotels_country[s3][hotel_cluster] += append_2
            else:
                best_hotels_country[s3][hotel_cluster] = append_2
        else:
            best_hotels_country[s3] = dict()
            best_hotels_country[s3][hotel_cluster] = append_2

    # Increase the overall score of a hotel cluster
    if hotel_cluster in popular_hotel_cluster:
        popular_hotel_cluster[hotel_cluster] += append_0
    else:
        popular_hotel_cluster[hotel_cluster] = append_0

f.close()


def create_top5_guess(d, filled, out):
    '''
    This function will produce a guess from a dictionary of scores, filling 'filled' if the guess is not present on it
    :param d: dictionary where keys are hotel_cluster and values are scores
    :param filled: array with hotel_cluster's already predicted
    :param out: write where to put output
    :return: the total number of guesses
    '''
    total = 0
    topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
    for i in range(len(topitems)):
        if topitems[i][0] in filled:
            continue
        if len(filled) == 5:
            break
        out.write(' ' + topitems[i][0])
        filled.append(topitems[i][0])
        total += 1

    return total


# Create submission file
now = datetime.datetime.now()
path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
out = open(path, "w")
out.write("id,hotel_cluster\n")

# Store 'zero' learn choice: the top-5 most popular hotel
topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

total = 0
total0 = 0
total00 = 0
total1 = 0
total2 = 0
total3 = 0
total4 = 0

f = open("../input/test.csv", "r")
f.readline()
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
    user_id = arr[8]
    srch_destination_id = arr[17]
    hotel_country = arr[20]
    hotel_market = arr[21]

    out.write(str(id) + ',')

    # Filled will contain the best guesses, processed in order of importance
    filled = []

    # 1st: explore the data leak (user_location_city, orig_destination_distance)
    s1 = (user_location_city, orig_destination_distance)
    if s1 in best_hotels_od_ulc:
        total1 += create_top5_guess(best_hotels_od_ulc[s1], filled, out)

    # 2nd: using key (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
    s0 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
    if orig_destination_distance == '':
        if s0 in best_hotels_uid_miss:
            total0 += create_top5_guess(best_hotels_uid_miss[s0], filled, out)

    # 3rd: using key (user_id, srch_destination_id, hotel_country, hotel_market), when previous guess is missed
    s01 = (user_id, srch_destination_id, hotel_country, hotel_market)
    if s01 in best_s01 and s0 not in best_s00:
        total00 += create_top5_guess(best_s01[s01], filled, out)

    # 4rd: using key (srch_destination_id, hotel_country, hotel_market)
    s2 = (srch_destination_id, hotel_country, hotel_market)
    if s2 in best_hotels_search_dest:
        total2 += create_top5_guess(best_hotels_search_dest[s2], filled, out)

    # 4rd: using key (hotel_country)
    s3 = (hotel_country)
    if s3 in best_hotels_country:
        total3 += create_top5_guess(best_hotels_country[s3], filled, out)

    # Last: use most popular hotel
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