# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
"""
author: klchang
created time: 2016.6.4
updates:
1. using heap.nlargest instead of sorted and min functions in top_five()
2. add datetime weight

"""
import time
import heapq


def sum_and_count(lst):
    # weight,result = 0.8456, 0.47147; 0.7456, 0.47152; 0.9456, 0.47062; 
    weight = 0.7456
    return sum(lst) * weight + len(lst) * (1 - weight)


def top_five(hc):
    sorted_hc = heapq.nlargest(5, hc, key=lambda x:x[0])
    return sorted_hc


# function: Pre-process the train data
def preprocess(k=5):
    print("Pre-processing train dataset...")
    # data = {}
    search_dest_dict = {}
    popular_hotel_clusters = {}
    total = 0
    with open("../input/train.csv") as fp:
        titles = fp.readline().strip().split(',')
        titles_index = {}
        for i, title in enumerate(titles):
            titles_index[title] = i
        # print(titles)
        # "is_booking","orig_destination_distance","hotel_cluster","srch_destination_id"
        ind_dist = titles_index['orig_destination_distance']
        ind_clus = titles_index['hotel_cluster']
        ind_srch_dest_id = titles_index['srch_destination_id']
        ind_user_loc_city = titles_index['user_location_city']
        ind_booking = titles_index['is_booking']
        ind_datetime = titles_index['date_time']
        
        # list(orig_destination_distance, hotel_cluster)
        dest_id_hotel_cluster_count = {}
        # list(srch_destination_id, hotel_cluster)
        dest_id_hotel_cluster_count2 = {}
        for line in fp:
            fields = line.strip().split(',')
            if not fields:
                break
            total += 1
            if total % 1000000 == 0:
                print("The count that has been processed in train dataset is %s." % total)

            cluster = int(fields[ind_clus])
            if cluster not in popular_hotel_clusters:
                popular_hotel_clusters[cluster] = 1
            else:
                popular_hotel_clusters[cluster] += 1
            # Add the fields "srch_destination_id" and "user_location_city"
            search_dest = fields[ind_srch_dest_id]
            user_city = fields[ind_user_loc_city]
            is_booking = int(fields[ind_booking])
            distance = fields[ind_dist]
            # Add datetime weight
            date_time = fields[ind_datetime]
            if date_time[:4] == '2014':
                is_booking *= 3
            # expedia_train[,sum_and_count(is_booking),by=list(orig_destination_distance, hotel_cluster)]
            count = dest_id_hotel_cluster_count.get((distance, cluster))
            if not count:
                dest_id_hotel_cluster_count[(distance, cluster)] = [is_booking]
            else:
                dest_id_hotel_cluster_count[(distance, cluster)].append(is_booking)
            # expedia_train[,sum_and_count(is_booking),by=list(srch_destination_id, hotel_cluster)]
            count = dest_id_hotel_cluster_count2.get((search_dest, cluster))
            if not count:
                dest_id_hotel_cluster_count2[(search_dest, cluster)] = [is_booking]
            else:
                dest_id_hotel_cluster_count2[(search_dest, cluster)].append(is_booking)

    # dest_id_hotel_cluster_count[,top_five(hotel_cluster,V1),by=orig_destination_distance]
    # group by orig_destination_distance, then sort by V1 - sum_and_count(is_booking)
    for key, value in dest_id_hotel_cluster_count.items():
        dest_id_hotel_cluster_count[key] = sum_and_count(value)
    dest_top_five = {}
    # key = (orig_destination_distance, hotel_cluster)
    for key, value in dest_id_hotel_cluster_count.items():
        if not dest_top_five.get(key[0]):
            dest_top_five[key[0]] = [(value, key[1])]
        else:
            dest_top_five[key[0]].append((value, key[1]))
    for key, value in dest_top_five.items():
        dest_top_five[key] = top_five(value)

    # dest_id_hotel_cluster_count1[,top_five(hotel_cluster,V1),by=srch_destination_id]
    # group by srch_destination_id, then sort by V1 - sum_and_count(is_booking)
    for key, value in dest_id_hotel_cluster_count2.items():
        dest_id_hotel_cluster_count2[key] = sum_and_count(value)
    dest_top_five2 = {}
    # key = (srch_destination_id, hotel_cluster)
    for key, value in dest_id_hotel_cluster_count2.items():
        if not dest_top_five2.get(key[0]):
            dest_top_five2[key[0]] = [(value, key[1])]
        else:
            dest_top_five2[key[0]].append((value, key[1]))
    for key, value in dest_top_five2.items():
        dest_top_five2[key] = top_five(value)
    # Get top k hotel clusters
    top_hotel_clusters = heapq.nlargest(k, popular_hotel_clusters.items(), key=lambda x: x[1])

    data = [dest_top_five, dest_top_five2]

    return [data, top_hotel_clusters]


# function: Compute the k Nearest Neighbor
# user_lst = [[dest_top_five, dest_top_five2], top_hotel_clusters]
# dest_top_five = {orig_destination_distance: [(sum_and_len(is_booking) group by (srch_destination_id, hotel_cluster),
#                                               hotel_cluster)]}
# dest_top_five2 = {srch_destination_id: [(sum_and_len(is_booking) group by (srch_destination_id, hotel_cluster),
#                                          hotel_cluster)]}
def computeNearestNeighbor(user, users_lst, k=5):
    # Get the test dataset fields
    user_city, dist, srch_dest = user[1]
    length = 0
    result = []
    if dist and dist in users_lst[0][0].keys():
        result = list(map(lambda x: x[1], users_lst[0][0][dist]))
        length += len(result)
    if length < k and srch_dest in users_lst[0][1].keys():
        tmp = list(map(lambda x: x[1], users_lst[0][1][srch_dest]))
        for clu in result:
            if clu in tmp:
                tmp.remove(clu)
        result += tmp
        length += len(result)
    if length < k:
        # setting to be top clusters
        addition = list(map(lambda x: x[0], users_lst[1]))
        # Remove repetitive clusters
        for clu in result:
            if clu in addition:
                addition.remove(clu)
        result += addition
        
    return result[:k]


# function: Test using test dataset
def test(train_data):
    # Output result file
    filename = 'submission_%s.csv' % time.strftime("%Y%m%d%H", time.localtime())

    with open('../input/test.csv') as fp:
        titles = fp.readline().strip().split(',')
        titles_index = {}
        for i, title in enumerate(titles):
            titles_index[title] = i
        ind_dist = titles_index['orig_destination_distance']
        ind_id = titles_index['id']
        ind_srch_dest = titles_index['srch_destination_id']
        ind_user_loc_city = titles_index['user_location_city']
        
        with open(filename, 'w') as output:
            output.write('id,hotel_cluster\n')
            for line in fp:
                fields = line.strip().split(',')
                if not fields:
                    break
                user_id = fields[ind_id]
                user_city = fields[ind_user_loc_city]
                srch_dest = fields[ind_srch_dest]
                dist = fields[ind_dist]
                user = (user_id, (user_city, dist, srch_dest))
                result = computeNearestNeighbor(user, train_data, 5)
                result = [str(i) for i in result]
                out_str = user_id + ',' + ' '.join(result) + '\n'
                output.write(out_str)


# function: main function                      
def main():
    print("Starting ... ")
    sta_time = time.time()
    # Pre-processing Train Data
    data = preprocess()
    preprocess_time = time.time()
    print("The pre-processing time is %s seconds." % (preprocess_time-sta_time))
    # Test Data
    test(data)
    print("The test time is %s seconds." % (time.time()-preprocess_time))
    print("Ending ... ")
    
if __name__ == '__main__':
    main()
