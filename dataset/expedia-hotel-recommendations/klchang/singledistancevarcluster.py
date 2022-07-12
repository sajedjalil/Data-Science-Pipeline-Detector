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
created time: 2016.5.27
updates:
2016.5.30
1. Add the is_booking field and its corresponding weight

"""
import time
import heapq


# function: Pre-process the train data
def preprocess(k=5):
    print("Pre-processing train dataset...")
    data = {}
    search_dest_dict = {}
    popular_hotel_clusters = {}
    total = 0
    with open("../input/train.csv") as fp:
        titles = fp.readline().strip().split(',')
        titles_index = {}
        for i, title in enumerate(titles):
            titles_index[title] = i
        # print(titles)
        ind_srch_dest_id = titles_index['srch_destination_id']
        ind_user_loc_city = titles_index['user_location_city']        
        ind_clus = titles_index['hotel_cluster']  
        ind_is_booking = titles_index['is_booking']
        
        for line in fp:
            fields = line.strip().split(',')
            if not fields:
                break
            total += 1
            if total % 1000000 == 0:
                print("The count that has been processed in train dataset is %s." % total)            
            # Extract single attribute - (user_city, search_dest)
            cluster = int(fields[ind_clus])
            if cluster not in popular_hotel_clusters:
                popular_hotel_clusters[cluster] = 1
            else:
                popular_hotel_clusters[cluster] += 1
            # Add the fields "srch_destination_id" and "user_location_city"
            search_dest = fields[ind_srch_dest_id]
            user_city = fields[ind_user_loc_city]
            # Add the field 'is_booking' and its weight
            booking_weight = 3
            if fields[ind_is_booking] == '1':
                is_booking = 1 * booking_weight
            else:
                is_booking = 0
            if search_dest and user_city:
                if not search_dest_dict.get((user_city, search_dest)):
                    search_dest_dict[(user_city, search_dest)] = {cluster: 1+is_booking}
                elif not search_dest_dict[(user_city, search_dest)].get(cluster):
                    search_dest_dict[(user_city, search_dest)][cluster] = 1+is_booking
                else:
                    search_dest_dict[(user_city, search_dest)][cluster] += 1+is_booking

    # Get top k hotel clusters
    top_hotel_clusters = heapq.nlargest(k, popular_hotel_clusters.items(), key=lambda x: x[1])
    # search_dest_dict - {(user_city, search_dest): {cluster1: n, cluster2: m, }, }
    for key, value in search_dest_dict.items():
        clusters_lst = sorted(value.items(), key=lambda x: x[1], reverse=True)
        data[key] = clusters_lst
    
    return [data, top_hotel_clusters]


# function: compute manhattan distance
# input: x1, x2 - list, int, float
def manhattan(x1, x2):
    dist = 0
    if isinstance(x1, list):
        for i in range(len(x1)):
            dist += abs(x1[i]-x2[i])
    else:
        dist = abs(x1-x2)
        
    return dist


# function: Compute the similarity between two users
def computeSimilarity(user1, user2):
    return manhattan(user1[2], user2[2])


# function: Compute the k Nearest Neighbor 
# users_lst = [data, top_hotel_clusters]
def computeNearestNeighbor(user, users_lst, k=5):
    key = user[1]
    if key in users_lst[0]:
        result = list(map(lambda x: x[0], users_lst[0][key]))
        length = len(result)
        if length < k:
            addition = list(map(lambda x:x[0], users_lst[1]))
            addition = addition[:k-length]
            result += addition
    else:
        # setting to be top clusters
        result = list(map(lambda x: x[0], users_lst[1]))

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
        ind_srch_dest = titles_index['srch_destination_id']
        ind_user_loc_city = titles_index['user_location_city']
        ind_id = titles_index['id']
        with open(filename, 'w') as output:
            output.write('id,hotel_cluster\n')
            for line in fp:
                fields = line.strip().split(',')
                if not fields:
                    break
                user_id = fields[ind_id]
                user_city = fields[ind_user_loc_city]
                srch_dest = fields[ind_srch_dest]
                user = (user_id, (user_city, srch_dest))
                result = computeNearestNeighbor(user, train_data, 5)
                output.write(user_id + ',')
                for clus in result:
                    output.write(str(clus) + ' ')
                output.write('\n')


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
