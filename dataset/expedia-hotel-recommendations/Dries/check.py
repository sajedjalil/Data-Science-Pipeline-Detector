from collections import defaultdict
from heapq import nlargest
from operator import itemgetter
import datetime
import math

def evaluate_dictionary_combination(dic_tr_list, dic_val, key_indices_tr, w_list):
    """ Returns how good a weighted sum of dictionaries predicts the hotel clusters in the valiation set.
    The evaluation method goes as follows:
    1) dic_tr_list is a list of training dictionaries. The scores stored in these dictionaries are determined by the training data.
    The key in each training dictionary is a tuple representing a set of features (e.g., (search_destionation_id, hotel_market)).
    The value in each training dictionary is a dictionary that maps the hotel clusters on their score. Ideally, for a given set of 
    features, the hotel clusters that are the most likely to be chosen have the highest score.
    2) dic_val is the validation dictionary. For each combination of features, this dictionary stores which hotel cluster is booked by 
    which amount of users in the validation set. The key in the validation dictionary is a tuple representing a set of features. 
    The value is a dictionary that maps each hotel cluster on how many users booked it.
    3) key_indices_tr is a list of integer-lists. The first integer-list corresponds to the first training dictionary, the second 
    integer-list corresponds to the second training dictionary, .... The lists map the features of each training dictionary to the features 
    of the validation dictionary. For example, if a training dictionary has a key of the form (search_destionation_id, the hotel_country)
    and the validation dictionary has a key of the form (search_destionation_id, the hotel_country, user_id), then the integer-list
    corresponding to this dictionary should equal [0, 1] because search_destionation_id and the hotel_country are the first and second
    feature in the validation set respectively. Note that the validation dictionary features need to be the union of the training 
    dictionary features.
    4) w_list represents how much each training dictionary is weighted. The algorithm aggregates the scores as follows. For each key in the 
    validation set, it will fetch the hotel cluster scores corresponding to the matching keys in the training dictionaries. We then take a weighted sum of 
    these scores where the weights are determined by w_list. The resulting hotel cluster with the highest score becomes the first choice, the one
    with the second highest score the seconds choice, and so on... A match occurs when the key of the training dictionary is a subset of
    the key in the validation dicionary. In other words, if all the features represtended by the training dictionary key equal the corresponding
    features represented by the validation set key.
    """    

    # Correct represents the current score (5 if hotel cluster is first choice, 4 if it is second choice, ...)
    correct = 0
    
    # Max correct will represent the maximum score, which equals 5 times the number of data points.
    max_correct = 0
    
    # We loop over each key (which represents a set of features) and value (which is a dictionary mapping the hotel clusters on how many users booked them)
    for key, value in dic_val.items():

        # hotel_clusters is a list the will contain all the hotel clusters that have a non-zero score in the training dictionaries
        hotel_clusters = []
        
        # For each training dictionary we build the matching key. This key is a subset of the validation dictionary key. 
        # It only contains the features considered by the training dictionary.
        for i in range(len(dic_tr_list)):
            tr_key = ()
            for j in key_indices_tr[i]:
                tr_key += (key[j],)      
            hotel_clusters = list(set(dic_tr_list[i][tr_key].keys()) | set(hotel_clusters))
        
        # For all the hotel clusters we take the weighted sum of the training dictionary scores. 
        # scores maps each hotel cluster on their corresponding aggregated score.
        scores = defaultdict(int)
        for key2 in hotel_clusters:
            score = 0
            for i in range(len(dic_tr_list)):
                tr_key = ()
                for j in key_indices_tr[i]:
                    tr_key += (key[j],)
                score += dic_tr_list[i][tr_key][key2]*w_list[i]
            scores[key2] = score
        
        # topitems contains the five hotel clusters with the highest aggregated score.
        topitems = nlargest(5, scores.items(), key=itemgetter(1))
        
        # We increase correct by: 5 times the number of users that booked the hotel cluster with the highest score,
        # 4 times the number of users that booked the hotel cluster with the second highest score, and so on
        for i in range(len(topitems)):
            correct += dic_val[key][topitems[i][0]]*(5-i)
        
        # we increase max_correct by 5 times the number of users corresponding to this specific 
        # validation dictionary key
        for key2, value2 in value.items():
            max_correct += dic_val[key][key2]*5

    # The performance equals correct/max_correct    f
    return math.floor(correct/max_correct*10000)/10000

	
def run_solution():

    print('Reading file...')
    f = open("../input/train.csv", "r")
    f.readline()
     
    search_dest_training = defaultdict(lambda: defaultdict(int))
    orig_destination_distance_training = defaultdict(lambda: defaultdict(int))
    feature_i_training = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    hotel_cluster_training = defaultdict(lambda: defaultdict(int))
    
    feature_i_booking_validation = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    base_line_booking_validation = defaultdict(lambda: defaultdict(int))
    
    feature_names = [
        'date_time',
        'site_name',
        'posa_continent',
        'user_location_country',
        'user_location_region',
        'user_location_city',
        'orig_destination_distance',
        'user_id',
        'is_mobile',
        'is_package',
        'channel',
        'srch_ci',
        'srch_co',
        'srch_adults_cnt',
        'srch_children_cnt',
        'srch_rm_cnt',
        'srch_destination_id',
        'srch_destination_type_id',
        'hotel_continent',
        'hotel_country',
        'hotel_market'
    ]
    
    total = 0

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 1000000 == 0:
            print('Read {} lines...'.format(total))
            
        if total % 3000000 == 0:
            break
            #break

        if line == '':
            break

        arr = line.split(",")
        book_year = int(arr[0][:4])
        book_month = int(arr[0][5:7])
        is_booking = int(arr[18])
        hotel_cluster = arr[23]
        srch_destination_id = arr[16]
        orig_destination_distance = arr[6]
        
        irange = [1,2,3,4,5,7,8,9,10,13,14,15,17,18,19,20]
        
        validation = (book_year == 2014) and (book_month > 6)
        
        if validation == False:
        
            if srch_destination_id != '':
                search_dest_training[(srch_destination_id,)][hotel_cluster] += 1 + 5*is_booking

            if orig_destination_distance != '':
                orig_destination_distance_training[(orig_destination_distance,)][hotel_cluster] += 1 + 5*is_booking

            for i in irange:
                if arr[i] != '':
                    feature_i_training[i][(arr[i],)][hotel_cluster] += 1 + 5*is_booking
        
            hotel_cluster_training[()][hotel_cluster] += 1 + 5*is_booking

        elif validation == True and is_booking == 1:
                
            for i in irange:
                feature_i_booking_validation[i][(srch_destination_id, orig_destination_distance, arr[i])][hotel_cluster] += 1
		    
            base_line_booking_validation[(srch_destination_id, orig_destination_distance)][hotel_cluster] += 1


    print("")    
    print("DESCRIPTION SCRIPT")
    print("This scripts checks 17 basic hotel cluster selection algorithms.")
    print("Each algorithm corresponds to three features in the data set; 'srch_destination_id', 'orig_destination_distance' and another feature that changes per algorithm.")
    print("The algorithms work as follows.") 
    print("First, for each datapoint in the validation set it collects the training data points that share 'orig_destination_distance'.") 
    print("Second, the predicted hotel clusters for the validation data point becomes the five most popular hotel clusters among those training datapoints.")
    print("Third, if the training data points do not contain five different hotel clusters, then we fill the remaining spots by matching the 'srch_destination_id' instead.")
    print("Fourth, if some hotel cluster spots are still open, then we fill the remaining spots based on the algorithm-specific feature.")
    print("Fifth, the remaining spots are filled based on general hotel popularity.")
    print("")    
    print("RESULTS")
    print("(Map@5, Map@1) | feature_number | feature_name") 
    print("-----------------------------")  
    
    dic_list = [
        search_dest_training,
        orig_destination_distance_training,
        hotel_cluster_training,
    ]
    ind_list = [[0],[1],[]]
    w_list = [1000000000, 1000000000*1000000000, 1]
    p = evaluate_dictionary_combination(dic_list, base_line_booking_validation, ind_list, w_list)
    
    print( str(p) + ' | | No feature(base_line)')
        
    for i in irange:
    
        p = 0
        dic_list = [
            search_dest_training,
            orig_destination_distance_training,
            feature_i_training[i],
            hotel_cluster_training,
        ]
        ind_list = [[0],[1],[2],[]]
        w_list = [100000000*100000000, 10000000*100000000*100000000, 100000000, 1]
        p = evaluate_dictionary_combination(dic_list, feature_i_booking_validation[i], ind_list, w_list)
        
        print( str(p) + ' | ' + str(i) + ' | ' + feature_names[i])

    print("") 
    print("CONCLUSION")
    print("hotel_market is the feature with the highest performance.")
    print("")    
   
    print('Script ended sucessfully')

run_solution()