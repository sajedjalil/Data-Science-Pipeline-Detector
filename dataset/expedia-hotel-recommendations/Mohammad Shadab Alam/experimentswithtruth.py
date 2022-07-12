# coding: utf-8
#################################
#      Simple Validation        #
#################################
# contributors:
# ZFTurbo - idea, main part
# Kagglers - tuning, development
# Grzegorz Sionkowski - simple validation


import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict
import collections
from collections import Counter

# validation ###############
validate = 0  # 1 - validation, 0 - submission
N0 = 10       # total number of parts
N1 = 2        # number of part
#--------------------------

def findTop5(values):
    weight=[100000,50000,20000,10900,1700,1601,1490,1100,670,560,450,440,430,420,410,400,390,380,370,360,350,340,330,320,310]
    counts = collections.defaultdict(int)
    flag=0
    filled=[]
    
    for v in values:
        counts[v] += weight[flag]    
        flag=flag+1
    top_words = Counter(counts).most_common(5)
    #print top_words
    for word, frequency in top_words:
        #print("%s %d" % (word, frequency))
        filled.append(word)
    return filled

def run_solution():
    print('Preparing arrays...')
    f = open("../input/train.csv", "r")
    f.readline()
    best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest1 = defaultdict(lambda: defaultdict(int))
    best_hotel_country = defaultdict(lambda: defaultdict(int))
    hits = defaultdict(int)
    tp = defaultdict(float)

    popular_hotel_cluster = defaultdict(int)
    total = 0

    # Calc counts
    while 1:
        line = f.readline().strip()
        total += 1
        if total % 10000000 == 0:
            print('Read {} lines...'.format(total))
        if line == '':
            break
        arr = line.split(",")
        book_year = int(arr[0][:4])
        book_month = int(arr[0][5:7])
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        user_id = int(arr[7])
        is_package = int(arr[9])
        srch_destination_id = arr[16]
        is_booking = int(arr[18])
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = arr[23]

        if validate == 1 and user_id % N0 == N1:
            continue

        append_0 = ((book_year - 2012)*12 + (book_month - 12)) * ((book_year - 2012)*12 + (book_month - 12))
        append_1 = append_0 *  (3 + 16*is_booking)
        append_2 = 3 + 5.1*is_booking

        if user_location_city != '' and orig_destination_distance != '':
            best_hotels_od_ulc[(user_location_city, orig_destination_distance)][hotel_cluster] += append_0

        if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year != '':
            best_hotels_search_dest[(srch_destination_id, hotel_country, hotel_market,is_package)][hotel_cluster] += append_1

        if srch_destination_id != '':
            best_hotels_search_dest1[srch_destination_id][hotel_cluster] += append_1

        if hotel_country != '':
            best_hotel_country[hotel_country][hotel_cluster] += append_2

        popular_hotel_cluster[hotel_cluster] += 1

    f.close()
    ###########################
    if validate == 1:
        print('Validation...')
        f = open("../input/train.csv", "r")
    else:
        print('Generate submission...')
        f = open("../input/test.csv", "r")
    now = datetime.datetime.now()
    path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    f.readline()
    total = 0
    totalv = 0
    out.write("id,hotel_cluster\n")
    topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    while 1:
        line = f.readline().strip()
        total += 1
        if total % 1000000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        if validate == 1:
            book_year = int(arr[0][:4])
            book_month = int(arr[0][5:7])
            user_location_city = arr[5]
            orig_destination_distance = arr[6]
            user_id = int(arr[7])
            is_package = int(arr[9])
            srch_destination_id = arr[16]
            is_booking = int(arr[18])
            hotel_country = arr[21]
            hotel_market = arr[22]
            hotel_cluster = arr[23]
            id = 0
            if user_id % N0 != N1:
               continue
            if is_booking == 0:
               continue
        else:
            id = arr[0]
            user_location_city = arr[6]
            orig_destination_distance = arr[7]
            user_id = int(arr[8])
            is_package = int(arr[10])
            srch_destination_id = arr[17]
            hotel_country = arr[20]
            hotel_market = arr[21]
            is_booking = 1

        totalv += 1
        out.write(str(id) + ',')
        filled = []

        s1 = (user_location_city, orig_destination_distance)
        if s1 in best_hotels_od_ulc:
            d = best_hotels_od_ulc[s1]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if len(filled) ==25:
                    break
                if topitems[i][0] == '':
                    continue
                #out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                if validate == 1:
                   if topitems[i][0]==hotel_cluster:
                      hits[len(filled)] +=1

        s2 = (srch_destination_id, hotel_country, hotel_market,is_package)
        if s2 in best_hotels_search_dest:
            d = best_hotels_search_dest[s2]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if len(filled) == 25:
                    break
                if topitems[i][0] == '':
                    continue
                #out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                if validate == 1:
                   if topitems[i][0]==hotel_cluster:
                      hits[len(filled)] +=1
        elif srch_destination_id in best_hotels_search_dest1:
            d = best_hotels_search_dest1[srch_destination_id]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if len(filled) == 25:
                    break
                if topitems[i][0] == '':
                    continue
                #out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                if validate == 1:
                   if topitems[i][0]==hotel_cluster:
                      hits[len(filled)] +=1


        if hotel_country in best_hotel_country:
            d = best_hotel_country[hotel_country]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if len(filled) == 25:
                    break
                if topitems[i][0] == '':
                    continue
                #out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                if validate == 1:
                   if topitems[i][0]==hotel_cluster:
                      hits[len(filled)] +=1


        for i in range(len(topclasters)):
            if len(filled) == 25:
                    break
            if topclasters[i][0] == '':
                continue
            #out.write(' ' + topclasters[i][0])
            filled.append(topclasters[i][0])
            if validate == 1:
                if topclasters[i][0]==hotel_cluster:
                    hits[len(filled)] +=1
        filled=findTop5(filled)
        for i in filled:
            out.write(' ' + i)
                


        out.write("\n")
    out.close()
    print('Completed!')
    # validation >>>
    scores = 0.0
    classified = 0
    if validate == 1:
        for jj in range(1,6):
           scores +=  hits[jj]*1.0/jj
           tp[jj] = hits[jj]*100.0/totalv
           classified += hits[jj]
        misclassified = totalv-classified
        miscp = misclassified*100.0/totalv
        print("")
        print(" validation")
        print("----------------------------------------------------------------")
        print("position %8d %8d %8d %8d %8d %8d+" % (1,2,3,4,5,6))
        print("hits     %8d %8d %8d %8d %8d %8d " % (hits[1],hits[2],hits[3],hits[4],hits[5],misclassified))
        print("hits[%%]  %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f " % (tp[1],tp[2],tp[3],tp[4],tp[5],miscp))
        print("----------------------------------------------------------------")
        print("MAP@5 = %8.4f " % (scores*1.0/totalv))
    # <<< validation

run_solution()