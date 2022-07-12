

# Any results you write to the current directory are saved as output.

import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict


def run_solution():
    print('Preparing arrays...')
    f = open("../input/train.csv", "r")
    f.readline()
    best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest = defaultdict(lambda: defaultdict(int))
    
    ##
    best_hotels_search_dest_length = defaultdict(lambda: defaultdict(int))
    ##
    
    best_hotels_search_dest1 = defaultdict(lambda: defaultdict(int))
    best_hotel_country = defaultdict(lambda: defaultdict(int))
    popular_hotel_cluster = defaultdict(int)
    total = 0
    
    # Calc counts
    while 1:
        line = f.readline().strip()
        total += 1
    
        if total % 1000000 == 0:
            print('Read {} lines...'.format(total))
    
        if line == '':
            break
    
        arr = line.split(",")
        book_year = int(arr[0][:4])
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        srch_destination_id = arr[16]
        srch_destination_type_id=arr[17]
        is_booking = int(arr[18])
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = arr[23]
        
        
        ##
        co=arr[12]
        ci=arr[11]
        try:
            days = int(co[8:10])-int(ci[8:10])
            month = int(co[5:7])-int(ci[5:7])
            year=int(co[0:4])-int(ci[0:4])
            length=year*365+month*30+days
            length=int((length-length%7)/7)
            if length<0:
                length=-1
        except ValueError:
            length=-1
        ##
    
        is_package=arr[9]
    
        append_1 = 1 + 4*is_booking
        append_2 = 1 + 5*is_booking
    
        if user_location_city != '' and orig_destination_distance != '':
            hsh = (hash('user_location_city_'+str(user_location_city) + '_orig_destination_distance_' + str(orig_destination_distance)))
            best_hotels_od_ulc[hsh][hotel_cluster] += 1
            
            
            
        if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year == 2014:
            hsh = (hash('srch_destination_id_' +str(srch_destination_id) +"_hotel_country_" + str(hotel_country) +"_hotel_market_"+str(hotel_market)))
            best_hotels_search_dest[hsh][hotel_cluster] += append_1
            
            
            if srch_destination_type_id!=None:
                hsh = (hash('srch_destination_id_' +str(srch_destination_id) +"_hotel_country_" + str(hotel_country)
                        +"_hotel_market_"+str(hotel_market)+"srch_destination_type_id"+str(srch_destination_type_id)))
                best_hotels_search_dest_length[hsh][hotel_cluster]+=append_1
            
    
        if srch_destination_id != '':
            hsh = hash('srch_destination_id_'+str(srch_destination_id))
            best_hotels_search_dest1[hsh][hotel_cluster] += append_1
    
        if hotel_country != '':
            hsh = hash('hotel_country_'+str(hotel_country))
            best_hotel_country[hsh][hotel_cluster] += append_2
    
        popular_hotel_cluster[hotel_cluster] += 1
        
    f.close()

    print('Generate submission...')
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
    
        if total % 500000 == 0:
            print('Write {} lines...'.format(total))
    
        if line == '':
            break
    
        arr = line.split(",")
        id = arr[0]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        srch_destination_id = arr[17]
        srch_destination_type_id=arr[18]
        hotel_country = arr[20]
        hotel_market = arr[21]
        
        #
        ci=arr[12]
        co=arr[13]
        try:
            days = int(co[8:10])-int(ci[8:10])
            month = int(co[5:7])-int(ci[5:7])
            year=int(co[0:4])-int(ci[0:4])
            length=year*365+month*30+days
            length=int((length-length%7)/7)
            if length<0:
                length=-1
        except ValueError:
            length=-1
        
        is_package=arr[10]
        #
    
        out.write(str(id) + ',')
        filled = []
    
        hsh = (hash('user_location_city_'+str(user_location_city) +
                        '_orig_destination_distance_' +
                        str(orig_destination_distance)))
        
        if hsh in best_hotels_od_ulc:
            d = best_hotels_od_ulc[hsh]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
    
        hsh1 = (hash('srch_destination_id_' +
                         str(srch_destination_id) +
                         "_hotel_country_" + str(hotel_country) +
                         "_hotel_market_"+str(hotel_market)))
        hsh2 = hash('srch_destination_id_'+str(srch_destination_id))
        hsh20 = hsh = (hash('srch_destination_id_' +str(srch_destination_id) +"_hotel_country_" + str(hotel_country)
                        +"_hotel_market_"+str(hotel_market)+"srch_destination_type_id"+str(srch_destination_type_id)))
        if (len(filled) < 5) and (hsh1 in best_hotels_search_dest):
            if hsh20 in best_hotels_search_dest_length:
                d = best_hotels_search_dest_length[hsh20]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                         break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])
            
            d = best_hotels_search_dest[hsh1]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
        elif (len(filled) < 5) and (hsh2 in best_hotels_search_dest1):
            d = best_hotels_search_dest1[hsh2]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
    
        hsh = hash('hotel_country_'+str(hotel_country))
        if (len(filled) < 5) and (hsh in best_hotel_country):   
            d = best_hotel_country[hsh]
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
    print('Completed!')

run_solution()