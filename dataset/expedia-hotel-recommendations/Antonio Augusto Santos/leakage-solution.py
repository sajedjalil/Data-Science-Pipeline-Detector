import datetime
import time
from heapq import nlargest
from operator import itemgetter
import re

# expressions used to parse the dates latter
re_datetime = re.compile(r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})')
re_date = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
def prepare_arrays_match():
    f = open("../input/train.csv", "r")
    f.readline()
    best_hotels_od_ulc = dict()
    best_hotels_stay_days = dict()
    best_hotels_days_to_travel = dict()
    best_hotels_search_dest = dict()
    popular_hotel_cluster = dict()
    total = 0

    # Calc counts
    start = time.time()
    
    while 1:
        line = f.readline().strip()
        total += 1

        if total % 1000000 == 0:
            spent = time.time() - start
            print('Read {} lines in {:.2f} secs...'.format(total, spent))
            break

        if line == '':
            break

        arr = line.split(",")
        user_location_city = arr[5]
        orig_destination_distance = (float(arr[6]) if arr[6] != '' else 0) // 10
        srch_destination_id = arr[16]
        is_booking = float(arr[18])
        hotel_cluster = arr[23]
        
        # Datetime parser per http://stackoverflow.com/a/14163523                
        # About 20x faster than 
        # date_time = strptime(arr[0], '%Y-%m-%d %H:%M:%S')
        # and 30x faster than
        # dateutil.parser.parse(arr[0])        
        
        try:
            date_time = datetime.datetime(*map(int, re_datetime.match(arr[0]).groups()))        
            check_in = datetime.datetime(*map(int, re_date.match(arr[11]).groups()))            
            days_to_travel = (check_in - date_time).days
            days_to_travel = days_to_travel // 3            
            try:
                check_out = datetime.datetime(*map(int, re_date.match(arr[12]).groups()))
                stay_days = (check_out - check_in).days
                stay_days = stay_days // 3
            except: 
                stay_days = -1            
        except:
            days_to_travel = -1
            stay_days = -1
                        
        if stay_days > 0:
            if stay_days in best_hotels_stay_days:
                if hotel_cluster in best_hotels_stay_days[stay_days]:
                   best_hotels_stay_days[stay_days][hotel_cluster] += 1
                else:
                   best_hotels_stay_days[stay_days][hotel_cluster] = 1
            else:
                best_hotels_stay_days[stay_days] = dict()
                best_hotels_stay_days[stay_days][hotel_cluster] = 1
                
        if days_to_travel > 0:
            if days_to_travel in best_hotels_days_to_travel:
                if hotel_cluster in best_hotels_days_to_travel[days_to_travel]:
                   best_hotels_days_to_travel[days_to_travel][hotel_cluster] += 1
                else:
                   best_hotels_days_to_travel[days_to_travel][hotel_cluster] = 1
            else:
                best_hotels_days_to_travel[days_to_travel] = dict()
                best_hotels_days_to_travel[days_to_travel][hotel_cluster] = 1                

        if user_location_city != '' and orig_destination_distance != '':
            s1 = (user_location_city, orig_destination_distance)

            if s1 in best_hotels_od_ulc:
                if hotel_cluster in best_hotels_od_ulc[s1]:
                    best_hotels_od_ulc[s1][hotel_cluster] += 1
                else:
                    best_hotels_od_ulc[s1][hotel_cluster] = 1
            else:
                best_hotels_od_ulc[s1] = dict()
                best_hotels_od_ulc[s1][hotel_cluster] = 1

        if srch_destination_id != '':
            if srch_destination_id in best_hotels_search_dest:
                if hotel_cluster in best_hotels_search_dest[srch_destination_id]:
                    best_hotels_search_dest[srch_destination_id][hotel_cluster] += is_booking*1 + (1-is_booking)*0.15
                else:
                    best_hotels_search_dest[srch_destination_id][hotel_cluster] = is_booking*1 + (1-is_booking)*0.15
            else:
                best_hotels_search_dest[srch_destination_id] = dict()
                best_hotels_search_dest[srch_destination_id][hotel_cluster] = is_booking*1 + (1-is_booking)*0.15

        if hotel_cluster in popular_hotel_cluster:
            popular_hotel_cluster[hotel_cluster] += 1
        else:
            popular_hotel_cluster[hotel_cluster] = 1

    f.close()
    return best_hotels_search_dest, best_hotels_od_ulc,  popular_hotel_cluster, best_hotels_stay_days, best_hotels_days_to_travel


def gen_submission(best_hotels_search_dest, best_hotels_od_ulc, popular_hotel_cluster, best_hotels_stay_days, best_hotels_days_to_travel):
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

        if total % 100000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        id = arr[0]
        user_location_city = arr[6]        
        orig_destination_distance = (float(arr[7]) if arr[7] != '' else 0) // 10
        srch_destination_id = arr[17]
                
        try:
            date_time = datetime.datetime(*map(int, re_datetime.match(arr[1]).groups()))
            check_in = datetime.datetime(*map(int, re_date.match(arr[12]).groups()))
            days_to_travel = (check_in - date_time).days
            days_to_travel = days_to_travel // 3            
            try:
                check_out = datetime.datetime(*map(int, re_date.match(arr[13]).groups()))
                stay_days = (check_out - check_in).days
                stay_days = stay_days // 3
            except: 
                stay_days = -1            
        except:
            days_to_travel = -1
            stay_days = -1

        out.write(str(id) + ',')
        filled = []

        s1 = (user_location_city, orig_destination_distance)
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

        if stay_days in best_hotels_stay_days:
            d = best_hotels_stay_days[stay_days]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                
        if days_to_travel in best_hotels_days_to_travel:
            d = best_hotels_days_to_travel[days_to_travel]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])        

        if srch_destination_id in best_hotels_search_dest:
            d = best_hotels_search_dest[srch_destination_id]
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

gen_submission(*prepare_arrays_match())