import datetime
import pandas as pd
import collections
import random
import math

import collections
uidc = collections.defaultdict(list)

with open('../input/train.csv') as trainfile:

    i=0
    for line in trainfile:
        if i==0:
            i+=1
            continue
        i+=1
        line = line.strip().split(',')
        hotel_cluster = line[23]

        if line[7]:
            user_id = line[7]
        else: 
            print('No user ID for line: {}'.format(i))
            continue
        
        uidc[user_id].append(hotel_cluster)
    
       
        if i%2000000==0:
            print('Line {} done!'.format(i))
            #break
            
i = 0
now = datetime.datetime.now()
path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'

with open('../input/test.csv') as infile, open(path, 'w') as outfile:
    outfile.write("id,hotel_cluster\n")
    for line in infile:
        cnt = collections.Counter()
        if i==0:
            i+=1  
            continue
        i+=1
        if i % 200000 == 0:
            print('Wrote {} lines...'.format(i))
            #break
            
        line = line.strip()
        arr = line.split(",")
        id = arr[0]
        #site_name = arr[2]
        user_location_city = arr[6]
        #orig_destination_distance = arr[7]
        user_id = arr[8]
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]
        
        if user_id in uidc:
            for hc in uidc[user_id]:
                cnt[hc]+=1
            besthc = cnt.most_common(5)
        else: 
            besthc = []
              
        outfile.write(str(id) + ',')
        outfile.write(' '.join(x[0] for x in besthc))
        outfile.write("\n")
