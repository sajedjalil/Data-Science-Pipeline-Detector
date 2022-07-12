# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
def run():
    file=open('../input/train.csv','r')
    print(file.readline().split(','))
    hotel_serch=defaultdict(lambda : defaultdict(int))
    hotel_UL2OD=defaultdict(lambda : defaultdict(int))
    hotel_HM2SID=defaultdict(lambda : defaultdict(int))
    #hotel_click_to_booking = defaultdict(lambda: defaultdict(int))
    hotel_popular=defaultdict(int)
    i=0
    while 1:
        i=i+1
        line=file.readline()
        if(not i%10000000):
            print("No."+str(i))
        if(line==''):#
            break
        array=line.split(',')
        #print(array)
        date_time= array[0]
        user_location= array[5]
        srch_destination_id = array[16]
        srch_destination_deistance=array[6]
        hotel_market= array[22]
        is_booking = int(array[18])
        hotel_cluster = array[23].split('\n')[0]
        year=int(date_time[:4])
        month= int(date_time[5:7])
        time= 12*(year-2012)+(month-12)
        if(srch_destination_deistance!='' and user_location!=''):
            hotel_UL2OD[(user_location,srch_destination_deistance)][hotel_cluster] += time
        if(hotel_market!=''and srch_destination_id!=''):
            hotel_HM2SID[(hotel_market,srch_destination_id)][hotel_cluster]+= time
        if(srch_destination_id!='' ):
            hotel_serch[srch_destination_id ][hotel_cluster]+=is_booking+0.09

        hotel_popular[hotel_cluster]=1+hotel_popular[hotel_cluster]
        #print(hotel_serch[srch_destination_id])
    file.close()
    path = 'submission_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    file=open('../input/test.csv','r')
    print(file.readline().split(','))
    out.write('id,hotel_cluster\n')
    hotel_popular_most=nlargest(5,hotel_popular.items(),key=itemgetter(1))
    i=0
    while 1:
        i=i+1
        line=file.readline()
        if(line==''):
            break
        if(not i%1000000):
            print('No.'+str(i))
        array=line.split(',')
        srch_destination_id = array[17]
        user_location = array[6]
        srch_destination_deistance = array[7]
        hotel_market= array[21]

        mots_pop_UL2OD = nlargest(5, hotel_UL2OD[(user_location ,srch_destination_deistance)].items(), key=itemgetter(1))
        most_pop_destinations=nlargest(5, hotel_serch[srch_destination_id].items(), key=itemgetter(1))
        most_pop_HM2SID=nlargest(5,hotel_HM2SID[(hotel_market,srch_destination_id)].items(),key=itemgetter(1))
        rec=[]

        data=str(array[0])+','
        for j in range(len(mots_pop_UL2OD)):
            rec.append(mots_pop_UL2OD[j][0])
        for j in range(len(most_pop_HM2SID)):
            if(not most_pop_HM2SID[j][0]in rec and len(rec)<5):
                rec.append(most_pop_HM2SID[j][0])
        for j in range(len(most_pop_destinations)):
            if(not most_pop_destinations[j][0]in rec and len(rec)<5):
                rec.append(most_pop_destinations[j][0])
        for j in rec:
            data=data+str(j)+' '
        for j in hotel_popular_most:
            if len(rec)<5 and not j in rec:
                rec.append(j[0])
                data=data+str(j[0])+' '

        #print(data)
        out.write(data+'\n')
    out.close()
    file.close()
    print('finished!')



run()


