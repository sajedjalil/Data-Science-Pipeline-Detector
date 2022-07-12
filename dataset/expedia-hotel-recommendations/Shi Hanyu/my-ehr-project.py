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
import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict
import math

hotel_serch=defaultdict(lambda : defaultdict(int))
hotel_UL2OD=dict()#defaultdict(lambda : defaultdict(int))
hotel_HM2SID=dict()
hotel_mark= defaultdict(lambda : defaultdict(int))
hotel_Uid_Sid= defaultdict(lambda : defaultdict(int))
best_s00 = dict()
best_s01 = dict()
best_hotels_uid_miss = dict()
hotel_popular=defaultdict(int)

def run():
    file=open('../input/train.csv','r')
    print(file.readline().split(','))
    #hotel_click_to_booking = defaultdict(lambda: defaultdict(int))

    i=0
    while 1:
        i=i+1
        line=file.readline()
        if(not i%10000000):
            print("No."+str(i))
        if(line=='' ):#
            print('No.'+str(i))
            break
        array=line.split(',')
        #print(array)
        date_time= array[0]
        user_location_region = array[4]
        user_location= array[5]
        srch_destination_deistance=array[6]
        user_id = array[7]
        srch_ci = array[11]
        srch_destination_id = array[16]
        is_package = array[9]
        is_booking =  float(array[18])
        hotel_country = array[21]
        hotel_market= array[22]
        hotel_cluster = array[23].split('\n')[0]
        #year=int(date_time[:4])
        #month= int(date_time[5:7])
        if(srch_ci!=''):
            year=int(srch_ci[:4])
            month= int(srch_ci[5:7])
        else:
            year=int(date_time[:4])
            month= int(date_time[5:7])

        time= 12*(year-2012)+(month-12)
        if not (time>0 and time<=36):
            time=0
            continue
        varible_1 =  pow(math.log(time), 1.36) * (-0.1+0.95*pow(time, 1.47)) * (3.5*((year - 2012)/2) + 22.56*is_booking)
        varible_2 = 3 + 5.56*is_booking
        
        if(srch_destination_deistance!='' and user_location!=''):
            s_UL2OD=(user_location,srch_destination_deistance)
            if(s_UL2OD in hotel_UL2OD):
                if(hotel_cluster in hotel_UL2OD[s_UL2OD]):
                    hotel_UL2OD[s_UL2OD][hotel_cluster] += time
                else:
                    hotel_UL2OD[s_UL2OD][hotel_cluster] = time
            else:
                hotel_UL2OD[s_UL2OD]=dict()
                hotel_UL2OD[s_UL2OD][hotel_cluster] = time

        if user_location != '' and srch_destination_deistance != '' and user_id !='' and srch_destination_id != '' and user_location_region != '' and is_booking==1:
            s00 = (user_id, user_location, srch_destination_id, user_location_region, hotel_market)
            if s00 in best_s00:
                if hotel_cluster in best_s00[s00]:
                    best_s00[s00][hotel_cluster] += time
                else:
                    best_s00[s00][hotel_cluster] = time
            else:
                best_s00[s00] = dict()
                best_s00[s00][hotel_cluster] = time

        if user_location != '' and srch_destination_deistance != '' and user_id !='' and srch_destination_id != '' and is_booking==1:
            s01 = (user_id, srch_destination_id, user_location_region, hotel_market)
            if s01 in best_s01:
                if hotel_cluster in best_s01[s01]:
                    best_s01[s01][hotel_cluster] += time
                else:
                    best_s01[s01][hotel_cluster] = time
            else:
                best_s01[s01] = dict()
                best_s01[s01][hotel_cluster] = time


        if user_location != '' and srch_destination_deistance == '' and user_id !='' and srch_destination_id != '' and user_location_region != '' and is_booking==1:
            s0 = (user_id, user_location, srch_destination_id, user_location_region, hotel_market)
            if s0 in best_hotels_uid_miss:
                if hotel_cluster in best_hotels_uid_miss[s0]:
                    best_hotels_uid_miss[s0][hotel_cluster] += time
                else:
                    best_hotels_uid_miss[s0][hotel_cluster] = time
            else:
                best_hotels_uid_miss[s0] = dict()
                best_hotels_uid_miss[s0][hotel_cluster] = time

        if(hotel_market!=''and srch_destination_id!='' and hotel_country!='' ):#and year==2014
            s_h2SID= (hotel_country,hotel_market,srch_destination_id,is_package)
            if s_h2SID in hotel_HM2SID:
                if hotel_cluster in hotel_HM2SID[s_h2SID]:
                    hotel_HM2SID[(hotel_country,hotel_market,srch_destination_id,is_package)][hotel_cluster]+= varible_1
                else:
                    hotel_HM2SID[(hotel_country,hotel_market,srch_destination_id,is_package)][hotel_cluster]= varible_1
            else:
                hotel_HM2SID[(hotel_country,hotel_market,srch_destination_id,is_package)]=dict()
                hotel_HM2SID[(hotel_country,hotel_market,srch_destination_id,is_package)][hotel_cluster]= varible_1
                
        #if(srch_destination_id!='' ):
        #    hotel_serch[srch_destination_id ][hotel_cluster]+=varible_1
        #if(hotel_country!=''):#hotel_market
        #    hotel_mark[hotel_country][hotel_cluster]=varible_2
        if(hotel_market!=''):#hotel_market
            hotel_mark[hotel_market][hotel_cluster]+=varible_2
        hotel_popular[hotel_cluster]+= time
    #print(hotel_serch[srch_destination_id])
    file.close()
def sub():
    path = 'submission_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    file=open('../input/test.csv','r')
    print(file.readline().split(','))
    out.write('id,hotel_cluster\n')
    hotel_popular_most=nlargest(5,hotel_popular.items(),key=itemgetter(1))
    i=0
    a=0
    b=0
    c=0
    d=0
    e=0
    while 1:
        i=i+1
        line=file.readline()
        if(line=='' ):
            print('No.'+str(i))
            break
        if(not i%500000):
            print('No.'+str(i))
        array=line.split(',')
        srch_destination_id = array[17]
        user_location_region = array[5]
        user_location = array[6]
        srch_destination_deistance = array[7]
        user_id = array[8]
        is_package = array[10]
        hotel_country= array[20]
        hotel_market= array[21].split('\n')[0]
        mots_pop_UL2OD =[]
        most_pop_destinations=[]
        most_pop_HM2SID=[]
        most_Uid_Sid=[]
        most_pop_mark=[]
        most_unknow_1=[]
        most_unknow_2=[]
        data=str(array[0])+','
        
        if(user_location!=''and srch_destination_deistance!=''):
            if((user_location ,srch_destination_deistance) in hotel_UL2OD):
                mots_pop_UL2OD = nlargest(5, sorted(hotel_UL2OD[(user_location ,srch_destination_deistance)].items()), key=itemgetter(1))
            
        s00 = (user_id, user_location, srch_destination_id, user_location_region, hotel_market)
        s01 = (user_id, srch_destination_id, user_location_region, hotel_market)
        if(srch_destination_deistance == ''):
            s0 = (user_id, user_location, srch_destination_id, user_location_region, hotel_market)
            if s0 in best_hotels_uid_miss:
                dk = best_hotels_uid_miss[s0]
                most_unknow_1 = nlargest(4, sorted(dk.items()), key=itemgetter(1))

        if(s01 in best_s01 and s00 not in best_s00):
            dk = best_s01[s01]
            most_unknow_2 = nlargest(4, sorted(dk.items()), key=itemgetter(1))
        #if(srch_destination_id!=''):
        #    most_pop_destinations=nlargest(5, hotel_serch[srch_destination_id].items(), key=itemgetter(1))
        if(srch_destination_id!='' and hotel_market!='' and hotel_country!=''):
            if((hotel_country,hotel_market,srch_destination_id,is_package)in hotel_HM2SID ):
                most_pop_HM2SID=nlargest(5,sorted(hotel_HM2SID[(hotel_country,hotel_market,srch_destination_id,is_package)].items()),key=itemgetter(1))
        if(hotel_market!=''):
            most_pop_mark=nlargest(5,hotel_mark[hotel_market].items(),key=itemgetter(1))#hotel_country
        rec=[]

        for j in range(len(mots_pop_UL2OD)):
            #if(not mots_pop_UL2OD[j][0] in rec and len(rec)<5):
                rec.append(mots_pop_UL2OD[j][0])
                a+=1
        for j in range(len(most_unknow_1)):
            #if(not mots_pop_UL2OD[j][0] in rec and len(rec)<5):
            if(not most_unknow_1[j][0]in rec and len(rec)<5):
                rec.append(most_unknow_1[j][0])
                b+=1
        for j in range(len(most_unknow_2)):
            #if(not mots_pop_UL2OD[j][0] in rec and len(rec)<5):
            if(not most_unknow_2[j][0]in rec and len(rec)<5):
                rec.append(most_unknow_2[j][0])
                c+=1
        #for j in range(len(most_Uid_Sid)):
        #    if(not most_Uid_Sid[j][0]in rec and len(rec)<5):
        #        rec.append(most_Uid_Sid[j][0])
        #        b+=1
        #if len(most_pop_HM2SID)>0 :
        for j in range(len(most_pop_HM2SID)):
            if(not most_pop_HM2SID[j][0]in rec and len(rec)<5):
                rec.append(most_pop_HM2SID[j][0])
                d+=1
        #else:
        #for j in range(len(most_pop_destinations)):
        #    if(not most_pop_destinations[j][0]in rec and len(rec)<5):
        #        rec.append(most_pop_destinations[j][0])
        #        d+=1
        for j in range(len(most_pop_mark)):
            if(not most_pop_mark[j][0]in rec and len(rec)<5):
                rec.append(most_pop_mark[j][0])
                e+=1
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
    print(str(a))
    print(str(b))
    print(str(c))
    print(str(d))
    print(str(e))
    print('finished!')
    return 0




run()
sub()
