import csv
import pickle

def get_top5(count):
    freq={}
    for dest in count:
        hotel_count = count[dest]
        r_count={}
        for hotel,cc in hotel_count.items():
            r_count[cc]=hotel
        tmp=[r_count[i] for i in sorted(r_count.keys(),reverse=True)]
        tmp=tmp[:min(len(tmp),5)]
        freq[dest]=' '.join(tmp)
    return freq

name='../input/train.csv'
count={}
fo=open('pred_sub.csv','w')
fo.write('id,hotel_cluster\n')
fo.close()
for c,row in enumerate(csv.DictReader(open(name))):
    if True:
        xx=row['srch_destination_id']
        if xx not in count:
            count[xx]={}
        if row['hotel_cluster'] not in count[xx]:
            count[xx][row['hotel_cluster']]=0
        count[xx][row['hotel_cluster']]+=1
        #if c%100000==0:
        #    print c,len(count)
frequent=get_top5(count)
name='../input/test.csv'
for c,row in enumerate(csv.DictReader(open(name))):
    if True:
        fo=open('pred_sub.csv','a')
        if row['srch_destination_id'] not in frequent:
            tmp=''
        else:
            tmp=frequent[row['srch_destination_id']]
        fo.write('%d,%s\n'%(c,tmp))
        fo.close()
