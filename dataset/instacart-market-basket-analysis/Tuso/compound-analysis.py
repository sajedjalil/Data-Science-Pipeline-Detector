__author__ = 'jianchao.jjc'
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import sys
#Support&Confidence
from itertools import combinations
# Any results you write to the current directory are saved as output.
df_od=pd.read_csv("../input/order_products__prior.csv")
df_od_t=pd.read_csv("../input/order_products__train.csv")
df_order=pd.read_csv("../input/orders.csv")
df_prd=pd.read_csv("../input/products.csv")
o_d=dict()
o_d_t=dict()
fast_map=dict()
p_d=dict()
o_d_f=dict()
for item in df_order.values:
    user_id=item[1]
    order_id=item[0]
    eval_set=item[2]
    order_num=item[3]
    if not eval_set in ["prior"]:
        continue
    if not user_id in o_d:
        o_d[user_id]=dict()
    fast_map[order_id]=user_id
    o_d[user_id][order_id]=dict()
    o_d[user_id][order_id]["lst"]=list()
    o_d[user_id][order_id]["order_num"]=order_num
    o_d[user_id][order_id]["info"]=item
    #if user_id==185648:
    #    print(item)
for item in df_order.values:
    user_id=item[1]
    order_id=item[0]
    eval_set=item[2]
    order_num=item[3]
    if not eval_set in ["train"]:
        continue
    if not user_id in o_d_t:
        o_d_t[user_id]=dict()
    fast_map[order_id]=user_id
    o_d_t[user_id][order_id]=dict()
    o_d_t[user_id][order_id]["lst"]=list()
    o_d_t[user_id][order_id]["order_num"]=order_num
    o_d_t[user_id][order_id]["info"]=item
for item in df_order.values:
    user_id=item[1]
    order_id=item[0]
    eval_set=item[2]
    order_num=item[3]
    if not eval_set in ["test"]:
        continue
    if not user_id in o_d_f:
        o_d_f[user_id]=dict()
    fast_map[order_id]=user_id
    o_d_f[user_id][order_id]=dict()
    o_d_f[user_id][order_id]["lst"]=list()
    o_d_f[user_id][order_id]["order_num"]=order_num
    o_d_f[user_id][order_id]["info"]=item
for item in df_od.values:
    order_id=item[0]
    product_id=item[1]
    order_num=item[2]
    reorder=item[3]
    user_id=fast_map.get(order_id,0)
    if not user_id in o_d:
        continue
    if not order_id in o_d[user_id]:
        continue
    o_d[user_id][order_id]["lst"].append(product_id)
#print(o_d[194845])
#raise
for item in df_od_t.values:
    order_id=item[0]
    product_id=item[1]
    order_num=item[2]
    reorder=item[3]
    user_id=fast_map.get(order_id,0)
    if not user_id in o_d_t:
        continue
    if not order_id in o_d_t[user_id]:
        continue
    o_d_t[user_id][order_id]["lst"].append(product_id)


for item in df_prd.values:
    prd_id=item[0]
    prd_name=item[1]
    a_id=item[2]
    d_id=item[3]
    p_d[prd_id]=[prd_name,a_id,d_id]
    
u_cnt=0
all_d=dict()
for user_id in o_d:
    u_cnt+=1
    if u_cnt==1001:
        break
    order_list=list(o_d[user_id].items())
    order_list.sort(key=lambda x:x[1]["order_num"])
    d=dict()
    last_order=[]
    for od in order_list:
        if not last_order:
            last_order=od[1]["lst"]
            continue
        for l_prd in last_order:
            for prd in od[1]["lst"]:
                l_prd_k=str(l_prd)
                prd_k=str(prd)
                if l_prd_k==prd_k:
                    continue
                cpd_prd_k=l_prd_k+"_"+prd_k
                all_d[cpd_prd_k]=all_d.get(cpd_prd_k,0)+1
all_cpd=list(all_d.items())
all_cpd.sort(key=lambda x:x[1],reverse=True)
print(all_cpd[:100])
#sus_lst=[item  for item in all_cpd if float(item[1])/float(u_cnt)>0.25]
sus_lst=all_cpd
new_dict=dict()
for item in sus_lst:
    key=item[0]
    num=item[1]
    key_spl=key.split("_")
    pre_key=int(key_spl[0])
    after_key=int(key_spl[1])
    if not after_key in new_dict:
        new_dict[after_key]=dict()
    if not pre_key in new_dict[after_key]:
        new_dict[after_key][pre_key]=float(item[1])/float(u_cnt)
for prd in p_d.keys():
    result=[0]
    print(prd)
    for user_id in o_d_t:
        for order_id in o_d_t[user_id]:
            if prd in o_d_t[user_id][order_id]["lst"]:
                result=[1]
            else:
                result=[0]
        info=o_d_t[user_id][order_id]["info"]
        dow=info[4]
        hod=info[5]
        lad=info[6]
        if int(hod)<=12 and int(hod)>=5:
            hod=0
        elif int(hod)<18:
            hod=1
        else:
            hod=2
        order_num=o_d_t[user_id][order_id]["order_num"]
        all_cnt=0
        buy_cnt=0
        all_cnt_d=0
        buy_cnt_d=0
        all_cnt_h=0
        buy_cnt_h=0
        all_cnt_p=0
        buy_cnt_p=0
        is_last=[0,0,0]
        pre_lst=[]
        for order_id in o_d[user_id]:
            info=o_d[user_id][order_id]["info"]
            p_dow=info[4]
            p_hod=info[5]
            p_lad=info[6]
            if int(p_hod)<=12 and int(p_hod)>=5:
                p_hod=0
            elif int(p_hod)<18:
                p_hod=1
            else:
                p_hod=2
            if p_dow==dow:
                all_cnt_d+=1
                if prd in o_d[user_id][order_id]["lst"]:
                    buy_cnt_d+=1
            if p_hod==hod:
                all_cnt_h+=1
                if prd in o_d[user_id][order_id]["lst"]:
                    buy_cnt_h+=1
            if p_lad and p_lad>5 and abs(p_lad-lad)<5:
                all_cnt_p+=1
                if prd in o_d[user_id][order_id]["lst"]:
                    buy_cnt_p+=1
            all_cnt+=1
            if prd in o_d[user_id][order_id]["lst"]:
                buy_cnt+=1
            o_num=o_d[user_id][order_id]['order_num']
            if o_num==order_num-1 and prd in o_d[user_id][order_id]["lst"]:
                is_last[0]=1
                pre_lst=o_d[user_id][order_id]["lst"]
            if o_num==order_num-2 and prd in o_d[user_id][order_id]["lst"]:
                is_last[1]=1
            if o_num==order_num-3 and prd in o_d[user_id][order_id]["lst"]:
                is_last[2]=1
        result.append(float(buy_cnt)/float(all_cnt))
        if all_cnt_d>0.1:
            result.append(float(buy_cnt_d)/float(all_cnt_d))
        else:
            result.append(0.0)
        if all_cnt_h>0.1:
            result.append(float(buy_cnt_h)/float(all_cnt_h))
        else:
            result.append(0.0)
        if all_cnt_p>0.1:
            result.append(float(buy_cnt_p)/float(all_cnt_p))
        else:
            result.append(0.0)
        result+=is_last
        spt=0.0
        if prd in new_dict:
            for pre_key in new_dict[prd]:
                if pre_key in pre_lst and new_dict[prd][pre_key]>spt:
                    spt=new_dict[prd][pre_key]
        result.append(spt)
        print(result)
        break
for prd in p_d.keys():
    result=[2]
    for user_id in o_d_f:
        for order_id in o_d_f[user_id]:
            if prd in o_d_f[user_id][order_id]["lst"]:
                result=[2]
        info=o_d_f[user_id][order_id]["info"]
        dow=info[4]
        hod=info[5]
        lad=info[6]
        if int(hod)<=12 and int(hod)>=5:
            hod=0
        elif int(hod)<18:
            hod=1
        else:
            hod=2
        order_num=o_d_f[user_id][order_id]["order_num"]
        all_cnt=0
        buy_cnt=0
        all_cnt_d=0
        buy_cnt_d=0
        all_cnt_h=0
        buy_cnt_h=0
        all_cnt_p=0
        buy_cnt_p=0
        is_last=[0,0,0]
        pre_lst=[]
        for order_id in o_d[user_id]:
            info=o_d[user_id][order_id]["info"]
            p_dow=info[4]
            p_hod=info[5]
            p_lad=info[6]
            if int(p_hod)<=12 and int(p_hod)>=5:
                p_hod=0
            elif int(p_hod)<18:
                p_hod=1
            else:
                p_hod=2
            if p_dow==dow:
                all_cnt_d+=1
            if prd in o_d[user_id][order_id]["lst"]:
                buy_cnt_d+=1
            if p_hod==hod:
                all_cnt_h+=1
                if prd in o_d[user_id][order_id]["lst"]:
                    buy_cnt_h+=1
            if p_lad and p_lad>5 and abs(p_lad-lad)<5:
                all_cnt_p+=1
                if prd in o_d[user_id][order_id]["lst"]:
                    buy_cnt_p+=1
            all_cnt+=1
            if prd in o_d[user_id][order_id]["lst"]:
                buy_cnt+=1
            o_num=o_d[user_id][order_id]['order_num']
            if o_num==order_num-1 and prd in o_d[user_id][order_id]["lst"]:
                is_last[0]=1
                pre_lst=o_d[user_id][order_id]["lst"]
            if o_num==order_num-2 and prd in o_d[user_id][order_id]["lst"]:
                is_last[1]=1
            if o_num==order_num-3 and prd in o_d[user_id][order_id]["lst"]:
                is_last[2]=1
        result.append(float(buy_cnt)/float(all_cnt))
        if all_cnt_d>0.1:
            result.append(float(buy_cnt_d)/float(all_cnt_d))
        else:
            result.append(0.0)
        if all_cnt_h>0.1:
            result.append(float(buy_cnt_h)/float(all_cnt_h))
        else:
            result.append(0.0)
        if all_cnt_p>0.1:
            result.append(float(buy_cnt_p)/float(all_cnt_p))
        else:
            result.append(0.0)
        result+=is_last
        spt=0.0
        if prd in new_dict:
            for pre_key in new_dict[prd]:
                if pre_key in pre_lst and new_dict[prd][pre_key]>spt:
                    spt=new_dict[prd][pre_key]
        result.append(spt)
        print(result)

        break
sys.exit(0)

