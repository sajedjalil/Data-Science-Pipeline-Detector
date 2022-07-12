import pandas as pd
from haversine import haversine
from copy import deepcopy

north_pole = (90,0)
weight_limit = 990.0
gifts = pd.read_csv("../input/gifts.csv").fillna(" ")
### FOR TESTING PURPOSES ONLY ###
#gifts = gifts
#weight_limit = 1000.0
### FOR TESTING PURPOSES ONLY ###
giftsNS = gifts.sort_values(by=['Latitude'], ascending=[1])
giftsWE = gifts.sort_values(by=['Longitude'], ascending=[1])
giftsWt = gifts.sort_values(by=['Weight'], ascending=[0])

trips = [[]]
totwt = 0

with open('submission.csv', 'w') as outfile:
    for i in range(len(giftsWE)):
        w = float(giftsWE[i:i+1]["Weight"])
        g_id = int(giftsWE[i:i+1]["GiftId"])
        if w <= weight_limit - totwt:
            trips[-1].append(g_id)
            totwt += w
        else:
            totwt = w
            trips.append([g_id])
    #giftsNS.set_index['GiftId']
    for i,t in enumerate(trips):
        #print(i)
        currgiftlist = gifts.loc[gifts.GiftId.isin(t)]
        u = currgiftlist.sort_values(by=['Latitude'], ascending=[1])
        deliv_order1 = []
        deliv_order2 = []
        top = True
        for j in range(len(u)):
            g_id = int(u[j:j+1]["GiftId"])
            if top:
                deliv_order1.append(g_id)
            else:
                deliv_order2.append(g_id)
        deliv_order2.reverse()
        actual_order = deliv_order1+deliv_order2
        for g in actual_order:
            outstring = str(g)+','+str(i+1)+'\n'
            outfile.write(outstring)
