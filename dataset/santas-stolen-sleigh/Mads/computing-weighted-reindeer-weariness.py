import numpy as np

gift_list=np.genfromtxt('../input/gifts.csv', delimiter=',')
gift_list[0]=[0,90,0.0,0.0]
wa_ = gift_list[::,3]
lo_ = gift_list[::,2]
la_ = gift_list[::,1]

def haversine_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def wrw_from_routes2(routes):
    leg_coord=[[],[],[],[]]
    leg_weight=[]
    for trip in routes:
        weight=wa_[trip].sum()
        weight+=10.0
        trip= [0]+trip+[0]
        leg_coord[0]+=list(la_[trip][:-1])
        leg_coord[1]+=list(lo_[trip][:-1])
        leg_coord[2]+=list(la_[trip][1:])
        leg_coord[3]+=list(lo_[trip][1:])
        for x in range(1,len(trip)):
            leg_weight.append(weight)
            weight-=wa_[trip[x]]
    leg_dist=haversine_np(leg_coord[0],leg_coord[1],leg_coord[2],leg_coord[3])
    leg_weight=np.array(leg_weight)
    leg_wrw=leg_dist*leg_weight
    return np.sum(leg_wrw)
    
rr=np.genfromtxt('../input/sample_submission.csv', delimiter=',')[1:]
rr=rr.astype(int)
a={}
for x,y in rr:
    a[y]=[]
for x,y in rr:
    a[y].append(x)

print (wrw_from_routes2(a.values()))