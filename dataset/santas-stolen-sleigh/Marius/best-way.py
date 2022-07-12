import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import *
import threading
import time
import networkx as nx
from sklearn.neighbors import NearestNeighbors

gifts=pd.read_csv("../input/gifts.csv")
lat=gifts['Latitude']
lon=gifts['Longitude']
wei=gifts['Weight']

length=100000 #数组长度
gifts['Is_North']=0
gifts['Trip_ID']=0

gifts=gifts.sort_values(by='Latitude')
start=[90,0]

#离北极点最远的1500个位置为终点
north_limited=1500
for i in np.arange(0,north_limited,1):
    gifts.iat[i,4]=1
#print(gifts.dtypes)    
#print(gifts[gifts['Latitude']<-85].count()['Latitude'])    

#print(gifts[gifts['Latitude']<-85])    
#gifts.to_csv('gifts_cleaned.csv')

#创建加权有向图
'''
G = nx.DiGraph()
D =[] #有向图坐标
for i in np.arange(0,length,1):
    s=(gifts['Latitude'][i],gifts['Longitude'][i],gifts['Weight'][i])
    D.append(s)

G.add_weighted_edges_from(D)
print(G.get_edge_data(16.3457688674,88.99948222))
'''

#计算距离
def calcDistance(Lat_A, Lng_A, Lat_B, Lng_B):
    lon1, lat1, lon2, lat2 = map(radians, [Lat_A, Lng_A, Lat_B, Lng_B])  
    # haversine公式  
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * asin(sqrt(a))   
    r = 6371 # 地球平均半径，单位为公里  
    return c * r * 1000  
    
    '''
     ra = 6378.140  # 赤道半径 (km)
     rb = 6356.755  # 极半径 (km)
     flatten = (ra - rb) / ra  # 地球扁率
     rad_lat_A = radians(Lat_A)
     rad_lng_A = radians(Lng_A)
     rad_lat_B = radians(Lat_B)
     rad_lng_B = radians(Lng_B)
     pA = atan(rb / ra * tan(rad_lat_A))
     pB = atan(rb / ra * tan(rad_lat_B))
     xx = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B))
     c1 = (sin(xx) - xx) * (sin(pA) + sin(pB)) ** 2 / cos(xx / 2) ** 2
     c2 = (sin(xx) + xx) * (sin(pA) - sin(pB)) ** 2 / sin(xx / 2) ** 2
     dr = flatten / 8 * (c1 - c2)
     distance = ra * (xx + dr)
     return distance
    '''
    
#print(calcDistance(lat[1],lon[1],lat[0],lon[0]))

#area=np.pi*wei**2
#colors=np.random.rand(lat.shape[0])

#plt.scatter(lat, lon, s=2, c=colors, alpha=0.5)
#plt.savefig('plot1.png', format='png')


#distance=np.zeros(shape=(length,length))
#length_range=np.arange(0,length,1)
#print(length_range)

#计算距离矩阵
'''
def cal_csv():
    for i in length_range:
        length_range_2=np.arange(i,length-i,1)
        for j in length_range_2:
            distance[i][j]=calcDistance(lat[i],lon[i],lat[j],lon[j])
        print('完成：%s' %(i/1000))
    return distance
'''

use=gifts.iloc[:,1:3]
nbrs = NearestNeighbors(n_neighbors=1500, algorithm='ball_tree').fit(use)
x=nbrs.radius_neighbors_graph()
print(x.toarray())    