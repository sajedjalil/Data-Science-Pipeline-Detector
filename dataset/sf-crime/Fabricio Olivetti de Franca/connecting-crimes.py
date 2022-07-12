from numpy import radians, sin, cos, sqrt, arcsin, square
import pandas as pd
import datetime
import networkx as nx
import matplotlib.pyplot as plt
import zipfile

# Some constants
walkspeed = 1.38
runspeed = 12.5
avgcarspeed = 14

z = zipfile.ZipFile('../input/train.csv.zip')
train = pd.read_csv(z.open('train.csv'))

# Convert Dates to datetime and calculate the difference between adjacent crimes in seconds
train['datetime'] = train['Dates'].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
train['timediff'] = (train.datetime - train.datetime.shift(1)).abs().dt.seconds

# Create some variables to calculate the haversine distance
train['lat'] = train['X'].map(lambda x: cos(radians(x)))
train['dLat'] = square(sin(radians(train.X - train.X.shift(1))/2.))
train['dLon'] = square(sin(radians(train.Y - train.Y.shift(1))/2.))
train['latmult'] = train.lat*train.lat.shift(1)

# Calculate the haversine distance between adjacent crimes 
# and the speed the criminal should have in order to perform both crimes
train['dist'] = arcsin(sqrt(train['dLat'] + train['latmult']*train['dLon']))*12745000.6
train['speed'] = train['dist']/train['timediff']
train['prevcrime'] = train.Category.shift(1)

# A directed graph to infer the link between crimes
G = nx.DiGraph()

# if it is at the same spot and with no timediff, it is the same crime
for crime, prev in train[ (train['dist'] == 0) & (train['timediff'] == 0)][ ['Category','prevcrime'] ].values:
    if G.has_edge(prev, crime):
        G[prev][crime]['weight'] += 1
    else:
        G.add_edge( prev, crime, weight=1 )
        
# if the difference between two crimes are at walking speed but not too slow, it is connected
for crime, prev in train[(train['speed']>1.0) & (train['speed']<=walkspeed)][ ['Category','prevcrime'] ].values:
    if G.has_edge(prev, crime):
        G[prev][crime]['weight'] += 1
    else:
        G.add_edge( prev, crime, weight=1 )
        

# filter the edges with weights higher than a threshold
thr = 700 
Gc = nx.DiGraph()
for v1,v2 in G.edges():
    if G[v1][v2]['weight'] > thr:
        Gc.add_edge(v1,v2, weight=G[v1][v2]['weight'])

pos=nx.spring_layout(Gc)
edgewidth=[]
for (u,v,d) in Gc.edges(data=True):
    edgewidth.append(len(G.get_edge_data(u,v)))

# plot the graph
plt.figure(figsize=(16,14))
nodesize=[Gc.degree(v)*50 for v in Gc]
nx.draw_networkx_nodes(Gc,pos,node_size=nodesize,node_color='w',alpha=0.4)
nx.draw_networkx_edges(Gc,pos,alpha=0.4,node_size=0,width=1,edge_color='k')
nx.draw_networkx_labels(Gc,pos,fontsize=14)
plt.savefig('network.png')