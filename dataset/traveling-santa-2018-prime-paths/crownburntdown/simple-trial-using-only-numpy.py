
'''Notes:

this is just a first test to get the basic distance and iteration logic coded up.

Some TBDs:
1) implement prime + 10% logic.
2) Make forward-looking logic. i.e., dont just minimize the next step, minimize the next x steps. 
3) knowing you have to return to the north pole, need to factor in a minimization of the last city to NP as well..
4) consider how to turn this into a machine learning solution? what are the observations, attributes/predictors, and targets??
5) consider topological solutions (graph theory, etc..)
6) is this a candidate for clustering solns ?

'''

import numpy as np
from scipy.spatial import distance

#load coordinates
C = np.genfromtxt('../input/cities.csv',
           delimiter=',',
           skip_header=1,
           )

#biggest distances, measure 1% as an iterative search bounding threshold
perc = 100
def maxer(perc):
    x_max = max(C[:,1]) // perc + 1
    y_max = max(C[:,2]) // perc + 1
    return x_max,y_max
x_max,y_max = maxer(perc)

#define distance between two points
def dist(x) :
    return distance.euclidean( curr, x[1:] ) 

#start at north pole where i = 0
i = 0
path = np.array([[0]])
eucl = np.array([[0]])


#iterate through each stop
while len(path) < C.shape[0] :
    
    #store current location
    curr = C[i][1:]
    
    #ID all locations ~1% away (based on max coords)
    onep = C[np.logical_and.reduce((C[:,1] > C[i,1] - x_max,C[:,1] < C[i,1] + x_max,C[:,2] > C[i,2] - y_max,C[:,2] < C[i,2] + y_max,np.isin(C[:,0],path) == False))]
    
    #in case you dont find anything, look in bigger area
    if onep.shape[0] == 0 :
        perc = perc - 3
        x_max,y_max = maxer(perc)
        onep = C[np.logical_and.reduce((C[:,1] > C[i,1] - x_max,C[:,1] < C[i,1] + x_max,C[:,2] > C[i,2] - y_max,C[:,2] < C[i,2] + y_max,np.isin(C[:,0],path) == False))]
        continue
    
    #reset perc
    perc = 100   
    
    #get their eucl distance
    dista = np.apply_along_axis(dist, 1, onep)

    #find the closest one. if multiple are equally close, arbitrarily choose the smaller city ID.
    d = min(dista)
    i = onep[dista == min(dista)][0][0].astype(int)

    path = np.append(path,[[i]],axis=0)
    eucl = np.append(eucl,[[d]],axis=0)
    

    
#return to north pole 
path = np.append(path,[[0]],axis=0)
eucl = np.append(eucl,[[distance.euclidean(C[i][1:], C[0][1:] )]],axis=0)

#submit results
np.savetxt('path.csv', path, delimiter=',', fmt='%i', header='path', comments='')
