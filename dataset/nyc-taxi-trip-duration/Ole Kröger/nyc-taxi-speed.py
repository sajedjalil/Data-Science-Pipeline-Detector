"""
As this is a script because I have absolutely no idea about jupyter notebooks I still would like to 
explain my script... (A readme would be a nice option @kaggle ;) )

Whatever:
This script visualizes trips in NYC. I used the DIMACS dataset for generating the shortest paths between the start and the end point 
of each trip. At the end I used the length and the given duration to colorize the visualization. Red => slow streets, yellow => normal
white => fast streets. 

This is a nice visualization I think and might be useful for the computation as you can find out which trips might be slow and which might be fast.

Hope you enjoy the visualization and if you do: Thanks for the upvote ;)

"""



import cv2
import io
import numpy as np
import random
import math
import csv as csv
import json
import zipfile
import pandas as pd
import pickle
import numpy as np
from igraph import *
import matplotlib.pyplot as plt
import numpy as np


# generates an image of size 2000x2000
IMAGE_SIZE = 2000
v_total, l_total, d_total = 0,0,0
v_max, l_max, d_max = 0,0,0
ignored = 0


def get_graph(file_source, coordinates, name="", source=1):
    """
        generate a graph of NY to be able to use dijkstra for the shortest paths
    """
    file = open(file_source + ".gr")

    g = Graph(directed=True)
    
    vertices = {}
    vert_names = []

    arr_of_edges = []
    arr_of_weights = []
    for line in file:
        line_parts = line.strip().split(" ")
        if line_parts[0] != "a":
            continue
        _, tail, head, cost = line_parts

        if tail not in vertices:
            vertices[tail] = len(vertices)
            vert_names.append(tail)
        if head not in vertices:
            vertices[head] = len(vertices)
            vert_names.append(head)

        arr_of_edges.append((vertices[tail], vertices[head]))
        arr_of_weights.append(int(cost))


    g.add_vertices(len(vertices))
    g.add_edges(arr_of_edges)
    g.es['weight'] = arr_of_weights
    print("#nodes", len(g.vs()))
    print("#edges", len(g.es()))
    g.vs['name'] = vert_names
    
    return g

def load_obj(name):
    with open('../input/nyc-taxis-combined-with-dimacs/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def create_graph(coordinates=False):
    G = get_graph("../input/nyc-taxis-combined-with-dimacs/USA-road-d.NY", coordinates)
    return G
    
def getDist(posStart,posEnd):
    dispair = (posStart-posEnd)*[83.818,111.23] # distance in kilometers for longitude and latitude
    return np.sqrt(np.power(dispair[0],2)+np.power(dispair[1],2))

def point(image,v,x,y):
    """
      add the velocity and the 1 to a given pixel to be able to calculate the average velocity at the end
    """
    if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
        image[x,y] += [v,0,1]
    return image

def drawTrips(image,data,duration,length,border):
    global v_total, l_total, d_total, v_max, l_max, d_max, ignored

    minlat,maxlon,maxlat,minlon = border
    absLat = abs(minlat-maxlat)
    absLon = abs(minlon-maxlon)
    
    
    v = length/duration
    
    # ignore this trip if the velocity is too high
    if v > 200:
        ignored += 1
        return image
    v_total += v
    l_total += length
    d_total += duration
    if v >= v_max:
        v_max = v
    if length >= l_max:
        l_max = length
    if duration >= d_max:
        d_max = duration
    

    for i in range(len(data)-1):
        if (data[i][0] != False): # only if there are no missing gps datas
            long = float(data[i][0])
            lat = float(data[i][1])
            longE = float(data[i+1][0])
            latE = float(data[i+1][1])
            
            
            pxXS = int(np.rint((long-minlon)*(np.shape(image)[1]/absLon))-1)
            pxYS = int(np.rint((maxlat-lat)*(np.shape(image)[0]/absLat))-1)

            pxXSE = int(np.rint((longE-minlon)*(np.shape(image)[1]/absLon))-1)
            pxYSE = int(np.rint((maxlat-latE)*(np.shape(image)[0]/absLat))-1)

            difX = pxXSE-pxXS    
            difY = pxYSE-pxYS

            if abs(difX)+abs(difY) > 70:
                break

            lX = pxXS
            lY = pxYS
            
            # set a point at the start position
            image = point(image,v,lY,lX)

            # draw a line to the next position
            while abs(difX) > 0 or abs(difY) > 0:
                if abs(difX) == 1 and abs(difY) == 1:
                    break

                nX, nY = 0, 0
                fX, fY = 0, 0

                if difY == 0:
                    difX = 0
                    if 0 <= pxYS < IMAGE_SIZE:
                        if lX != pxXS:
                            s = 2
                        else:
                            s = 1
                        if difX < 0:
                            fX = -1
                        for nX in range(s,abs(difX)):
                            image = point(image,v,lY,lX+fX*nX)  
                elif difX == 0:
                    difY = 0
                    if 0 <= pxXS < IMAGE_SIZE:
                        if difY < 0:
                            fY = -1
                        if lY != pxYS:
                            s = 2
                        else:
                            s = 1
                        for nY in range(s,abs(difY)):
                            image = point(image,v,lY+fY*nY,lX)  

                else:
                    if 0.5 < abs(difX)/abs(difY) < 1.5:
                        if difX < 0:
                            difX += 1
                            nX = -1
                        else:
                            difX -= 1
                            nX = 1

                        if difY < 0:
                            difY += 1
                            nY = -1
                        else:
                            difY -= 1
                            nY = 1
                        
                        image = point(image,v,lY+nY,lX+nX)
                        lY = lY+nY
                        lX = lX+nX

                    elif abs(difX)/abs(difY) >= 1.5:
                        if difX < 0:
                            difX += 1
                            nX = -1
                        else:
                            difX -= 1
                            nX = 1
                        
                        image = point(image,v,lY,lX+nX)
                        lX += nX

                    elif abs(difX)/difY <= 0.5:
                        if difY < 0:
                            difY += 1
                            nY = -1
                        else:
                            difY -= 1
                            nY = 1
                        
                        image = point(image,v,lY+nY,lX)
                        lY += nY

            if i+1 == len(data)-1:
                image = point(image,v,pxYSE,pxXSE)
    return image

def draw(image,name="img"):
    """
        Draw the average velocity by dividing the total velocity by the number of samples for each point
    """
    
    img = np.copy(image)

    max0 = np.max(img[:,:,0])
    max2 = np.max(img[:,:,2])


    img[:,:,2] = np.nan_to_num(img[:,:,0]/img[:,:, 2])
    img[:,:,1] = 0
    img[:,:,0] = 0

    # get the maximum accumlated distance and set that one to 255
    maxF = np.max(img[:,:,2])
    minF = np.min(img[:,:,2])
    
    img[:,:,2] *= 255/maxF

    numpy_hist = plt.figure()

    flat = img[:,:,2].flatten()
    flatNoZ = flat[flat > 0]
    
    maxF = np.max(flatNoZ)
    minF = np.min(flatNoZ)
    
    # plt.hist(flatNoZ, bins=range(0,200,10))

    # plt.savefig("numpy_hist")

    img[:,:,1]  = img[:,:,2]
    img[:,:,0] = img[:,:,2]
    img[:,:,1][img[:,:,1] < 60] = 0 # slow taxis => red
    img[:,:,0][img[:,:,0] < 100] = 0 # fast taxis => yellow, else white
    img[:,:,2][img[:,:,2] > 0] = 255
    img[:,:,1][img[:,:,1] > 0] = 255
    img[:,:,0][img[:,:,0] > 0] = 255

    cv2.imwrite("%s.png" % name, img)
        
# generate a 4000 x 4000 image in RGB mode
image = np.zeros((IMAGE_SIZE,IMAGE_SIZE,3))


minlat = 40.68 # most southern part
minlon = -74.1 # most western part
maxlat = 40.86 # most northern part
maxlon = -73.8 # most eastern part


G = create_graph()

mapping = load_obj('nyc_gps')
G.vs['long'] = mapping[:,0]
G.vs['lat'] = mapping[:,1]



df = pd.read_csv('../input/nyc-taxis-combined-with-dimacs/train_added_node.csv')

# 1458644 for full dataset now only 10001
for i in range(10001):
    s = int(df.iloc[i]['s_id'])
    t = int(df.iloc[i]['t_id'])
    duration = int(df.iloc[i]['trip_duration'])
    if i % 500 == 0:
        print(i,s,t)

    # if at least part of the trip is on the image
    if -74.1 <= G.vs[s]['long'] <= -73.8 and 40.68 <= G.vs[s]['lat'] <= 40.86:
        if -74.1 <= G.vs[t]['long'] <= -73.8 and 40.68 <= G.vs[t]['lat'] <= 40.86:
            sp = G.get_shortest_paths(s,t,'weight')[0]
            l = G.shortest_paths_dijkstra(s,t,'weight')[0][0]

            data=[] 
            for p in sp:
                data.append([G.vs[p]['long'],G.vs[p]['lat']])
            image = drawTrips(image,data,duration,l,[minlat,maxlon,maxlat,minlon])
            
    if i % 1000 == 0:
        draw(image)

