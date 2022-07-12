north_pole = (90,0)
weight_limit = 1000
sleigh_weight = 10

import pandas as pd
import numpy as np
from haversine import haversine

def npHaversine(x):
    return haversine((x[0],x[1]),(x[2],x[3]))

def weighted_trip_length(this_trip): 
    tuples = this_trip[['Latitude','Longitude']].values
    # adding the last trip back to north pole, with just the sleigh weight

    coordinates = np.zeros( (tuples.shape[0]+1,4) )
    cumWeights = np.zeros( tuples.shape[0]+1 )
    
    coordinates[:-1,:2] = tuples
    coordinates[-1,:2] = north_pole
    coordinates[1:,2:] = tuples
    coordinates[0,2:] = north_pole
        
    dists = np.array([npHaversine(x) for x in coordinates])
       
    cumWeights[:-1] = this_trip.Weight.values[::-1].cumsum()[::-1]
    cumWeights = cumWeights + sleigh_weight
  
    return (dists*cumWeights).sum()

def weighted_reindeer_weariness(all_trips):
    groups = all_trips.groupby('TripId')    
    
    if any(groups.Weight.sum() > weight_limit):
        raise Exception("One of the sleighs over weight limit!")
 
    dist = groups.apply(weighted_trip_length)
    
    return dist.sum()

gifts = pd.read_csv('../input/gifts.csv')
sample_sub = pd.read_csv('../input/sample_submission.csv')

all_trips = sample_sub.merge(gifts, on='GiftId')

print(weighted_reindeer_weariness(all_trips))
