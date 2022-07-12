import pdb
north_pole = (90,0)
weight_limit = 1000
sleigh_weight = 10
import sqlite3

import pandas as pd
import numpy as np
from haversine import haversine

def weighted_trip_length(stops, weights): 
    tuples = [tuple(x) for x in stops.values]
    # adding the last trip back to north pole, with just the sleigh weight
    tuples.append(north_pole)
    weights.append(sleigh_weight)
    
    dist = 0.0
    prev_stop = north_pole
    prev_weight = sum(weights)
    for location, weight in zip(tuples, weights):
        dist = dist + haversine(location, prev_stop) * prev_weight
        prev_stop = location
        prev_weight = prev_weight - weight
    return dist

def weighted_reindeer_weariness(all_trips):
    uniq_trips = all_trips.TripId.unique()
    
    if any(all_trips.groupby('TripId').Weight.sum() > weight_limit):
        raise Exception("One of the sleighs over weight limit!")
 
    dist = 0.0
    for t in uniq_trips:
        this_trip = all_trips[all_trips.TripId==t]
        dist = dist + weighted_trip_length(this_trip[['Latitude','Longitude']], this_trip.Weight.tolist())
    
    return dist    



gifts = pd.read_csv('../input/gifts.csv')
sample_sub = pd.read_csv('../input/sample_submission.csv')

all_trips = sample_sub.merge(gifts, on='GiftId')

#print(weighted_reindeer_weariness(all_trips))



gifts = pd.read_csv("../input/gifts.csv").fillna(" ")
c = sqlite3.connect(":memory:")
gifts.to_sql("gifts",c)
cu = c.cursor()
cu.execute("ALTER TABLE gifts ADD COLUMN 'TripId' INT;")
cu.execute("ALTER TABLE gifts ADD COLUMN 'LongDensity' FLOAT;")
cu.execute("ALTER TABLE gifts ADD COLUMN 'LatitudeWeightedAverage' INT;")
cu.execute("ALTER TABLE gifts ADD COLUMN 'TotalTripMass' FLOAT;")
c.commit()

Ascending_Longitude = pd.read_sql("Select * from gifts order by Longitude;",c)
s = 0

for p in Ascending_Longitude:
    
    Accum_mass = 0
    n= 0 # the number of presents after present s that can fit in a sleigh
    Post_Present_Mass = 0
    Min_Long = Ascending_Longitude.Longitude[s]
    Max_Long = Ascending_Longitude.Longitude[s]
    while Post_Present_Mass <1000:
        Next_Present_Index = s + n
        #print(Ascending_Longitude.Weight[Next_Present_Index])
        Post_Present_Mass = Accum_mass + Ascending_Longitude.Weight[Next_Present_Index]
        if Post_Present_Mass < 1000:
            Accum_mass = Accum_mass + Ascending_Longitude.Weight[s+n]
            Min_Long = min(Ascending_Longitude.Longitude[Next_Present_Index], Min_Long)
            Max_Long = max(Ascending_Longitude.Longitude[Next_Present_Index], Max_Long)
        n=n+1
        
    print(s)
    print(Accum_mass)
    print(Max_Long - Min_Long)
    print(Ascending_Longitude.GiftId[s])

    
    cu.execute("UPDATE gifts Set TotalTripMass = " + str(Accum_mass) + ", LongDensity = " + str(Accum_mass/(Max_Long - Min_Long)) + " where GiftId = " + str(Ascending_Longitude.GiftId[s]) + ";")
    #c.commit()
    s=s+1

def get_posts():
    cu.execute("Select * from gifts order by Longitude LIMIT 8;")
    print(cu.fetchall())
   # cu.commit
    
get_posts ()



