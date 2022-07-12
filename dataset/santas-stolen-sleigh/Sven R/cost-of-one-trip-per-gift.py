"""
this will compute the WRW of doing one trip per gift,
which is far from optimal:
29.12101101558951 G
This takes some time (over 1:30 min on my system)
because of the large number of trips.
"""

import pandas as pd
import numpy as np
from haversine import haversine

north_pole = (90,0)
weight_limit = 1000
sleigh_weight = 10


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


def cost(gifts, solution):
    all_trips = solution.merge(gifts, on='GiftId')
    return weighted_reindeer_weariness(all_trips)


def separate_trips(gifts):
    solution = pd.DataFrame({"GiftId": range(1,100001), "TripId": range(0,100000)})
    return cost(gifts, solution)


if __name__ == "__main__":
    gifts = pd.read_csv('../input/gifts.csv')
    print("Cost of one trip per gift: {} G".format(separate_trips(gifts)/1E9))

