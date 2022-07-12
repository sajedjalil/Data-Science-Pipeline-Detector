import pandas as pd
import numpy as np

AVG_EARTH_RADIUS = 6371  # in km
north_pole = (90,0)
weight_limit = 1000
sleigh_weight = 10

def haversine_v2(lat1, lng1, lat2, lng2):

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng / 2) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def weighted_trip_length_v2(trip_df):
    
    lat = np.append(trip_df.Latitude, north_pole[0])
    lng = np.append(trip_df.Longitude, north_pole[1])
    lat1 = np.roll(lat, 1)
    lng1 = np.roll(lng, 1)

    trip_wgts = np.append((trip_df.Weight.sum() - trip_df.Weight.cumsum().shift(1).fillna(0)), 0.0) + sleigh_weight 

    trip_dist = haversine_v2(lat1, lng1, lat, lng)

    return trip_dist.dot(trip_wgts)

def weighted_reindeer_weariness_v2(all_trips):
    
    if any(all_trips.groupby('TripId').Weight.sum() > weight_limit - sleigh_weight):
        raise Exception("One of the sleighs over weight limit!")
    return all_trips.groupby('TripId').apply(weighted_trip_length_v2).sum()

if __name__ == '__main__':
    gifts = pd.read_csv('../input/gifts.csv')
    sample_sub = pd.read_csv('../input/sample_submission.csv')

    all_trips = sample_sub.merge(gifts, on='GiftId')

    print (weighted_reindeer_weariness_v2(all_trips))
