import pandas as pd
import numpy as np

north_pole = np.array([90.0,0.0])
weight_limit = 1000
sleigh_weight = 10
AVG_EARTH_DIAMETER = 2 * 6371

def haversine(orig, dest):
    '''
    orig: nx2 matrix (lat, lon)
    dest: nx2 matrix (lat, lon)
    '''
    orig, dest = np.radians(orig), np.radians(dest)
    deltas = orig - dest
    dLat, dLon = deltas[:, 0], deltas[:, 1]
    a = np.sin(dLat/2)**2 + np.cos(orig[:,0]) * np.cos(dest[:,0]) * np.sin(dLon/2)**2
    return AVG_EARTH_DIAMETER * np.arcsin(np.sqrt(a))

def weighted_trip_length(group):
    stops = np.vstack([north_pole, group[:, :2]])
    weights = np.append(group[:, 2], sleigh_weight)
    orig, dest = stops, np.roll(stops, -1, axis=0)
    d = haversine(orig, dest)
    inv_cum_weights = np.cumsum(weights[::-1])[::-1]
    return d.dot(inv_cum_weights)

def weighted_reindeer_weariness(all_trips):
    grouped_by_trip = all_trips.groupby('TripId', sort=False)

    if any(grouped_by_trip.Weight.sum() > weight_limit - sleigh_weight):
        raise Exception("One of the sleighs over weight limit!")

    mat = np.ascontiguousarray(all_trips[['Latitude', 'Longitude', 'Weight']].values)

    return np.sum([weighted_trip_length(mat.take(idx, axis=0)) for idx in grouped_by_trip.indices.values()])

if __name__ == '__main__':
    gifts = pd.read_csv('../input/gifts.csv', index_col=0)
    sample_sub = pd.read_csv('../input/sample_submission.csv', index_col=0)
    all_trips = sample_sub.join(gifts)
    print(weighted_reindeer_weariness(all_trips))
