north_pole = (90,0)
weight_limit = 1000
sleigh_weight = 10

import pandas as pd
import numpy as np
from haversine import haversine

def lower_bounds(gifts):
    """
    gifts is a DataFrame according to Kaggle documentation.
    To obtain a true lower bound, we compute separately 
    - a lower bound on the delivery costs
    - a lower bound on the empty sleigh trip costs
    For the latter, allow "fractional" gift weights, i.e., splitting
    a gift at any weight.
    """
    deliverycost = 0.0
    tripcost = 0.0
    trips = 0
    accw = weight_limit  # ensure we start a new trip immediately
    gifts = gifts.sort_values('Latitude', ascending=True)  # furthest first
    # accumulate minimal costs for delivering gifts
    for gift in gifts[['Latitude','Longitude','Weight']].values:
        lat, lng, w = gift
        dist = haversine(north_pole, (lat, lng))
        deliverycost += w * dist  # minimal cost for this gift
        if accw + w > weight_limit:
            # start new trip to this latitude with the rest of this weight
            w -= weight_limit - accw
            tripcost += 2.0 * sleigh_weight * dist
            trips += 1
            accw = w
        else:
            accw += w
        assert 0 < w <= weight_limit
    return (deliverycost, tripcost, trips)


def main():
    gifts = pd.read_csv('../input/gifts.csv')
    (dc, tc, trips) = lower_bounds(gifts)
    ngifts = gifts.shape[0]
    print("{} gifts for cost >= {} G".format(ngifts, dc/1E9))
    print(">= {} trips for cost >= {} G".format(trips, tc/1E9))
    print("Lower bound: {} G ".format((dc+tc)/1E9))


if __name__ == "__main__":
    main()
