import pandas as pd
from haversine import haversine

north_pole = (90,0)
weight_limit = 1000.0

def distance(row):
    return haversine(north_pole, ( row['Latitude'], row['Longitude']) )

gifts = pd.read_csv("../input/gifts.csv") 
gifts['distance_from_north'] = gifts.apply(distance, axis = 1)
sgifts = gifts.sort(['distance_from_north'], ascending=[False])


accums = [0, weight_limit]

# lower bound is sum(distance_from_north * weight) for each gift + 
# 20 * (distance from north to the the most distant point of the route)
for i, row in enumerate(sgifts.values):
    GiftId,Latitude,Longitude,Weight,distance_from_north = row;
    accums[0] = accums[0]+Weight*distance_from_north
    if accums[1] + Weight > weight_limit:
        accums[0] = accums[0]+20*distance_from_north
        accums[1] = Weight
    else:
        accums[1] = accums[1] + Weight

print (accums[0])

