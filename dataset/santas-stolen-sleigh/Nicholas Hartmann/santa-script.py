import csv
from haversine import haversine

f = open("../input/gifts.csv", 'r')

reader = csv.DictReader(f)

data = []

north_pole = (90.0, 0.0)
for row in reader:
    obs = {}
    obs['latitude'] = float(row['Latitude'])
    obs['longitude'] = float(row['Longitude'])
    obs['weight'] = float(row['Weight'])
    obs['pole_distance'] = haversine(north_pole, (obs['latitude'], obs['longitude']))
    data.append((row['GiftId'], obs))

f.close()

data.sort(key = lambda x: x[1]['pole_distance'], reverse = True)
output = []

trip = 0
trip_weight = 0
while(len(data) > 0):
    group = []
    base = data.pop(0)
    trip_weight = base[1]['weight']
    group.append({'GiftId':base[0], 'TripId':trip, 'weight':base[1]['weight']})
    data.sort(key = lambda x: haversine((base[1]['latitude'], base[1]['longitude']), (x[1]['latitude'], x[1]['longitude'])))
    while True:
        if(len(data) == 0):
            for j in range(len(group)):
                group[j].pop('weight', None)
                output.append(group[j])
            break
        neighbor = data.pop(0)
        if neighbor[1]['weight'] + trip_weight > 1000:
            data.append(neighbor)
            trip += 1
            trip_weight = 0
            group.sort(key = lambda x: x['weight'], reverse = True)
            for j in range(len(group)):
                group[j].pop('weight', None)
                output.append(group[j])
            break
        else:
            trip_weight += neighbor[1]['weight']
            group.append({'GiftId':neighbor[0], 'TripId':trip, 'weight':neighbor[1]['weight']})
        

with open('submission.csv', 'w') as csvfile:
    fields = ['GiftId', 'TripId']
    writer = csv.DictWriter(csvfile, fieldnames = fields)
    
    writer.writeheader()
    for i in range(len(output)):
        writer.writerow(output[i])
        