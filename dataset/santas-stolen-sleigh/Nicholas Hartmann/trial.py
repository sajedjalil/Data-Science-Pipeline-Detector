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

data.sort(key = lambda x: x[1]['pole_distance'])
output = []

trip = 0
trip_weight = 0
for i in range(len(data)):
    if 1000 - trip_weight >= data[i][1]['weight']:
        output.append({'GiftId':data[i][0], 'TripId':trip})
        trip_weight += data[i][1]['weight']
    else:
        trip += 1
        output.append({'GiftId':data[i][0], 'TripId':trip})
        trip_weight = data[i][1]['weight']

with open('submission.csv', 'w') as csvfile:
    fields = ['GiftId', 'TripId']
    writer = csv.DictWriter(csvfile, fieldnames = fields)
    
    writer.writeheader()
    for i in range(len(output)):
        writer.writerow(output[i])
        