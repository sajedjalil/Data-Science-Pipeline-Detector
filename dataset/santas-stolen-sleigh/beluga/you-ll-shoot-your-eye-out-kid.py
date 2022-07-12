import pandas as pd
from os import getcwd
from os.path import join
import matplotlib.pyplot as plt
from haversine import haversine
import sqlite3


plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = [16, 12]
north_pole = (90,0)
weight_limit = 1000.0

f_ = 0.0
t_ = 0
d_ = 0.0

gifts = pd.read_csv("../input/gifts.csv").fillna(" ")
c = sqlite3.connect(":memory:")
gifts.to_sql("gifts",c)
cu = c.cursor()
cu.execute("ALTER TABLE gifts ADD COLUMN 'TripId' INT;")
cu.execute("ALTER TABLE gifts ADD COLUMN 'i' INT;")
cu.execute("ALTER TABLE gifts ADD COLUMN 'j' INT;")
c.commit()

i_ = 0
j_ = 0
n = 90
for i in range(90,-90,int(-180/n)):
    i_ += 1
    j_ = 0
    for j in range(180,-180,int(-360/n)):
        j_ += 1
        cu = c.cursor()
        cu.execute("UPDATE gifts SET i=" + str(i_) + ", j=" + str(j_) + " WHERE ((Latitude BETWEEN " + str(i - (180/n)) + " AND  " + str(i) + ") AND (Longitude BETWEEN " + str(j - (360/n)) + " AND  " + str(j) + "));")
        c.commit()

trips = pd.read_sql("SELECT * FROM (SELECT * FROM gifts WHERE TripId IS NULL ORDER BY j DESC, Longitude ASC, Latitude ASC LIMIT 94) ORDER BY Latitude DESC",c)


while len(trips.GiftId)>0:
    g = []
    t_ += 1
    w_ = 0.0
    l_ = north_pole
    for i in range(len(trips.GiftId)):
        if (w_ + float(trips.Weight[i]))<= weight_limit:
            w_ += float(trips.Weight[i])
            d_ += haversine(l_, (trips.Latitude[i],trips.Longitude[i]))
            f_ += d_ * trips.Weight[i]
            l_ = (trips.Latitude[i],trips.Longitude[i])
            g.append(trips.GiftId[i])
    d_ += haversine(l_, north_pole)
    f_ += d_ * 10  # sleigh weight for whole trip
    # print(t_,d_,f_)
    cu = c.cursor()
    cu.execute("UPDATE gifts SET TripId = " + str(t_) + " WHERE GiftId IN(" + (",").join(map(str,g)) + ");")
    c.commit()

    trips = pd.read_sql("SELECT * FROM (SELECT * FROM gifts WHERE TripId IS NULL ORDER BY j DESC, Longitude ASC, Latitude ASC LIMIT 94) ORDER BY Latitude DESC",c)
    d_ = 0.0
    #break

all_trips = pd.read_sql("SELECT * FROM gifts ORDER BY TripId ASC, Latitude DESC;",c)
fig = plt.figure()
plt.scatter(all_trips['Longitude'], all_trips['Latitude'], c=all_trips['TripId'], cmap= plt.cm.viridis, alpha=0.8, s=8, linewidths=0)
for t in all_trips.TripId.unique():
    trip = all_trips[all_trips['TripId'] == t]
    plt.plot(trip['Longitude'], trip['Latitude'], 'k.-', alpha=0.1)

plt.colorbar()
plt.grid()
plt.title('Trips')
plt.tight_layout()
fig.savefig('Trips.png', dpi=300)

fig = plt.figure()
plt.scatter(all_trips['Longitude'].values, all_trips['Latitude'].values, c='k', alpha=0.1, s=1, linewidths=0)
for t in all_trips.TripId.unique():
    previous_location = north_pole
    trip = all_trips[all_trips['TripId'] == t]
    i = 0
    for _, gift in trip.iterrows():
        plt.plot([previous_location[1], gift['Longitude']], [previous_location[0], gift['Latitude']],
                 color=plt.cm.copper_r(i/90.), alpha=0.1)
        previous_location = tuple(gift[['Latitude', 'Longitude']])
        i += 1
    plt.scatter(gift['Longitude'], gift['Latitude'], c='k', alpha=0.5, s=20, linewidths=0)

plt.scatter(gift['Longitude'], gift['Latitude'], c='k', alpha=0.5, s=20, linewidths=0, label='TripEnds')
plt.legend(loc='upper right')
plt.grid()
plt.title('TripOrder')
plt.tight_layout()
fig.savefig('TripsinOrder.png', dpi=300)
