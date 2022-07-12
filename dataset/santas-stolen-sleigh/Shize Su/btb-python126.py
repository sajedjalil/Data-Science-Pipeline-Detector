import sqlite3
import pandas as pd
from haversine import haversine

north_pole = (90,0)
weight_limit = 1010.0 #1000.0

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
n = 60
for i in range(90,-90,int(-180/n)):
    i_ += 1
    j_ = 0
    for j in range(180,-180,int(-360/n)):
        j_ += 1
        cu = c.cursor()
        cu.execute("UPDATE gifts SET i=" + str(i_) + ", j=" + str(j_) + " WHERE ((Latitude BETWEEN " + str(i - (180/n)) + " AND  " + str(i) + ") AND (Longitude BETWEEN " + str(j - (360/n)) + " AND  " + str(j) + "));")
        c.commit()

trips = pd.read_sql("SELECT * FROM (SELECT * FROM gifts WHERE TripId IS NULL ORDER BY j DESC, Longitude ASC, Latitude ASC LIMIT 100) ORDER BY Latitude DESC",c)

while len(trips.GiftId)>0:
    g = []
    t_ += 1
    w_ = 0.0
    l_ = north_pole

    for i in range(len(trips.GiftId)):
        if (((w_ + float(trips.Weight[i]))<= weight_limit) and (trips.GiftId[i] not in g)):
            w_ += float(trips.Weight[i])
            d_ += haversine(l_, (trips.Latitude[i],trips.Longitude[i]))
            f_ += d_ * trips.Weight[i]
            l_ = (trips.Latitude[i],trips.Longitude[i])
            g.append(trips.GiftId[i])
            
    d_ += haversine(l_, north_pole)
    f_ += d_ * 10 #sleigh weight for whole trip
    print(t_,d_,f_)
    cu = c.cursor()
    cu.execute("UPDATE gifts SET TripId = " + str(t_) + " WHERE GiftId IN(" + (",").join(map(str,g)) + ");")
    c.commit()

    trips = pd.read_sql("SELECT * FROM (SELECT * FROM gifts WHERE TripId IS NULL ORDER BY j DESC, Longitude ASC, Latitude ASC LIMIT 100) ORDER BY Latitude DESC",c)
    d_ = 0.0
    #break

submission = pd.read_sql("SELECT GiftId, TripId FROM gifts ORDER BY TripId ASC, Latitude DESC;",c)
submission.to_csv("submission.csv", index=False)

if f_ < 12630720472.0:
    print("Improvement", f_, f_-12630720472.0, 12630720472.0)
else:
    print("Try again", f_,f_-12630720472.0, 12630720472.0)