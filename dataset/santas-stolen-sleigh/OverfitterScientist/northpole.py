import sqlite3
import pandas as pd
from haversine import haversine

north_pole = (90,0)
weight_limit = 990.0

def bb_sort(ll):
    s_limit = 4900
    optimal = False
    ll = [[0,north_pole,10]] + ll[:] + [[0,north_pole,10]] 
    while not optimal:
        optimal = True
        for i in range(1,len(ll) - 2):
            lcopy = ll[:]
            lcopy[i], lcopy[i+1] = ll[i+1][:], ll[i][:]
            if path_opt_test(ll[1:-1]) > path_opt_test(lcopy[1:-1]):
                #print("swap")
                ll = lcopy[:]
                optimal = False
                s_limit -= 1
                if s_limit < 0:
                    optimal = True
                    break
    return ll[1:-1]

def path_opt_test(llo):
    f_ = 0.0
    d_ = 0.0
    l_ = north_pole
    for i in range(len(llo)):
        d_ += haversine(l_, llo[i][1])
        f_ += d_ * llo[i][2]
        l_ = llo[i][1]
    d_ += haversine(l_, north_pole)
    f_ += d_ * 10 #sleigh weight for whole trip
    return f_

gifts = pd.read_csv("../input/gifts.csv").fillna(" ")
c = sqlite3.connect(":memory:")
gifts.to_sql("gifts",c)
cu = c.cursor()
cu.execute("ALTER TABLE gifts ADD COLUMN 'TripId' INT;")
cu.execute("ALTER TABLE gifts ADD COLUMN 'i' INT;")
cu.execute("ALTER TABLE gifts ADD COLUMN 'j' INT;")
c.commit()


for n in [1.25252525]:
    i_ = 0
    j_ = 0
    for i in range(90,-90,int(-180/n)):
        i_ += 1
        j_ = 0
        for j in range(180,-180,int(-360/n)):
            j_ += 1
            cu = c.cursor()
            cu.execute("UPDATE gifts SET i=" + str(i_) + ", j=" + str(j_) + " WHERE ((Latitude BETWEEN " + str(i - (180/n)) + " AND  " + str(i) + ") AND (Longitude BETWEEN " + str(j - (360/n)) + " AND  " + str(j) + "));")
            c.commit()
    
    for limit_ in [67]:
        trips = pd.read_sql("SELECT * FROM (SELECT * FROM gifts WHERE TripId IS NULL ORDER BY i, j, Longitude, Latitude LIMIT " + str(limit_) + " ) ORDER BY Latitude DESC",c)
        t_ = 0
        while len(trips.GiftId)>0:
            g = []
            t_ += 1
            w_ = 0.0
            for i in range(len(trips.GiftId)):
                if (w_ + float(trips.Weight[i]))<= weight_limit:
                    w_ += float(trips.Weight[i])
                    g.append(trips.GiftId[i])
            cu = c.cursor()
            cu.execute("UPDATE gifts SET TripId = " + str(t_) + " WHERE GiftId IN(" + (",").join(map(str,g)) + ");")
            c.commit()
        
            trips = pd.read_sql("SELECT * FROM (SELECT * FROM gifts WHERE TripId IS NULL ORDER BY i, j, Longitude, Latitude LIMIT " + str(limit_) + " ) ORDER BY Latitude DESC",c)
            #break
        
        ou_ = open("submission_opt" + str(limit_) + " " + str(n) + ".csv","w")
        ou_.write("TripId,GiftId\n")
        bm = 0.0
        submission = pd.read_sql("SELECT TripId FROM gifts GROUP BY TripId ORDER BY TripId;", c)
        for s_ in range(len(submission.TripId)):
            trip = pd.read_sql("SELECT GiftId, Latitude, Longitude, Weight FROM gifts WHERE TripId = " + str(submission.TripId[s_]) + " ORDER BY Latitude DESC, Longitude ASC;",c)
            a = []
            for x_ in range(len(trip.GiftId)):
                a.append([trip.GiftId[x_],(trip.Latitude[x_],trip.Longitude[x_]),trip.Weight[x_]])
            b = bb_sort(a)
            if path_opt_test(a) <= path_opt_test(b):
                print(submission.TripId[s_], "No Change", path_opt_test(a) , path_opt_test(b))
                bm += path_opt_test(a)
                for y_ in range(len(a)):
                    ou_.write(str(submission.TripId[s_])+","+str(a[y_][0])+"\n")
            else:
                print(submission.TripId[s_], "Optimized", path_opt_test(a) - path_opt_test(b))
                bm += path_opt_test(b)
                for y_ in range(len(b)):
                    ou_.write(str(submission.TripId[s_])+","+str(b[y_][0])+"\n")
        ou_.close()
        
        benchmark = 12514008574.2
        if bm < benchmark:
            print(n, limit_, "Improvement", bm, bm - benchmark, benchmark)
        else:
            print(n, limit_, "Try again", bm, bm - benchmark, benchmark)
        cu = c.cursor()
        cu.execute("UPDATE gifts SET TripId = NULL;")
        c.commit()
