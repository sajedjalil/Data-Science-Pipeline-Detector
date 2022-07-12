import pandas as pd
import zipfile
import math

#formula to calculate distance among two gps points
def haversine(coord1,coord2):
    import math
    sin = math.sin
    cos = math.cos
    atan2 = math.atan2
    sqrt = math.sqrt
    
    lon1,lat1=coord1
    lon2,lat2=coord2
    R=6371000 #metres
    phi1=lat1 * (3.1415 / 180)
    phi2=lat2 * (3.1415 / 180)
    Dphi= phi2 - phi1
    Dlambda = (lon2 -lon1) *  (3.1415 / 180)

    a = sin(Dphi / 2) ** 2 + cos(phi1)*cos(phi2) *sin(Dlambda/2)**2
    c = 2 * atan2(sqrt(a),sqrt(1-a))
    d = R*c
    return d
    


def speeds(polyline):
    N=len(polyline)
    v = [0.]*N
    #lon =[]*N
    #Lat =[]*N
    if N == 0:
        return []
    
    for i in range(N - 1):
        v[i] = haversine(polyline[i],polyline[i+1]) / 15.
        #lon[i] = polyline[i]
        #lat[i] = polyline[i+1]
    
    v[N-1]  = haversine(polyline[N-1],polyline[N-2]) / 15.
    
    return v
    #return v,lon,lat



def onSpeed(v):
    N=len(v)
    
    if N < 44:
        return 660.0
    
    if v[N-1] < 3:
        return (N-1)*15
    
    return (N*1.5 - 1) * 15
    
def printSpeeds(v):
    str = 0.0
    for i in range(len(v)):
        #print("#####################")
        #print(v[i])
        for m in range(len(v[i])):
            #print(v[i][m])
            str += math.floor(v[i][m])
        return str
        #print("#####################")
       # str += math.floor(v[i])
    
    #return str

def main():
    zf = zipfile.ZipFile('../input/test.csv.zip')
    test = pd.read_csv(zf.open('test.csv'),usecols=['POLYLINE','TRIP_ID'])    
    #test = pd.read_csv("test.csv",usecols=['POLYLINE'])
    
    test.POLYLINE = test.POLYLINE.apply(eval)#from string to list of coords
    
    test['SPEED'] = test.POLYLINE.apply(speeds)
    
    sub = pd.DataFrame()
    sub['TRIP_ID'] = test.TRIP_ID
    #sub['TRAVEL_TIME'] = printSpeeds(test['SPEED'])
    sub['TRAVEL_TIME'] = test['SPEED'].apply(onSpeed)
    #sub['POLYLINE'] = test.POLYLINE
    
    print (sub)
    
    
    
main()