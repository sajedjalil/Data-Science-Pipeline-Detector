import pandas as pd
import zipfile
import json

def main():
    zf = zipfile.ZipFile('../input/test.csv.zip')
    test = pd.read_csv(zf.open('test.csv'),usecols=['POLYLINE','TRIP_ID'])    
    #test = pd.read_csv("test.csv",usecols=['POLYLINE'])
    
    test.POLYLINE = test.POLYLINE.apply(json.loads)#from string to list of coords
    
    sub = pd.DataFrame()
    sub['TRIP_ID'] = test.TRIP_ID
    sub['TRAVEL_TIME'] = test['POLYLINE'].apply(lambda x:(15.*len(x) / 0.7) if (len(x) > 25) else (50.*len(x)))
    sub.to_csv('Solution_1-7-2015_1.csv', index = False)
    
main()