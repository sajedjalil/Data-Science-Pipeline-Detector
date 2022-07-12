# Author: Dr Vivi
# Date: Jun 27, 2015

import json
import pandas as pd
import zipfile

zipped = zipfile.ZipFile('../input/test.csv.zip')
taxi_test = pd.read_csv(zipped.open('test.csv'))
taxi_test['POLYLINE'] = taxi_test['POLYLINE'].apply(json.loads)
taxi_test['duration'] = ((taxi_test['POLYLINE'].apply(len))-1)*15
taxi_test['TRAVEL_TIME'] = taxi_test['duration']+495 #why 495? Optimized from my validation sets - might not be the optimal value for the test set
taxi_test[['TRIP_ID', 'TRAVEL_TIME']].to_csv('submission.csv', index=False) #scored .5995