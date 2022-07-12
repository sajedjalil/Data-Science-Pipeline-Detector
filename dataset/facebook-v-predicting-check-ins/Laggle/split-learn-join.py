# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import os

if not os.path.exists("train/"):
    os.makedirs("train/")
    
if not os.path.exists("test/"):
    os.makedirs("test/")
    
if not os.path.exists("out/"):
    os.makedirs("out/")

############################################split train##########################################

print('starting...')
f = open("../input/train.csv", "r")
header = f.readline()
total = 0

sub_region = dict()
length = 10.0
divisions = 10
block = length / divisions
overlap = .05

while 1:
#for i in range (0,100000):
    line = f.readline().strip()
    total += 1
    
    if total % 1000000 == 0:
        print (total)

    if line == '':
        break

    arr = line.split(",")
    row_id = arr[0]
    x = float(arr[1])
    y = float(arr[2])
    accuracy = arr[3]
    time = arr[4]
    place_id = arr[5]
    
    #main
    istart = int(x/length * divisions)
    jstart = int(y/length * divisions)
    
    xstart = istart * block
    xend = xstart + block

    if xstart == 10.0:
        xstart = xstart - block
        xend = xstart + block
    
    ystart = jstart * block
    yend = ystart + block  
    
    if ystart == 10.0:
        ystart = ystart - block
        yend = ystart + block 
    
    key = str(xstart) + " " + str(xend) + " " + str(ystart) + " " + str(yend)    
    
    if key not in sub_region.keys():         
        sub_region[key] = []
        pass 
        
    sub_region[key].append(line)        

    
    #x overlap
    if x <= xstart + overlap and xstart > 0.0:
        xstart2 = xstart - block
    elif x >= xend - overlap and xend < 10.0:
        xstart2 = xstart + block
    else:
        xstart2 = -1
        
    if xstart2 != -1:  
        xend2 = xstart2 + block    
        keyx = str(xstart2) + " " + str(xend2) + " " + str(ystart) + " " + str(yend)   
        if keyx not in sub_region.keys():         
            sub_region[keyx] = []
            pass
        sub_region[keyx].append(line)  

    #y overlap
    if y <= ystart + overlap and ystart > 0.0:
        ystart2 = ystart - block
    elif y >= yend - overlap and yend < 10.0:
        ystart2 = ystart + block
    else:
        ystart2 = -1

    if ystart2 != -1:  
        yend2 = ystart2 + block  
        keyy = str(xstart) + " " + str(xend) + " " + str(ystart2) + " " + str(yend2)   
        if keyy not in sub_region.keys():         
            sub_region[keyy] = []
            pass
        sub_region[keyy].append(line)   
 
    #xy overlap
    if x <= xstart + overlap and xstart > 0.0:
        xstart3 = xstart - block
        if y <= ystart + overlap and ystart > 0.0:
            ystart3 = ystart - block
        elif y >= yend - overlap and yend < 10.0:
            ystart3 = ystart + block   
        else:
            ystart3 = -1
    elif x >= xend - overlap and xend < 10.0:
        xstart3 = xstart + block
        if y <= ystart + overlap and ystart > 0.0:
            ystart3 = ystart - block
        elif y >= yend - overlap and yend < 10.0:
            ystart3 = ystart + block  
        else:
            ystart3 = -1            
    else:
        xstart3 = -1   
        ystart3 = -1

    if ystart3 != -1:  
        xend3 = xstart3 + block
        yend3 = ystart3 + block  
        keyxy = str(xstart3) + " " + str(xend3) + " " + str(ystart3) + " " + str(yend3)   
        if keyxy not in sub_region.keys():         
            sub_region[keyxy] = []
            pass
        sub_region[keyxy].append(line)         
       
f.close()

print('writing...')   

total = 0
for key in sub_region.keys():
    total += 1
    if total % 10 == 0:
        print (total)
    f = open("train/"+key+".csv", "w")
    f.write(header+"\n")
    for line in sub_region[key]:
        f.write(line+"\n")
    f.close()
    
print('done...')    


############################################split test##########################################
print('starting...')
f = open("../input/test.csv", "r")
header = f.readline()
total = 0

sub_region = dict()
length = 10.0
divisions = 10
block = length / divisions
overlap = .05

while 1:
#for i in range (0,10000):
    line = f.readline().strip()
    total += 1
    
    if total % 100000 == 0:
        print (total)

    if line == '':
        break

    arr = line.split(",")
    row_id = arr[0]
    x = float(arr[1])
    y = float(arr[2])
    accuracy = arr[3]
    time = arr[4]
    #place_id = arr[5]
    
    #main
    istart = int(x/length * divisions)
    jstart = int(y/length * divisions)
    
    xstart = istart * block
    xend = xstart + block

    if xstart == 10.0:
        xstart = xstart - block
        xend = xstart + block
    
    ystart = jstart * block
    yend = ystart + block  
    
    if ystart == 10.0:
        ystart = ystart - block
        yend = ystart + block 
    
    key = str(xstart) + " " + str(xend) + " " + str(ystart) + " " + str(yend)    
    
    if key not in sub_region.keys():         
        sub_region[key] = []
        pass 
        
    sub_region[key].append(line)        

f.close()

print('writing...')   

total = 0
for key in sub_region.keys():
    total += 1
    if total % 10 == 0:
        print (total)
    f = open("test/"+key+".csv", "w")
    f.write(header+"\n")
    for line in sub_region[key]:
        f.write(line+"\n")
    f.close()
    
print('done...')   

############################################learn##########################################

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing

length = 10.0
divisions = 10
block = length / divisions

for i in range (0, divisions):
    xstart = i * block
    xend = xstart + block
    for j in range (0, divisions):
        ystart = j * block
        yend = ystart + block

        key = str(xstart) + " " + str(xend) + " " + str(ystart) + " " + str(yend)

        train_df = pd.read_csv("train/"+key+".csv", encoding="ISO-8859-1", header=0)
        test_df = pd.read_csv("test/"+key+".csv",  encoding="ISO-8859-1", header=0)

        # feature engineering
        initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]') #Arbitrary decision
        d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') for mn in train_df.time.values)

        train_df['hour'] = d_times.hour
        train_df['weekday'] = d_times.weekday
        train_df['day'] = d_times.day
        train_df['month'] = d_times.month
        train_df['year'] = d_times.year
        train_df['logacc'] = np.log(train_df.accuracy)
        train_df = train_df.drop(['time'], axis=1)

        train_df = train_df.groupby("place_id").filter(lambda x: len(x) > 500)

        d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') for mn in test_df.time.values)
        test_df['hour'] = d_times.hour
        test_df['weekday'] = d_times.weekday
        test_df['day'] = d_times.day
        test_df['month'] = d_times.month
        test_df['year'] = d_times.year
        test_df['logacc'] = np.log(test_df.accuracy)
        test_df = test_df.drop(['time'], axis=1)

        # label encoding
        le = preprocessing.LabelEncoder()
        le.fit(train_df['place_id'].as_matrix())
        train_Y = le.transform(train_df['place_id'].as_matrix())

        # prepare for learning
        train_X = train_df.drop('place_id', 1).drop('row_id', 1).as_matrix()
        test_X = test_df.drop('row_id', 1).as_matrix()

        xg_train = xgb.DMatrix( train_X, label=train_Y)
        xg_test = xgb.DMatrix( test_X )

        # setup parameters for xgboost
        param = {}
        # use softmax multi-class classification
        param['objective'] = 'multi:softprob'
        # scale weight of positive examples
        param['eta'] = 0.2
        param['max_depth'] = 10
        param['silent'] = 1
        param['nthread'] = 4
        param['num_class'] = len(train_df['place_id'].unique())

        #watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
        watchlist = [ (xg_train,'train') ]
        num_round = 10
        bst = xgb.train(param, xg_train, num_round, watchlist );

        # process results
        pred_raw = bst.predict(xg_test)

        def get_top3_places_and_probs(row):
            row.sort_values(inplace=True)
            inds = row.index[-3:][::-1].tolist()
            return inds
            
        result_xgb_df = pd.DataFrame(index=test_df.row_id,data=pred_raw)
        result_xgb_df['pred'] = result_xgb_df.apply(get_top3_places_and_probs,axis=1)
        result_xgb_df['pred_0'] = result_xgb_df['pred'].map(lambda x: x[0])
        result_xgb_df['pred_1'] = result_xgb_df['pred'].map(lambda x: x[1])
        result_xgb_df['pred_2'] = result_xgb_df['pred'].map(lambda x: x[2])

        result_xgb_df['pred_0'] = result_xgb_df['pred_0'].apply(le.inverse_transform)
        result_xgb_df['pred_1'] = result_xgb_df['pred_1'].apply(le.inverse_transform)
        result_xgb_df['pred_2'] = result_xgb_df['pred_2'].apply(le.inverse_transform)
        result_xgb_df['place_id'] = result_xgb_df['pred_0'].map(str) + " " + result_xgb_df['pred_1'].map(str) + " " + result_xgb_df['pred_2'].map(str)

        submit = result_xgb_df[['place_id']]
        submit.to_csv("out/"+key+".csv")
        print (key)

############################################join results##########################################
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing

length = 10.0
divisions = 10
block = length / divisions

df_all = None

for i in range (0, divisions):
    xstart = i * block
    xend = xstart + block
    for j in range (0, divisions):
        ystart = j * block
        yend = ystart + block   
        
        key = str(xstart) + " " + str(xend) + " " + str(ystart) + " " + str(yend) 

        df = pd.read_csv("out/"+key+".csv", encoding="ISO-8859-1", header=0)
        
        if df_all is None:
            df_all = df
        else:
            df_all = df_all.append(df)
            
        print (key)
        
df_all.to_csv("out.csv", index = False)





















