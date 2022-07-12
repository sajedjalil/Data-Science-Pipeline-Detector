import pandas as pd
import numpy as np
import datetime
import math
import time
import os
from sklearn.ensemble import RandomForestClassifier

types = {'row_id': np.dtype(int),
        'x': np.dtype(float),
        'y': np.dtype(float),
        'accuracy': np.dtype(int),
        'time': np.dtype(int),
        'place_id': np.dtype(str )}

dataset = pd.read_csv('../input/train.csv',dtype=types,index_col=0)
submit = pd.read_csv('../input/test.csv',dtype=types,index_col=0)
#sample = pd.read_csv('sample_submission.csv',dtype=types,index_col=0)
start_time = time.time()

total_scores=0.0
num_scores=0.0
split_t=math.floor((0.9)*786239)

size = 10.0
x_step = 0.1
y_step = 0.1
n_estimators=300
n_jobs=1

def check():
	global start_time
	return time.time() - start_time

def gen_ranges(size,step):
	return list(zip(np.arange(0,size,step), np.arange(step, size+step, step)));

x_ranges = gen_ranges(size,x_step)
y_ranges = gen_ranges(size,y_step)

print('Calculate hour, weekday, month and year for train and test')

def preprocess(df):
	df['hour'] = (df['time']//60)%24+1 # 1 to 24
	df['weekday'] = (df['time']//1440)%7+1
	df['month'] = (df['time']//43200)%12+1 # rough estimate, month = 30 days
	df['year'] = (df['time']//525600)+1 
	return df

dataset = preprocess(dataset)
submit = preprocess(submit)

train = dataset[dataset.time < split_t]
test = dataset[dataset.time >= split_t]

def select_columns(df):
    return df[['x','y','accuracy','time', 'hour', 'weekday', 'month', 'year']]

def get_grid(df,x_min,x_max,y_min,y_max):
    return df[(df['x'] >= x_min) &
            (df['x'] < x_max) &
            (df['y'] >= y_min) &
            (df['y'] < y_max)]

def get_preds(train,test,x_ranges,y_ranges):
    global total_scores
    global num_scores
    preds_total = pd.DataFrame()
    for x_min,x_max in x_ranges:
        for y_min,y_max in y_ranges:
            x_max = round(x_max,4)
            x_min = round(x_min,4)
            y_max = round(y_max,4)
            y_min = round(y_min,4)
            
            if x_max == size:
                x_max += 0.001
            if y_max == size:
                y_max += 0.001 	
                
            grid_train = get_grid(train,x_min,x_max,y_min,y_max)
            grid_test = get_grid(test,x_min,x_max,y_min,y_max)
            X_train_grid = select_columns(grid_train)
            y_train_grid = grid_train[['place_id']].values.ravel()
            X_test_grid = select_columns(grid_test)
			
            clf = RandomForestClassifier(n_estimators=n_estimators,n_jobs=n_jobs)	
            clf.fit(X_train_grid,y_train_grid)
			
            preds = dict(zip([el for el in clf.classes_],zip(*clf.predict_proba(X_test_grid))))
            preds = pd.DataFrame.from_dict(preds)
            
            preds['predict1'], preds['predict2'],preds['predict3'] = zip(*preds.apply(lambda x: preds.columns[x.argsort()[::-1][:3]].tolist(),axis=1))
            preds['row_id']=X_test_grid.index.values
            if 'place_id' in grid_test.columns.values:
                preds['place_id'] = grid_test[['place_id']].values.ravel()		
                preds['score'] = ((preds.place_id == preds.predict1)*1.0 + (preds.place_id == preds.predict2)*0.5- (preds.predict1==preds.predict2)*0.5 + (preds.place_id == preds.predict3)*(1.0/3.0) - ((preds.predict1 == preds.predict2) | (preds.predict1 == preds.predict3))*(1.0/3.0)).astype(float)
                sum = preds.score.sum()
                count = preds.score.count()
                score =sum/count
                if y_max > 10 or y_max == math.floor(y_max):
                    print (x_min,x_max,y_min,y_max,"Score:",score)
                total_scores += sum
                num_scores += count
				
            preds = preds[['predict1','predict2','predict3','row_id']]
				
            preds_total = pd.concat([preds_total,preds], axis=0)
            if y_max > 10 or y_max == math.floor(y_max):
                print (x_min,x_max,y_min,y_max,"Elapsed time cell:",check())
        print (x_min,x_max,y_min,y_max,"Elapsed time row:",check())
        exit()
    return preds_total
			

preds_test_total = get_preds(train,test,x_ranges,y_ranges)

print ("Final Score: ",(total_scores/(num_scores+.0000000001)))

preds_submit_total = get_preds(dataset,submit,x_ranges,y_ranges)

preds_submit_total['place_id']=[str(p1)+" "+str(p2)+" "+str(p3) for (p1,p2,p3) in zip(preds_submit_total.predict1.values,preds_submit_total.predict2.values,preds_submit_total.predict3.values)]

sub_file = os.path.join('submission_rf_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
preds_submit_total.set_index('row_id',inplace=True)
#print preds_submit_total.head()
preds_submit_total[['place_id']].to_csv(sub_file)
#print "Score: ",(preds_test_total.score.sum()/preds_test_total.score.count())
print("Elapsed time overall: %s seconds" % (time.time() - start_time))
