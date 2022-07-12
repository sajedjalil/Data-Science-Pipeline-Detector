import pandas as pd
import numpy as np
import datetime
import time
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import cPickle
import os.path
import xgboost as xgb

def calcRange(size, step):
   x_ranges = list(zip(np.arange(size['x_min'], size['x_max'], step['x']), np.arange(step['x'], size['x_max'] + step['x'],  step['x'])))
   y_ranges = list(zip(np.arange(size['y_min'], size['y_max'], step['y']), np.arange(step['y'], size['y_max'] + step['y'], step['y'])))

   aRange = dict(x=x_ranges,
                 y=y_ranges)
   print (aRange['x'])
   print (aRange['y'])
   return aRange

def timeEngineering(train):
   print('Calculate hour, weekday, month and year')
   train['hour'] = (train['time']//60)%24+1 # 1 to 24
   train['weekday'] = (train['time']//1440)%7+1
   train['month'] = (train['time']//43200)%12+1 # rough estimate, month = 30 days
   train['year'] = (train['time']//525600)+1 

   print('shape after time engineering')
   print(train.shape)
   return train

def updateArea(area, size):
   area['x_min'] = round(area['x_min'], 4) 
   area['x_max'] = round(area['x_max'], 4)     
   area['y_min'] = round(area['y_min'], 4) 
   area['y_max'] = round(area['y_max'], 4)     
   if area['x_max'] == size['x_max']:
      area['x_max'] = area['x_max'] + 0.001      
   if area['y_max'] == size['y_max']:
      area['y_max'] = area['y_max'] + 0.001
   return area

def getTestGrid(data, area):
   grid = data[(data['x'] >= area['x_min']) &
               (data['x'] < area['x_max']) &
               (data['y'] >= area['y_min']) &
               (data['y'] < area['y_max'])]
   return grid
   
def findBorders(data, aRange):
   start_time = time.time()
   data = data.sort_values(by=['x','y'], ascending=[True, True])
   print("Elapsed time for sorting: %s seconds" % (time.time() - start_time))
   x_min = 0
   x_max = 0
   for row in data.iterrows():
     # rows = [row['x'] > 1.0:
          print (row)
       
def predictY(x_test, clf):
   #x_test = x_test[features]
   preds = dict(zip([el for el in clf.classes_], zip(*clf.predict_proba(x_test[features]))))
   preds = pd.DataFrame.from_dict(preds)
   b = preds
   #a = zip(*preds.apply(lambda x: preds.columns[x.argsort()[::-1][:2]].tolist(), axis=1))
   try:
      preds['0_'], preds['1_'], preds['2_'] = zip(*preds.apply(lambda x: preds.columns[x.argsort()[::-1][:3]].tolist(), axis=1))
      preds = preds[['0_','1_','2_']]
   except TypeError:
      colnames = []
      for i in range(0, a.shape[1]):
         colname = str(i)+'_'
         preds[colname] = a[a.columns[i]]
         colnames.append(colname)
      preds = preds[colnames]
   #TODO something wrong (double check)
   preds['row_id'] = x_test['row_id'].reset_index(drop=True)
   preds['place_id'] = preds[['0_','1_','2_']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
   preds = preds[['row_id', 'place_id']]
   return preds

def splitData(aRange, train, test):
   #print 'hello'
   X_train = train[features]
   y_train = train[['place_id']]
   X_test = test[features]
   total_iterations = sum(1 for _ in aRange['x']) * sum(1 for _ in aRange['y'])
   print(total_iterations)
   itera = 0
   #df = pd.DataFrame(columns=['row_id', 'place_id'])
   #df.to_csv(filename, mode='a', index=False)
   start_time = time.time()
   for x_min, x_max in aRange['x']:
       start_time_row = time.time()
       for y_min, y_max in aRange['y']: 
           area = dict(x_min=x_min,
                       x_max=x_max,
                       y_min=y_min,
                       y_max=y_max)
           area = updateArea(area, size)
           name = 'x_' + str(x_min) + '_' + str(x_max) + '_y_' + str(y_min) + '_' + str(y_max)
           train_grid = getTestGrid(train, area)
           train_grid.to_csv('chunks/train/' + name + '.csv', index=False)
           test_grid = getTestGrid(test, area)
           test_grid.to_csv('chunks/test/' + name + '.csv', index=False)
   print("Elapsed time for splitting: %s seconds" % (time.time() - start_time))

def trainTestModels(aRange):
   total_iterations = sum(1 for _ in aRange['x']) * sum(1 for _ in aRange['y'])
   print(total_iterations)
   itera = 0
   for x_min, x_max in aRange['x']:
       for y_min, y_max in aRange['y']:
           start_time = time.time() 
           name = 'x_' + str(x_min) + '_' + str(x_max) + '_y_' + str(y_min) + '_' + str(y_max)    
           if not os.path.isfile('chunks/results1/' + name + '.csv'):
              print 'training ' + name
              train_grid = pd.read_csv('chunks/train/' + name + '.csv')
              test_grid = pd.read_csv('chunks/test/' + name + '.csv') 
              X_train_grid = train_grid[features]
              y_train_grid = train_grid[['place_id']].values.ravel()
              label_mapping = dict(zip(list(set(y_train_grid)),range(len(list(set(y_train_grid))))))
              y = np.array([label_mapping[x] for x in y_train_grid])

              print (X_train_grid.shape)
              #print (test_grid.shape)
              #clf = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
              #clf = GradientBoostingClassifier()  
              xg_train = xgb.DMatrix(X_train_grid, label=y)
              #print X_train_grid
              #print train_grid['place_id']
              #print classes
              param = {}
              # use softprob multi-class classification
              param['objective'] = 'multi:softprob'
              param['eta'] = 0.7
              #param['eta'] = 0.8
              param['max_depth'] = 2
              param['silent'] = 1
              param['nthread'] = 4
              param['tree_method'] = 'approx'
              param['num_class'] = len(y)
              #param['max_delta_step'] = 1
              num_round = 2
              print ('training a model')
              bst = xgb.train(param, xg_train, num_round)
              print ('model built')
              #clf.fit(X_train_grid, y_train_grid) 
              print("finished training: %s seconds" % (time.time() - start_time))
              if not test_grid.empty:
                 start_time = time.time() 
                 preds = predictY(test_grid, bst)
                 #preds = predictY(test_grid, clf)
                 print("Finished predicting: %s seconds" % (time.time() - start_time))
                 #preds.to_csv(filename, mode='a', header=False, index=False)
                 preds.to_csv('chunks/results1/' + name + '.csv', mode='a', index=False)
              else: 
                 print('empty dataset')
              itera = itera + 1
              print(itera, " out of ", total_iterations)
           else:
              print ('file ' + name + 'already exists')
           

def createSubmission(aRange):
   #print 'hello'
   total_iterations = sum(1 for _ in aRange['x']) * sum(1 for _ in aRange['y'])
   print(total_iterations)
   itera = 0
   df = pd.DataFrame([], columns =['row_id', 'place_id'])
   print (df)
   for x_min, x_max in aRange['x']:
       start_time_row = time.time()
       for y_min, y_max in aRange['y']: 
           name = 'x_' + str(x_min) + '_' + str(x_max) + '_y_' + str(y_min) + '_' + str(y_max)
           results = pd.read_csv('chunks/results1/' + name + '.csv')   
           df = pd.concat([df, results], axis=0)
           # load it again
       print (df.shape)
       print("Elapsed time row: %s seconds" % (time.time() - start_time_row))
   return df

#TODO first step from 0 to 3
size = dict(x_min=0.0,
            x_max=10.0,
            #y_min=0.0,
            y_min=0.0,
            y_max=10.0)
step = dict(x=0.2, y=0.2)
features = ['x', 'y', 'accuracy', 'time', 'hour', 'weekday', 'month', 'year']

#train = pd.read_csv('input/train.csv')
#test = pd.read_csv('input/test.csv')
#print train.describe()
#print test.describe()
#train = train[0:4000]
#test = test[0:3000]

#print(train.shape)
#print(test.shape)
#train = timeEngineering(train)
#test = timeEngineering(test)
 
start_time = time.time()
aRange = calcRange(size, step)
#print aRange



#splitData(aRange, train, test)
#saveModels(aRange)
trainTestModels(aRange)
#df = createSubmission(aRange)
#mlStep(aRange)
df = pd.read_csv('submission1.csv')
df = df[['row_id', 'place_id']]
df['row_id'] = df['row_id'].astype(int)
df.to_csv('submission.csv', index=False)
print (df.shape)
print("Elapsed time overall: %s seconds" % (time.time() - start_time))
