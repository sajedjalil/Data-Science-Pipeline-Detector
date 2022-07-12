# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

__author__ = 'Abhi'

'''Contains additions from several excellent scripts and ideas posted on the forum

The script does the following:
- calculates time features for test and train
- calculates knn using a grid. Top 10 probabilities are calculated
- calculates probability lookup tables for main features like hour etc
- multiplies knn probabilities with probabilities from lookup tables to give total probability
- selects top 3 placeids based on total probability and generates submission file
- cross-validation is included (for all grid cells as well as one grid cell)
- takes 30 min  to run and produces a score of 0.5865 lb
'''


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import time


## Parameters
cross_validation = 0 # 1 = cross validation, 0 = test data submission
onecell=0 # 0 = all grid cells, 1 = one grid cell specified by grid_onecell
grid_onecell = 200


n_cell_x=20 
n_cell_y=40
th = 8 ## threshold events below which placeids are excluded from train data


########### set weights for knn

## I have used a weight matrix below. Weights for each grid can be customized (not included in this script)
# 0-x,1-y,2-hour,3-weekday,4-month, 5-year, 6 - accuracy, 7-nearestneighbors 
weights = np.tile(np.array([490.0, 980.0, 4.0, 3.1, 2.1, 10.0, 10.0, 36])[:,None],n_cell_x*n_cell_y).T #feature weights

def calcgridwisemap3(group): 
    score = ([1/1.0, 1/2.0, 1/3.0]*(np.asarray(group[['ytest']]) == np.asarray(group[['id1','id2','id3']])) ).sum()/group.shape[0]
    return score
    
def makeprobtable(train, feature, threshold):
    table = train.groupby('place_id')[feature].value_counts()
    table = table/train.groupby('place_id')[feature].count()
    table = table.reset_index(level=0, drop=True)  #drop placeid index
    table[0]=0 # all missing indices have zero probability
    table[table < threshold]= threshold  #threshold small probabilities including zeros to threshold value
    return table

    
def getprob(ind, table, nn):
    split = len(ind)*nn//3 # split array operations for memory management
    temp = ind.reshape(-1)
    temp1=np.invert(np.in1d(temp[:split], table.index.values))
    temp2=np.invert(np.in1d(temp[split:split*2], table.index.values))
    temp3=np.invert(np.in1d(temp[split*2:], table.index.values))
    temp[np.concatenate((temp1,temp2,temp3))] = 0 # find indices that are not in lookup and set to zero
    temp1=table[temp[:split]]
    temp2=table[temp[split:split*2]]
    temp3=table[temp[split*2:]]
    prob = np.concatenate((temp1,temp2,temp3))
    prob = prob.reshape(-1,nn)
    return prob

def extendgrid(extension_x, extension_y, size_x, size_y, n_cell_x, n_cell_y):
    xmin =np.linspace(0,10-size_x,n_cell_x)    
    xmax =np.linspace(0+size_x,10,n_cell_x)    
    ymin =np.linspace(0,10-size_y,n_cell_y)    
    ymax =np.linspace(0+size_y,10,n_cell_y)    
    
    grid1 = np.tile(xmin,n_cell_y)
    grid2 = np.tile(xmax,n_cell_y)
    grid3 = np.repeat(ymin,n_cell_x)
    grid4 = np.repeat(ymax,n_cell_x)
    grid1 = grid1 - extension_x
    grid2 = grid2 + extension_x
    grid3 = grid3 - extension_y
    grid4 = grid4 + extension_y
    
    grid = np.vstack((grid1,grid2,grid3,grid4)).T
    return grid


def calculate_distance(distances):
    return distances ** -2


    
################## read data #####################

train = pd.read_csv('../input/train.csv',dtype={'place_id': np.int64}, index_col = 0) 


train['hour'] = ( (train['time']+120)/60)%24+1 
train['weekday'] = (train['time']/1440)%7+1 
train['month'] = ( train['time'] /43800)%12+1 
train['year'] = (train['time']/525600)+1 
train['four_hour'] = (train['time']/240)%6+1
train['acc'] = np.log10(train['accuracy'])

pd.options.mode.chained_assignment = None
add_data = train[train.hour<2.5]# add data for periodic time that hit the boundary
add_data.hour = add_data.hour+24
add_data2 = train[train.hour>22.5]
add_data2.hour = add_data2.hour-24
train = train.append(add_data)
train = train.append(add_data2)
del add_data,add_data2


if cross_validation == 1:
    print('Loading cross validation data ...')
    
    test = train.query('month >=5.0 and year >=2.0')  
    train = train.query('~(month >=5.0 and year >=2.0)')
    ytrain = train['place_id']
    test = test.query('place_id in @ytrain')
    ytest = test['place_id']
    del test['place_id']
    test.reset_index(inplace=True) 
    test['row_id'] = test.index.values                     
    test.set_index('row_id',inplace=True)
    
    
else:    
    print('Loading data ...')
    test = pd.read_csv('../input/test.csv', index_col = 0)
  
    test['hour'] = ((test['time']+120)/60)%24+1 
    test['weekday'] = (test['time']/1440)%7+1 
    test['month'] = (test['time']/43800)%12+1 
    test['year'] = (test['time']/525600)+1 
    test['four_hour'] = (test['time']/240)%6+1
    test['acc'] = np.log10(test['accuracy']) 
    
################## process data #####################

#Make grid
size_x = 10. / n_cell_x
size_y = 10. / n_cell_y
    
eps = 0.00001  
xs = np.where(train.x.values < eps, 0, train.x.values - eps)
ys = np.where(train.y.values < eps, 0, train.y.values - eps)
pos_x = (xs / size_x).astype(np.int)
pos_y = (ys / size_y).astype(np.int)
train['grid_cell'] = pos_y * n_cell_x + pos_x

xs = np.where(test.x.values < eps, 0, test.x.values - eps)
ys = np.where(test.y.values < eps, 0, test.y.values - eps)
pos_x = (xs / size_x).astype(np.int)
pos_y = (ys / size_y).astype(np.int)
test['grid_cell'] = pos_y * n_cell_x + pos_x


### extend grid for train data to avoid edge effects
extension_x = 0.03
extension_y = 0.015
    
grid = extendgrid(extension_x, extension_y, size_x, size_y, n_cell_x, n_cell_y)


######## run knn on all grids ##############

indices = np.zeros((test.shape[0], 10), dtype=np.int64)
knn_prob = np.zeros((test.shape[0], 10), dtype=np.float64)

tr = train[['x','y']]

t=time.time()



if onecell==0:
    repeats = range(n_cell_x*n_cell_y)
else:
    repeats = [grid_onecell]



for g_id in repeats:
    if g_id % 100 == 0:
        print('iter: %s' %(g_id))
        print (time.time()-t)/60
    
    #Applying classifier to one grid cell
    xmin, xmax, ymin, ymax =grid[g_id]   
    grid_train = train[(tr.x > xmin) & (tr.x < xmax) & (tr.y > ymin) & (tr.y < ymax)]    

    place_counts = grid_train.place_id.value_counts()
    mask = (place_counts[grid_train.place_id.values] >= th).values
    grid_train = grid_train.loc[mask]

    grid_test = test.loc[test.grid_cell == g_id]
    row_ids = grid_test.index
       
    #Preparing data
    le = LabelEncoder()
    y = le.fit_transform(grid_train.place_id.values)
    X = grid_train[['x', 'y', 'hour', 'weekday', 'month', 'year', 'acc']].values * weights[g_id][:7]
    X_test = grid_test[['x', 'y', 'hour', 'weekday', 'month', 'year', 'acc']].values * weights[g_id][:7]
    
    ###Applying the knn classifier
    #nearest = (weights[g_id][7]).copy().astype(int)
    nearest = np.floor(np.sqrt(y.size)/5.1282).astype(int)
    clf = KNeighborsClassifier(n_neighbors=nearest, weights=calculate_distance,
                               metric='cityblock')

    clf.fit(X, y)
    y_pred = clf.predict_proba(X_test)
    
    indices[row_ids] = le.inverse_transform(  np.argsort(y_pred, axis=1)[:,::-1][:,:10]  )  
    knn_prob[row_ids] = np.sort(y_pred, axis=1)[:,::-1][:,:10]

print ('knn calculations complete')

### create indices for probability lookup tables. For example:
### for placeid = 999999999 and weekday = 5, then the index is wkday_ind = 99999999905
train['wkday_ind'] = 10*train['place_id']+np.floor(train['weekday']).astype(np.int64)   
train['hr_ind'] = 100*train['place_id']+np.floor(train['hour']).astype(np.int64)
train['four_hour_ind'] = 100*train['place_id']+np.floor(train['four_hour']).astype(np.int64)


weekday = makeprobtable(train, 'wkday_ind', 0.001)
hour = makeprobtable(train, 'hr_ind', 0.001)
four_hour = makeprobtable(train, 'four_hour_ind', 0.001)

   
nn=10

wkday_indices=10*indices+np.tile(np.floor(test.weekday[:,None]).astype(np.int64),nn )
hr_indices=100*indices+np.tile(np.floor(test.hour[:,None]).astype(np.int64),nn )
four_hour_indices=100*indices+np.tile(np.floor(test.four_hour[:,None]).astype(np.int64),nn )

weekday_prob = getprob(wkday_indices, weekday, nn)
hour_prob = getprob(hr_indices, hour, nn)
four_hour_prob = getprob(four_hour_indices, four_hour, nn)


total_prob = np.log10(four_hour_prob)*0.1 \
                + np.log10(knn_prob)*1 \
                + np.log10(hour_prob)*0.1 \
                + np.log10(weekday_prob)*0.4


total_prob_sorted = np.sort(total_prob)[:,::-1] 
max3index = np.argsort(-total_prob)
a = np.indices(max3index.shape)[0]
max3placeids = indices[a,max3index]

   
if cross_validation==1: 
    ## calculation assumes unique values  
    print ('indices:',([1/1.0, 1/2.0, 1/3.0]*(ytest[:,None] == indices[:,0:3]) ).sum()/indices[np.nonzero(indices[:,0])].shape[0])
    print ('map3', ([1/1.0, 1/2.0, 1/3.0]*(ytest[:,None] == max3placeids[:,0:3]) ).sum()/max3placeids[np.nonzero(max3placeids[:,0])].shape[0])
 
    ## calculate map3 for each grid 
    max3placeids1 = pd.DataFrame({'row_id':test.index.values, 'grid_cell': test['grid_cell'], 'ytest': ytest.values, 'id1':max3placeids[:,0],'id2':max3placeids[:,1],'id3':max3placeids[:,2]} )                  
    gridwisemap3 = max3placeids1.groupby('grid_cell').apply(calcgridwisemap3)
 
else:    
    print ('writing submission file...')
    max3placeids = pd.DataFrame({'row_id':test.index.values,'id1':max3placeids[:,0],'id2':max3placeids[:,1],'id3':max3placeids[:,2]} )
    max3placeids['place_id']=max3placeids.id1.astype(str).str.cat([max3placeids.id2.astype(str),max3placeids.id3.astype(str)], sep = ' ')       
    max3placeids[['row_id','place_id']].to_csv('submission.csv', header=True, index=False )
    print ('End of program')
