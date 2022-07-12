# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, scale
from sklearn import svm, linear_model, model_selection
from sklearn.ensemble import GradientBoostingClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
fw = [1, 1, 10./23., 10./6., 10./29., 10./11., 10, 10./1033.]
fw = [1, 1, 1, 1, 1, 1, 1, 1]
#fw = [500, 1000, 4, 3, 1./22., 2, 10]
def time_preprocess(df):
    df.x = df.x.values * fw[0]
    df.y = df.y.values * fw[1]
    df['hour'] = ((df['time']//60)%24) * fw[2]
    df['weekday'] = ((df['time']//(60*24))%7) * fw[3]
    df['monthday'] = ((df['time']//(60*24))%30) * fw[4]
    df['month'] = ((df['time']//(60*24*30))%12) * fw[5]
    df['year'] = (df['time']//(60*24*30*12)) * fw[6]
    #df['accuracy'] = df['accuracy'] * fw[7]
    df = df.drop(['time', 'accuracy'], axis = 1)
    
    return df
    
def process_one_cell(df_train, max_x, min_x, max_y, min_y):
    extra = 0.02
    df_train_cell = df_train[(df_train['x'] >= min_x - extra) & (df_train['x'] <= max_x + extra) & (df_train['y'] >= min_y - extra) & (df_train['y'] <= max_y + extra)]
    #df_test_cell = df_test[(df_test['x'] >= min_x) & (df_test['x'] <= max_x) & (df_test['y'] >= min_y) & (df_test['y'] <= max_y)]
    #df_train_cell.info()
    th = 5
    place_counts = df_train_cell.place_id.value_counts()
    mask = (place_counts[df_train_cell.place_id.values] >= th).values
    df_train_cell = df_train_cell.loc[mask]
    
    #row_id = df_test_cell.index
    y = df_train_cell['place_id']
    le = LabelEncoder()
    y = le.fit_transform(y)
    X = df_train_cell.drop(['place_id'], axis=1)
    #X_test = df_test_cell
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = KNeighborsClassifier(n_neighbors = 30, weights='distance', metric='manhattan', n_jobs=4)

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    #y_test = clf.predict_proba(X_test)
    #y_test = le.inverse_transform(np.argsort(y_test, axis=1)[:,::-1][:,:3])
    return accuracy

def grid_process(df_train, total_num_grid_x, total_num_grid_y):
    #res = np.zeros((df_test.shape[0], 3), dtype=np.int64)
    delta_x = fw[0]*10/total_num_grid_x
    delta_y = fw[1]*10/total_num_grid_y
    res = 0
    for num_grid_x in range(total_num_grid_x):
        for num_grid_y in range(total_num_grid_y):
            print("x:" + str(num_grid_x) + "y:" + str(num_grid_y))
            min_x = num_grid_x * delta_x
            max_x = min_x + delta_x
            min_y = num_grid_y * delta_y
            max_y = min_y + delta_y
            
            res = res + process_one_cell(df_train, max_x, min_x, max_y, min_y)
            

    print('finished predict')
    print(res /total_num_grid_x/total_num_grid_y)
#    df_aux = pd.DataFrame(res, dtype=str, columns=['l1', 'l2', 'l3'])  
    
    #Concatenating the 3 predictions for each sample
#    ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3], sep=' ')
    
    #Writting to csv
#    ds_sub.name = 'place_id'
#    ds_sub.to_csv('sub_knn.csv', index=True, header=True, index_label='row_id')    
    
    
df_train = pd.read_csv('../input/train.csv',
                           usecols=['row_id','x','y','time', 'accuracy', 'place_id'], 
                           index_col = 0)
#df_test = pd.read_csv('../input/test.csv',
#                          usecols=['row_id','x','y','time', 'accuracy'],
#                          index_col = 0)
#df_train.info()
print('finished load data')
print(df_train.describe())
df_train = time_preprocess(df_train)
print(df_train.describe())
#df_test = time_preprocess(df_test)
print('finished preprocessing')
total_num_grid_x = 20
total_num_grid_y = 40
grid_process(df_train, total_num_grid_x, total_num_grid_y)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#clf = KNeighborsClassifier(n_neighbors = 5)
#clf = linear_model.SGDClassifier()
#clf = GradientBoostingClassifier()
#clf.fit(X_train, y_train) 
#print('finished fit')
#accuracy = clf.score(X_test, y_test)
#print(accuracy)

# Any results you write to the current directory are saved as output.