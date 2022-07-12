from sklearn.preprocessing import LabelEncoder
import math
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import time
from datetime import timedelta
import gc


# Found at: https://www.kaggle.com/rshekhar2/facebook-v-predicting-check-ins/xgboost-cv-example-with-small-bug


def load_data(data_name):
    types = {'row_id': np.dtype(np.int32),
             'x': np.dtype(float),
             'y': np.dtype(float),
             'accuracy': np.dtype(np.int16),
             'place_id': np.int64,
             'time': np.dtype(np.int32)}
    df = pd.read_csv(data_name, dtype=types, index_col=0, na_filter=False)
    return df


def process_one_cell(df_cell_train, df_cell_test, fw, th, n_neighbors):

    #print (str(len(df_cell_train)) + " " + str(len(df_cell_test)))

    # Remove infrequent places
    df_cell_train = remove_infrequent_places(df_cell_train, th)
    
    #print "after removing"
    #print (str(len(df_cell_train)) + " " + str(len(np.unique(df_cell_train['place_id']))))

    # Store row_ids for test
    row_ids = df_cell_test.index

    # Preparing data
    y = df_cell_train.place_id.values
    X = df_cell_train.drop(['place_id'], axis=1).values

    # Applying the classifier
    clf = KNeighborsClassifier(n_neighbors=n_neighbors,
                               weights=calculate_distance, p=1,
                               n_jobs=2, leaf_size=30)
    clf.fit(X, y)
    y_pred = clf.predict_proba(df_cell_test.values)
    y_pred_labels = np.argsort(y_pred, axis=1)[:, :-4:-1]
    pred_labels = clf.classes_[y_pred_labels]
    cell_pred = np.column_stack((row_ids, pred_labels)).astype(np.int64)
    
    return cell_pred


def calculate_distance(distances):
    return 1.0/(np.e**(1+1.5*distances))




def process_grid(df_train, df_test, x_cuts, y_cuts,
                 x_border_aug, y_border_aug, fw, th, n_neighbors):
    print (str(len(df_train)))
    preds_list = []
    x_slice = df_train['x'].max() / x_cuts
    y_slice = df_train['y'].max() / y_cuts


    for i in range(x_cuts):
        row_start_time = time.time()
        x_min = x_slice * i
        x_max = x_slice * (i + 1)
        x_max += int((i + 1) == x_cuts)  # expand edge at end

        mask = (df_test['x'] >= x_min)
        mask = mask & (df_test['x'] < x_max)
        col_test = df_test[mask]
        x_min -= x_border_aug
        x_max += x_border_aug
        mask = (df_train['x'] >= x_min)
        mask = mask & (df_train['x'] < x_max)
        col_train = df_train[mask]

        for j in range(y_cuts):
            y_min = y_slice * j
            y_max = y_slice * (j + 1)
            y_max += int((j + 1) == y_cuts)  # expand edge at end
            mask=(col_test['y']>=y_min)
            mask=mask&(col_test['y']<y_max)
            cell_test = col_test[mask]

            y_min -= y_border_aug
            y_max += y_border_aug
            mask = (col_train['y'] >= y_min)
            mask = mask & (col_train['y'] < y_max)
            cell_train = col_train[mask]
            
            for m in range(0,4):
                m_min=y_min+(y_slice/4)*m
                m_max=y_min+(y_slice/4)*(m+1)
                m_max += int((m + 1) == 4)  # expand edge at end
                mask=(cell_test['y']>=m_min)
                mask=mask&(cell_test['y']<m_max)
                final_test=cell_test[mask]
                
                m_min -= y_border_aug
                m_max += y_border_aug
                mask=(cell_train['y']>=m_min)
                mask=mask&(cell_train['y']<m_max)
                final_train=cell_train[mask]
                cell_pred = process_one_cell(final_train, final_test,
                                         fw, th, n_neighbors)
                preds_list.append(cell_pred)
            #elapsed = (time.time() - row_start_time)
            #print('row', j, 'data prepared in:', timedelta(seconds=elapsed))
            
            
            #fname="./output_cell/cell-"+str(i)+"-"+str(j)
            #np.savetxt(fname, cell_pred.astype('str'), fmt='%s,%s,%s,%s')
        

        elapsed = (time.time() - row_start_time)
        print('Col', i, 'completed in:', timedelta(seconds=elapsed))
    #return preds_list
    preds = np.vstack(preds_list)
    return preds



def generate_submission(preds):
    print('Writing submission file')
    print('Pred shape:', preds.shape)
    with open('KNN_submission.csv', "w") as out:
        out.write("row_id,place_id\n")
        rows = [''] * 8607230
        for pred in preds:
            rows[pred[0]] = '%d,%d %d %d\n' % (pred[0], pred[1], pred[2], pred[3])
        out.writelines(rows)



def remove_infrequent_places(df, th=5):
    place_counts = df.place_id.value_counts()
    mask = (place_counts[df.place_id.values] >= th).values
    df = df.loc[mask]
    return df


def prepare_data(datapath):
    df_train = load_data(datapath + 'train.csv')

    print('Feature engineering on train')
    #df_train = remove_inaccurate(df_train)
    df_train = feature_engineering(df_train)
    df_test = load_data(datapath + 'test.csv')
    print('Feature engineering on test')
    df_test = feature_engineering(df_test)
    print(str(len(df_train))+" "+str(len(df_test)))
    df_train = apply_weights(df_train, fw)
    df_test = apply_weights(df_test, fw)
    return df_train, df_test


def apply_weights(df, fw):
    df['accuracy'] *= fw[0]
    df['day_of_year_sin'] *= fw[1]
    df['day_of_year_cos'] *= fw[1] *1.6
    df['minute_sin'] *= fw[2] *1.1
    df['minute_cos'] *= fw[2]
    df['weekday_sin'] *= fw[3] *1.1
    df['weekday_cos'] *= fw[3]
    df.x *= fw[4]
    df.y *= fw[5]
    df['year'] *= fw[6]
    return df


def feature_engineering(df):
    minute= (df['time'] % (24*60))*1.0/(24*60) * 2 * np.pi
    df['minute_sin'] = (np.sin(minute) + 1).round(4)
    df['minute_cos'] = (np.cos(minute) + 1).round(4)
    del minute
    day = 2 * np.pi * ((df['time'] // 1440) % 365) / 365
    df['day_of_year_sin'] = (np.sin(day) + 1).round(4)
    df['day_of_year_cos'] = (np.cos(day) + 1).round(4)
    del day
    weekday = 2 * np.pi * ((df['time'] // 1440) % 7) / 7
    df['weekday_sin'] = (np.sin(weekday) + 1).round(4)
    df['weekday_cos'] = (np.cos(weekday) + 1).round(4)
    del weekday
    df['year'] = (((df['time']) // 525600))
    df.drop(['time'], axis=1, inplace=True)
    df['accuracy'] = np.log10(df['accuracy'])
    return df

print('Starting...')
start_time = time.time()
# Global variables
datapath = '../input/'
th = 10  # Threshold at which to cut places from train
fw = [0.6, 0.32935, 0.56515, 0.2670, 22, 52, 0.51785]  

# Defining the size of the grid
x_cuts = 20  # number of cuts along x
y_cuts = 20  # number of cuts along y
x_border_aug = 0.02 * fw[4]   #expansion of x border on train
y_border_aug = 0.01 * fw[5]  #expansion of y border on train
n_neighbors =30

df_train, df_test = prepare_data(datapath)
gc.collect()

elapsed = (time.time() - start_time)
print('Data prepared in:', timedelta(seconds=elapsed))


preds = process_grid(df_train[9000000:], df_test, x_cuts, y_cuts,
                     x_border_aug, y_border_aug,
                     fw, th, n_neighbors)
elapsed = (time.time() - start_time)
print('Predictions made in:', timedelta(seconds=elapsed))

# del df_train, df_test
generate_submission(preds)
elapsed = (time.time() - start_time)
print('Task completed in:', timedelta(seconds=elapsed))