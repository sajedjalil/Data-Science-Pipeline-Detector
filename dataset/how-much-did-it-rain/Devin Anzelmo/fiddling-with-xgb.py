import pandas as pd
import numpy as np
#import sys
#sys.path.append('xgboost-master/wrapper/')
import xgboost as xgb
import zipfile


#counts the number of radar scans in the row by using white spaces
def get_num_radar_scans(timetoend_row):
    return timetoend_row.count(' ') + 1

#takes the mean of all the scan values in one row.
def mean_of_row(row):
    return np.mean(list(map(np.double, row.split(' '))))

def make_cdf_step(true_label_value):
    step_cdf = np.ones(70)
    step_cdf[0:true_label_value] = 0
    return step_cdf

def make_cdf_distribution(in_class_labels):
    pdf = in_class_labels.value_counts()/float(len(in_class_labels))
    pdf = pdf.sort_index()
    cdf = np.zeros(70)
    for e,i in enumerate(pdf.index.values.tolist()):
        cdf[i] = pdf.iloc[e]
    return cdf.cumsum()

#return a list of cumalative distributions. 
def make_cdf_list(first_agg, num_lab, new_lab, actual_labels,offset):
    cdfs = []
    for i in range(num_lab):
        if i < first_agg: 
            cdfs.append(make_cdf_step(i))
        else:
            cdfs.append(make_cdf_distribution(actual_labels.reindex(new_lab.iloc[offset:][new_lab.iloc[offset:]==i].index)))
    return cdfs


def create_full_predictions(CDFs, predictions):
    data_length = len(predictions)
    for e,i in enumerate(CDFs):
        if e == 0:
            temp = predictions.iloc[:,0].values.reshape(data_length,1)*CDFs[0].reshape(1,len(CDFs[0]))
        else:
            temp += predictions.iloc[:,e].values.reshape(data_length,1)*CDFs[e].reshape(1,len(CDFs[e]))
    return temp

def aggregate_labels(label_list, ceiled_labels):
    new_lab = ceiled_labels.replace(label_list[0][0],label_list[0][1])    
    for i in range(1,len(label_list)):
        new_lab = new_lab.replace(label_list[i][0],label_list[i][1])
    return new_lab

#this is faster with many features, but didn't end up improving score so its not used in final solution.
def train_linear_xgb(data,lmbda,alpha, lmbda_bias, num_classes,num_threads, num_rounds,early_stop=3):
    xg_train = xgb.DMatrix(data[0].values,label=data[1].values.ravel(),missing=np.nan)
    xg_val = xgb.DMatrix(data[2].values,label=data[3].values.ravel(),missing=np.nan)
    param1 = {}
    param1['objective'] = 'multi:softprob'
    param1['lambda'] = lmbda
    param1['alpha'] = alpha
    param1['lambda_bias'] = lmbda_bias
    param1['silent'] = 1
    param1['nthread'] = num_threads
    param1['num_class'] = num_classes
    param1['eval_metric'] = 'mlogloss'

    watchlist = [(xg_train, 'train'),(xg_val, 'test')]
    #num_rounds = 10000
    bst1 = xgb.train(param1,xg_train, num_rounds, evals=watchlist ,early_stopping_rounds=early_stop) #early stopping not working python3.4
    return bst1

# make predictions with a xgboost model on some data. 
def predict_bst(bst,validation):
    xg_val = xgb.DMatrix(validation.values,missing=np.nan)
    pred= bst.predict(xg_val);
    pred = pd.DataFrame(pred,index=validation.index)
    return pred


#a fast CRPS calculation with no checks for the validaty of the of input
#adapted from code by Alexander Guschin, as posted on the forums
def calc_crps(thresholds,predictions, actuals):
    obscdf = (thresholds.reshape(70,1) >= actuals).T
    crps = np.mean(np.mean((predictions - obscdf) ** 2))
    return crps


#read in just the data we need for the plot. 
z = zipfile.ZipFile('../input/train_2013.csv.zip')
train = pd.read_csv(z.open('train_2013.csv'),usecols=['Expected','Reflectivity'])

train['num_scans'] = train.Reflectivity.apply(get_num_radar_scans)
train = train.query('num_scans > 17')

train['mean_reflectivity'] =  train.Reflectivity.apply(mean_of_row)
labels = train.Expected

train.drop(['Reflectivity','Expected'], axis=1,inplace=True)

integer_labels = np.ceil(labels)
integer_labels = integer_labels[integer_labels < 70]

reduced_labels = aggregate_labels([[range(8,10),8],[range(10,14),9],[range(14,19),10],[range(19,70),11]], pd.DataFrame(integer_labels)).iloc[:,0] 
train = train.reindex(integer_labels.index)
cutoff_value = int(len(train)*.5)

#need this to make it work, only training on mean_reflectivity for some reason.
train = pd.concat([train.mean_reflectivity,train.num_scans],axis=1)

data = (train.iloc[cutoff_value:,:], reduced_labels.iloc[cutoff_value:], train.iloc[:cutoff_value,:], reduced_labels.iloc[:cutoff_value])

#The parameters for the wrapper function
#train_linear_xgb(data,lmbda,alpha, lmbda_bias, num_classes,num_threads, num_rounds)
bst = train_linear_xgb(data,5,5,2,12,1,num_rounds=56,early_stop=2)

preds =  predict_bst(bst, train.iloc[:cutoff_value])
cdfs_tst = make_cdf_list(8, 12, reduced_labels, integer_labels, 0)
preds_full = create_full_predictions(cdfs_tst, preds)

labels = labels.reindex(integer_labels.index)

print('my best score with 50% validation for this subset was 0.014, with a 430 features')
print('CRPS using Reflectivity =', calc_crps(np.arange(70),preds_full, labels.iloc[:cutoff_value].values))

