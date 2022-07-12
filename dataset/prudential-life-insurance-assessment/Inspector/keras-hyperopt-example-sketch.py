
#keras hyperopt tuning experiment

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split,KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from keras.callbacks import Callback

import ml_metrics

from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.mongoexp import MongoTrials

"""
Load and process (one-hot) both training and test together
Not currently scoring the test set

"""
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

train_rows=train.shape[0]

y=train.Response

train.drop("Id",axis=1, inplace=True)
test.drop("Id",axis=1, inplace=True)
train.drop("Response",axis=1, inplace=True)

combined=pd.concat([train,test],axis=0)


catCols=[
         'Product_Info_1',
         'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 'Employment_Info_2',
         'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4',
         'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7',
         'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4',
         'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1',
         'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5',
         'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9',
         'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14',
         'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20',
         'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26',
         'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31',
         'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36', 'Medical_History_37',
         'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41']

pdf=pd.get_dummies(combined[catCols].astype(object))

listAll=combined.columns.values.tolist()

numCols=[x for x in listAll if x not in catCols]
X=pd.concat([pdf,combined[numCols]],axis=1)
X.fillna(X.mean(),inplace=True)

X = StandardScaler().fit_transform(X)
train_X=X[0:train_rows,:]
test_X=X[train_rows:,:]



#hold out data from training to use for hyperopt
X_train, X_test, y_train, y_test = train_test_split(train_X, y, test_size=0.20)

#sample out another parition to use for early stopping 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20)



"""
This class is used as a call back with Keras
It will use kappa on the validation data in order to use early stopping,
returning the best model

filepath is where to save the hdf5 table in order to persist the best model

"""
class clsvalidation_kappa(Callback):  #inherits from Callback
    
    def __init__(self, filepath, validation_data=(), patience=5):
        super(Callback, self).__init__()

        self.patience = patience
        self.X_val, self.y_val = validation_data  #tuple of validation X and y
        self.best = 0.0
        self.wait = 0  #counter for patience
        self.filepath=filepath
        self.best_rounds =1
        self.counter=0

    def on_epoch_end(self, epoch, logs={}):
        
        self.counter +=1
        p = self.model.predict(self.X_val, verbose=0) #score the validation data 
        
             
        #current kappa
        current = ml_metrics.quadratic_weighted_kappa(self.y_val.values.ravel(),np.clip(np.round(p.astype(int).ravel()), 1, 8))
       
        print('Epoch %d Kappa: %f | Best Kappa: %f \n' % (epoch,current,self.best))
    
    
        #if improvement over best....
        if current > self.best:
            self.best = current
            self.best_rounds=self.counter
            self.wait = 0
            self.model.save_weights(self.filepath, overwrite=True)
        else:
            if self.wait >= self.patience: #no more patience, retrieve best model
                self.model.stop_training = True
                print('Best number of rounds: %d \nKappa: %f \n' % (self.best_rounds, self.best))
                
                self.model.load_weights(self.filepath)
                           
            self.wait += 1 #incremental the number of times without improvement
        
        


#This is the parameter space to explore with hyperopt

#Simply offers several discrete choices for numnber of hidden units and drop out rates for
#a 2 or 3 layer MLP and also batch size

#This can be expanded over other parameters and to sample from a distribution instead of a discrete choice
#for # of units etc.



space = {'choice':


hp.choice('num_layers',
    [
                    {'layers':'two',
                     
                                                    
                    },
        
                     {'layers':'three',
                      
                      
                      'units3': hp.choice('units3', [64, 128, 256, 512]),
                      'dropout3': hp.choice('dropout3', [0.25,0.5,0.75])
                                
                    }
        
    
    ]),
    
    'units1': hp.choice('units1', [64, 128, 256, 512]),
    'units2': hp.choice('units2', [64, 128, 256, 512]),
                 
    'dropout1': hp.choice('dropout1', [0.25,0.5,0.75]),
    'dropout2': hp.choice('dropout2', [0.25,0.5,0.75]),
    
    'batch_size' : hp.choice('batch_size', [28,64,128]),

    'nb_epochs' :  100,
    'optimizer': 'adadelta',
    'activation': 'relu'
    
    
    }


#Objective function that hyperopt will minimize


def objective(params):
    
    import ml_metrics
    
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta
    from keras.layers.normalization import BatchNormalization
    from keras.callbacks import Callback
    
    print ('Params testing: ', params)
    print ('\n ')
    model = Sequential()
    model.add(Dense(output_dim=params['units1'], input_dim = X_train.shape[1], init = "glorot_uniform")) 
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))
    model.add(BatchNormalization())
    
    model.add(Dense(output_dim=params['units2'], init = "glorot_uniform")) 
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout2']))
    model.add(BatchNormalization())
    
    if params['choice']['layers']== 'three':
        model.add(Dense(output_dim=params['choice']['units3'], init = "glorot_uniform")) 
        model.add(Activation(params['activation']))
        model.add(Dropout(params['choice']['dropout3']))
        model.add(BatchNormalization())
        patience=25
    else:
        patience=15
    
     
    model.add(Dense(1, init = "glorot_uniform"))    #End in a single output node for regression style output
    model.compile(loss='rmse', optimizer=params['optimizer'])
    
    
    #object of class for call back early stopping 
    val_call = clsvalidation_kappa(validation_data=(X_val, y_val), patience=patience, filepath='"../input/best.h5') #instantiate object

    #includes the call back object
    model.fit(X_train, y_train, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'], verbose = 0,callbacks=[val_call])
     
    #predict the test set 
    preds=model.predict(X_test, batch_size = 5000, verbose = 0)
    
    
    predClipped = np.clip(np.round(preds.astype(int).ravel()), 1, 8) #simple rounding of predictionto int
    score=ml_metrics.quadratic_weighted_kappa(y_test.values.ravel(),predClipped)
 
    return {'loss': score*(-1), 'status': STATUS_OK, 'rounds':val_call.best_rounds }


trials = Trials()

best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=100)

print (best)
print (trials.best_trial)
