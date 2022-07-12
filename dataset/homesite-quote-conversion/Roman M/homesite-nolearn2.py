
import matplotlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import log_loss, auc, roc_auc_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from nolearn.lasagne import NeuralNet
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.updates import adagrad, nesterov_momentum
from lasagne.nonlinearities import softmax
from lasagne.objectives import binary_crossentropy, binary_accuracy
import theano

# ### Preprocessing Homesite Data
print("Preprocessing Homesite Data")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')
y = train.QuoteConversion_Flag.values
encoder = LabelEncoder()
y = encoder.fit_transform(y).astype(np.int32)


train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
test = test.drop('QuoteNumber', axis=1)

print("Lets take out some dates")

# Lets take out some dates
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train = train.drop('Original_Quote_Date', axis=1)
train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek
train = train.drop('Date', axis=1)

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)
test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['weekday'] = test['Date'].dt.dayofweek
test = test.drop('Date', axis=1)

# we fill the NA's and encode categories
train = train.fillna(-1)
test = test.fillna(-1)

for f in train.columns:
    if train[f].dtype=='object':
        # print(f)
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values)+  list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))
        print(f)


# ### Get the data in shape for Lasagne
# Now we prep the data for a neural net
print("Now we prep the data for a neural net")
X = train
num_classes = len(encoder.classes_)
num_features = X.shape[1]

X_test = test
num_classes = len(encoder.classes_)
num_features = X_test.shape[1]

# Convert to np.array to make lasagne happy
X = np.array(X)
X = X.astype(np.float32)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_test = np.array(X_test)
X_test = X_test.astype(np.float32)

X_test= scaler.transform(X_test)

# Take the first 200K to train, rest to validate
#split = 200000 
epochs = 12
#val_auc = np.zeros(epochs)
#val_auc_all = np.zeros(epochs)
print(epochs)
# ### Train the Neural Net on the train set
# 
# Comment out second layer for run time.
layers = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('output', DenseLayer)
           ]
           
net1 = NeuralNet(layers=layers,
                 input_shape=(None, num_features),
                 dense0_num_units=150, # 512, - reduce num units to make faster
                 dropout0_p=0.4,
                 dense1_num_units=25,
                 dropout1_p=0.4,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 update=adagrad,
                 update_learning_rate=0.1,
                 eval_size=0.2,
              
                 # objective_loss_function = binary_accuracy,
                 verbose=1,
                 max_epochs=epochs)
net1.fit(X, y)
#for i in range(epochs):
     #net1.fit(X[:split], y[:split])
     #pred = net1.predict_proba(X[split:])[:,1]
     #pred_all= net1.predict_proba(X)[:,1]
     #val_auc[i]     = roc_auc_score(y[split:],pred)
     #val_auc_all[i] = roc_auc_score(y,pred_all)
    #val_auc     = roc_auc_score(y[split:],pred)
    #val_auc_all = roc_auc_score(y,pred_all)    
pred = net1.predict_proba(X)[:,1]
print(roc_auc_score(y,pred))

pred = net1.predict_proba(X_test)[:,1]

sample.QuoteConversion_Flag = pred
sample.to_csv('Lessage.csv', index=False)