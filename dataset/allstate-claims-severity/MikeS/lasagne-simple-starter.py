import pandas as pd
import numpy as np
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from sklearn.model_selection import KFold
import theano.tensor as T
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import gc


# constants
shift = 260
nfolds = 10
bags = 10


def mae(a, b):
    
    return T.mean(T.abs_(a - b))


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

cats = [x for x in train.columns if 'cat' in x]
conts = [x for x in train.columns if 'cont' in x]

joined = pd.concat([train, test])

cats = [x for x in train.columns if 'cat' in x]
conts = [x for x in train.columns if 'cont' in x]

for cat in cats:
    joined[cat] = pd.factorize(joined[cat], sort=True)[0]

scaler = StandardScaler()

joined[cats] = scaler.fit_transform(joined[cats])
scaler = StandardScaler()
joined[conts] = scaler.fit_transform(joined[conts])

train = joined[joined['loss'].notnull()]
test = joined[joined['loss'].isnull()]

del joined
gc.collect()

y = np.log(train['loss'] + shift)
ids = test['id']
X = np.asarray(train.drop(['loss', 'id', 'cont6', 'cont11', 'cont9'], 1))
X_test = np.asarray(test.drop(['loss', 'id', 'cont6', 'cont11', 'cont9'], 1))

#converting to float32 as Theano requered
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)
X_test = np.asarray(X_test, dtype=np.float32)


#Kfold
kf = KFold(n_splits=nfolds, random_state=42, shuffle=True)

''' Layer parameters'''
def model():
    l_in = layers.InputLayer(shape=(None, X.shape[1]))
    l1 = layers.batch_norm(layers.DenseLayer(l_in, num_units=200))
    ld1 = layers.DropoutLayer(l1, p=0.2)
    l2 = layers.batch_norm(layers.DenseLayer(ld1, num_units=50))
    ld2 = layers.DropoutLayer(l2, p=0.1)
    l_out = layers.DenseLayer(ld2, num_units=1)

    return layers.get_all_layers(l_out)


pred = 0

i = 0

for tr, val in kf.split(y):
    i += 1
    print('iteration: ', i)    
    X_tr, y_tr = X[tr], y[tr]
    X_val, y_val = X[val], y[val]

    for _ in range(bags):
        net = NeuralNet(layers=model(),
                        update=nesterov_momentum,
                        objective_loss_function = mae,
                        batch_iterator_train=BatchIterator(batch_size=500, shuffle=True),
                        batch_iterator_test=BatchIterator(batch_size=500),
                        regression=True,
                        max_epochs=55,
                        verbose=0,
                        update_learning_rate=0.01,
                        objective_l2 = 0.0025)
        
        net.fit(X_tr, y_tr)
        
        tr_pr = net.predict(X_tr)
        valid = net.predict(X_val)
        print('TRAIN MAE: ', mean_absolute_error(np.exp(y_tr) - shift, np.exp(tr_pr) - shift))
        print('VALIDATION MAE: ', mean_absolute_error(np.exp(y_val) - shift, np.exp(valid) - shift))
        pred += net.predict(X_test)
    
        
    

pred = np.exp(pred/bags/nfolds) - shift
pred = np.reshape(pred, pred.shape[0])
df = pd.DataFrame({'id': ids, 'loss': pred})

df.to_csv('sub_lasagne.csv',index=False)