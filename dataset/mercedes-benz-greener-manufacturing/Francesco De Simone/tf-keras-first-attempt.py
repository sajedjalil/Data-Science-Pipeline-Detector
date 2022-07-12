import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
# neural network model
# Keras is a deep learning library that wraps the efficient numerical libraries Theano and TensorFlow.
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
# To evaluate models using cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# To perform data preparation in order to improve skill with Keras models
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Ensemble
from sklearn.ensemble import RandomForestRegressor
# Feature Selection
from sklearn.feature_selection import SelectFromModel

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        
for c in test.columns:
    if test[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(test[c].values)) 
        test[c] = lbl.transform(list(test[c].values))
        
targets = train['y']
train_noTarget = train.drop(['ID','y'], axis=1)
# Tree-based estimators can be used to compute feature importances,
# which in turn can be used to discard irrelevant features

clf = RandomForestRegressor(n_estimators=100, max_features='log2')
clf = clf.fit(train_noTarget, targets)

features = pd.DataFrame()
features['feature'] = train_noTarget.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)


model = SelectFromModel(clf, prefit=True)
# no DataFrame
train_reduced = model.transform(train_noTarget)
id_test = test['ID']
test_noID = test.drop("ID", axis=1).copy()
# no DataFrame
test_reduced = model.transform(test_noID)



# define base model
def baseline_model():
	# create model
    # Topology 13 inputs -> [13 -> 6] -> 1 output
	model = Sequential()
	model.add(Dense(376, input_dim=376, kernel_initializer='normal', activation='relu'))
	model.add(Dense(36, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
	
	
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=150, batch_size=5, verbose=1)

# Numpy representation of NDFrame
dataset = train.values
# split into input (X) and output (Y) variables
X = dataset[:,2:378]
Y = dataset[:,1]

datasetTest = test_noID.values
# split into input (X) and output (Y) variables
X_test = datasetTest[:,0:376]

estimator.fit(X,Y)
res = clf.predict(X_test)

output = pd.DataFrame({'id': id_test, 'y': res})
output.to_csv('preds.csv', index=False)