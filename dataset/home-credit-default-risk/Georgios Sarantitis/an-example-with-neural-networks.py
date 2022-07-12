import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import regularizers
import matplotlib.pyplot as plt

print('Importing data...')
data = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
prev = pd.read_csv('../input/previous_application.csv')
buro = pd.read_csv('../input/bureau.csv')
NN_submission = pd.read_csv('sample_submission.csv')

#Separate target variable
y = data['TARGET']
del data['TARGET']

#One-hot encoding of categorical features in data and test sets
categorical_features = [col for col in data.columns if data[col].dtype == 'object']

one_hot_df = pd.concat([data,test])
one_hot_df = pd.get_dummies(one_hot_df, columns=categorical_features)

data = one_hot_df.iloc[:data.shape[0],:]
test = one_hot_df.iloc[data.shape[0]:,]

#One-hot encoding of categorical features in previous data set
prev_cat_features = [pcol for pcol in prev.columns if prev[pcol].dtype == 'object']

prev = pd.get_dummies(prev, columns=prev_cat_features)

#Do weird stuff vol1
print('doing weird stuf...')
avg_prev = prev.groupby('SK_ID_CURR').mean()
cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
avg_prev['nb_app'] = cnt_prev['SK_ID_PREV']
del avg_prev['SK_ID_PREV']

#One-hot encoding of categorical features in buro data set
buro_cat_features = [bcol for bcol in buro.columns if buro[bcol].dtype == 'object']

buro = pd.get_dummies(buro, columns=buro_cat_features)

#Do weird stuff vol2
print('doing weird stuf vol 2...')
avg_buro = buro.groupby('SK_ID_CURR').mean()
avg_buro['buro_count'] = buro[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
del avg_buro['SK_ID_BUREAU']

#Join data bases
data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

#Remove features with many missing values
print('Removing features with more than 80% missing')
test = test[test.columns[data.isnull().mean() < 0.85]]
data = data[data.columns[data.isnull().mean() < 0.85]]

#Impute missing
print('Imputing data...')
data = data.fillna(value = 0)
test = test.fillna(value = 0)

#Delete customer Id
del data['SK_ID_CURR']
del test['SK_ID_CURR']

#Create new, balanced train set
data['y'] = y
data_all_ones = data[data.y==1]
data_all_zeros = data[data.y==0]
data_all_zeros2 = data_all_zeros.iloc[0:data_all_ones.shape[0],:]

data = pd.concat([data_all_ones,data_all_zeros2], axis = 0)

y= data.y
del data['y']

#Delete low variance features
print('deleting low variance features...')
all_data = pd.concat([data,test], axis = 0)
selector = VarianceThreshold(0.05)
all_data = selector.fit_transform(all_data)

data = all_data[:data.shape[0],:]
test = all_data[data.shape[0]:,:]

#Scale data to feed Neural Net
print('scaling...')
scaler = MinMaxScaler().fit(data)
data = scaler.transform(data)
test= scaler.transform(test)

#Create train and validation set
train_x, valid_x, train_y, valid_y = train_test_split(data, y, test_size=0.15, shuffle=True)

#-----------Build Neural Network model for evaluation----------
print('Building Neural Network model...')
model = Sequential()
model.add(Dense(96, input_dim=train_x.shape[1],
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(0.02),
                activation="elu"))
#model.add(Dropout(0.2))
model.add(Dense(32,
                #kernel_regularizer=regularizers.l2(0.02),
                activation="elu"))
# #model.add(Dropout(0.3))
# model.add(Dense(8,
#                 kernel_regularizer=regularizers.l2(0.01),
#                 activation="tanh"))
#model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='adam')

history = model.fit(train_x, train_y, validation_split=0.2, epochs=18, batch_size=32)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#Predict on validation set
predictions_NN_01 = model.predict(valid_x)

predictions_NN_01[predictions_NN_01>0.5] = 1
predictions_NN_01[predictions_NN_01<=0.5] = 0

predictions_NN_prob = model.predict(valid_x)
predictions_NN_prob = predictions_NN_prob[:,0]

#Print Confusion matrix
conf_mat_NN = confusion_matrix(valid_y, predictions_NN_01)
print('Confusion Matrix of Neural Network Model:\n', conf_mat_NN)

#Print accuracy
acc_NN = accuracy_score(valid_y, predictions_NN_01)
print('Overall accuracy of Neural Network model:', acc_NN)

#Print Area Under Curve
ROC_NN = roc_auc_score(valid_y, predictions_NN_01)
print('Area Under Curve (AUC) of Neural Network model:', ROC_NN)

#-----------Build Neural Network model for prediction----------
print('Building Neural Network model...')
model = Sequential()
model.add(Dense(96, input_dim=train_x.shape[1],
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(0.02),
                activation="elu"))
#model.add(Dropout(0.2))
model.add(Dense(32,
                #kernel_regularizer=regularizers.l2(0.02),
                activation="elu"))
# #model.add(Dropout(0.3))
# model.add(Dense(8,
#                 kernel_regularizer=regularizers.l2(0.01),
#                 activation="tanh"))
#model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='adam')

history = model.fit(data, y, validation_split=0.2, epochs=18, batch_size=32)

#Predict on test set
predictions_NN = model.predict(test)

NN_submission.TARGET = predictions_NN

NN_submission.to_csv('NN_submission.csv', index=False)
