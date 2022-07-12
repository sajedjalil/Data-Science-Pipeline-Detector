import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_addons as tfa



#Getting Dummies
def Dummies(Data):
    dum1 = pd.get_dummies(Data['cp_type'])
    dum2 = pd.get_dummies(Data['cp_dose'])
    del Data['sig_id']
    print("test feat")
    print(Data)
    del Data['cp_dose']
    del Data['cp_type']
    dums = pd.concat([dum1,dum2],axis=1)
    Data = pd.concat([Data, dums], axis=1)
    return Data

def Feature_selection_methods(Data,Col_names):
    #Standart scaler
    print(Data.shape, 'before selection')
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(Data)
    Data = scaler.transform(Data)

    from sklearn.feature_selection import VarianceThreshold
    var = VarianceThreshold()
    var.fit(Data)
    Data = var.transform(Data)
    print(Data.shape,'after selection')
    Data = pd.DataFrame(data=Data, columns=Col_names)
    return Data


from tensorflow.keras.callbacks import *
def get_callback(fold):
    return [
#         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-5),
        ModelCheckpoint(f'best_model.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
    ]



def Feature_selection_methods(Data,Col_names):
    #Standart scaler
    print(Data.shape, 'before selection')
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(Data)
    Data = scaler.transform(Data)

    from sklearn.feature_selection import VarianceThreshold
    var = VarianceThreshold()
    var.fit(Data)
    Data = var.transform(Data)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(Data)
    Data = scaler.transform(Data)

    print(Data.shape,'after selection')
    Data = pd.DataFrame(data=Data, columns=Col_names)
    return Data


from tensorflow.keras.callbacks import *
def get_callback(fold):
    return [
#         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-5),
        ModelCheckpoint(f'best_model.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
    ]



def build_model(train_features):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(len(train_features.columns))))
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Dense(1000,activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.2))
    #model.add(tf.keras.layers.Dense(1000,activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.2))
    #model.add(tf.keras.layers.Dense(500,activation='tanh'))
    #model.add(tf.keras.layers.Dropout(0.2))
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Dense(500,activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(1524, activation='relu')))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(1524, activation='relu')))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(206,activation='sigmoid'))
    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004), loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=1e-15))
    model.compile(optimizer=tfa.optimizers.AdamW(lr = 1e-4, weight_decay = 1e-5, clipvalue = 756),
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=1e-15),metrics=tf.keras.metrics.Accuracy()
                  )
    return model

def k_fold_model_estimator(SPLITS,features,targets):
    val_losses = []
    history = []
    for i in range(SPLITS):
        print('fold',i)
        from sklearn.model_selection import train_test_split
        import random
        X_train , X_val , y_train , y_val = train_test_split(features,targets, random_state=random.randint(30,45),test_size=0.2)
        X_train = np.array(X_train).astype(np.float32)
        X_val = np.array(X_val).astype(np.float32)
        y_train = np.array(y_train).astype(np.float32)
        y_val = np.array(y_val).astype(np.float32)
        model = build_model(features)
        hist = model.fit(X_train,y_train,validation_data=(X_val, y_val), callbacks=[ModelCheckpoint(f'best_model{i}.h5', monitor='val_loss', save_best_only=True,
        save_weights_only=False, verbose=1),ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_lr=1e-6, patience=4, verbose=1, mode='auto'),EarlyStopping(monitor="val_loss", mode="min", restore_best_weights=True, patience= 10, verbose = 1)],epochs=35,batch_size=128)
        history.append(hist)
    best_val_loss = [np.min(hist.history['val_loss']) for hist in history]
    print("Best val loss for each fold: ", best_val_loss)
    print("MIN val loss: ", np.min(best_val_loss))
    min_loss = np.argwhere(best_val_loss==min(best_val_loss))
    min_loss = int(min_loss)
    model = tf.keras.models.load_model(f'best_model{min_loss}.h5')
    return model


train_features = pd.read_csv("../input/lish-moa/train_features.csv", encoding="utf-8-sig")
test_features = pd.read_csv("../input/lish-moa/test_features.csv", encoding="utf-8-sig")
#extracting IDS into submission dict
submission_data = {}

submission_data['sig_id']=test_features['sig_id'].values
##########


train_features = Dummies(train_features)
Col_names = train_features.columns.tolist()
test_features  = Dummies(test_features)
train_features = Feature_selection_methods(train_features,Col_names)
test_features = Feature_selection_methods(test_features,Col_names)
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
targets = train_targets_scored.columns.tolist()
del train_targets_scored["sig_id"]
#signs = test_features['sig_id']

model = k_fold_model_estimator(8,train_features,train_targets_scored)
pred = model.predict(test_features)

print(targets)
del targets[0]
for i, target in enumerate(targets):
    submission_data[target] = pred[:, i]
submission_csv = pd.DataFrame(data=submission_data)
submission_csv.to_csv('submission.csv', index=False)
print(submission_csv)
