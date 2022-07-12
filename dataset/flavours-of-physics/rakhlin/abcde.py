'''
This is not a model that brougth me to the 5th place
This one is based on post-competition forum chat re. ensemblig strong and weak classifiers, and
particularly on remarks made by Josef Slavicek (3rd place) who wonder why raising strong classifier's
prediction to a big power works so well, and on my thought what a physical soundness of a model is.
Should produce a score 0.999+ with n_models=5+ and n_epochs=100..120 (increase and run locally)
'''

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
import pandas as pd
import numpy as np

np.random.seed(1337) # for reproducibility

def add_features(df):
    # features from phunter
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    df['NEW_IP_dira'] = df['IP']*df['dira']
    df['p0p2_ip_ratio']=df['IP']/df['IP_p0p2']
    df['p1p2_ip_ratio']=df['IP']/df['IP_p1p2']
    df['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)
    df['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
    df['iso_min'] = df.loc[:, ['isolationa', 'isolationb', 'isolationc','isolationd', 'isolatione', 'isolationf']].min(axis=1)
    # Stepan Obraztsov's magic features
    df['NEW_FD_SUMP'] = df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])
    df['NEW5_lt'] = df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3
    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    # "super" feature from Grzegorz Sionkowski 
    df['NEW_FD_LT']=df['FlightDistance']/df['LifeTime']
    df['flight_dist_sig2'] = (df['FlightDistance']/df['FlightDistanceError'])**2
    return df

def load_data(data_file, output_y=True):
    df = pd.read_csv(data_file)
    df = add_features(df)
    if output_y:    # shuffle training set
        df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)
    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'weight', 'signal']
    features = list(f for f in df.columns if f not in filter_out)
    return df[features].values, df['signal'].values if output_y else None, df['id']

def model_factory(n_inputs):
    model = Sequential()
    model.add(Dense(n_inputs, 800))
    model.add(PReLU((800,)))
    model.add(Dropout(0.5))
    model.add(Dense(800, 2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model
    
# load and preprocess data
X_train, y_train, _ = load_data("../input/training.csv")
X_test, _, id = load_data("../input/test.csv", output_y=False)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
y_train = np_utils.to_categorical(y_train)
X_test = scaler.transform(X_test)

# ensemble of single-layer neural nets
n_models = 1    # 5 or more models is better
n_epochs = 100   # must be increased to 120..150
probs = None
for i in range(n_models):
    print("\n----------- Keras: train Model %d/%d ----------\n" % (i+1,n_models))
    model = model_factory(X_train.shape[1])
    model.fit(X_train, y_train, batch_size=64, nb_epoch=n_epochs, validation_data=None, verbose=2, show_accuracy=True)
    p = model.predict(X_test, batch_size=256, verbose=0)[:, 1]
    probs = p if probs is None else probs + p
probs /= n_models

# Forum's idea of combination of 'strong' and 'weak' classifier brought to extreme.
# Keep only firm predictions from 'strong' classifier, substitute the other with noise
np.random.seed(1337) # for reproducibility
random_classifier = np.random.rand(len(probs))
q = 0.98
combined_probs = q * (probs ** 30) + (1 - q) * random_classifier
df = pd.DataFrame({"id": id, "prediction": combined_probs})
df.to_csv("submission.csv", index=False);
