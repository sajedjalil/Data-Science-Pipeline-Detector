import numpy as np
import pandas as pd
from keras import models
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from keras import layers
from sklearn.model_selection import train_test_split
from tensorflow import math
import tensorflow as tf
from keras import backend
from keras.layers import Dropout


def boolMasking(data, column, value):
    
    return data.loc[(data.loc[:, column] == value)]


def encoder(genome, batchSize, epochs, testMode=True):
    model = models.Sequential()
    model.add(layers.Dense(772, activation='relu', input_shape=(genome.shape[-1],)))

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dense(772, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss="mae",
                  metrics=["mse"])

    if testMode:

        print(genome.std(axis=1).mean())

        trainX, testX, trainY, testY = train_test_split(genome, genome, train_size=0.75, shuffle=True)

        history = model.fit(genome, genome, batch_size=batchSize, epochs=epochs, validation_data=(testX, testY))

        return history.history


    else:

        model.fit(genome, genome, batch_size=batchSize, epochs=epochs)

        return model
    
def vEncoder(viability, batchSize, epochs, testMode=True):
    model = models.Sequential()
    model.add(layers.Dense(100, activation='relu', input_shape=(viability.shape[-1],)))

    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))

    model.add(layers.Dense(100, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss="mae",
                  metrics=["mse"])

    if testMode:

        print(viability.std(axis=1).mean())

        trainX, testX, trainY, testY = train_test_split(viability, viability, train_size=0.75, shuffle=True)

        history = model.fit(viability, viability, batch_size=batchSize, epochs=epochs, validation_data=(testX, testY))

        return history.history


    else:

        model.fit(viability, viability, batch_size=batchSize, epochs=epochs)

        return model




def viabilityPreprocessing(genome, viability, nodes, hiddenLayers, batchSize, epochs, genomePCA, viaPCA, testMode=True):


    ss = StandardScaler()


    genome = genomePCA.transform(genome)
    
    v = viability.mean(axis=1)
    
    v = v.to_numpy().reshape(-1, 1)

    viability = viaPCA.transform(viability)
    
    #genome = np.append(genome, v, axis=1)

    model = models.Sequential()
    model.add(layers.Dense(nodes, activation='relu', input_shape=(len(genome[0, :]),)))

    for n in range(hiddenLayers):
        model.add(layers.Dense(nodes, activation='relu'))
        model.add(Dropout(0.2))

    model.add(layers.Dense(viability.shape[-1], activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=["mae"])


    if testMode:

        genome = ss.fit_transform(genome)
        
        print(viability.std(axis=1).mean())

        trainX, testX, trainY, testY = train_test_split(genome, viability, train_size=0.75, shuffle=True)

        history = model.fit(trainX, trainY, batch_size=batchSize, epochs=epochs, validation_data=(testX, testY))

        return (history.history, ss)


    else:

        genome = ss.fit_transform(genome)

        model.fit(genome, viability, batch_size=batchSize, epochs=epochs)

        return (model, ss)





def dataPreprocessing(genome, viability, otherData, genomePCA, viaPCA, ppModel, ppScaler, OHE = None):
    
    if OHE is None:
        
        ohe = OneHotEncoder()
        
    else:
        
        ohe = OHE

    genome = genomePCA.transform(genome)
    
    v = viability.mean(axis=1)

    viability = viaPCA.transform(viability)
    
    v = v.to_numpy().reshape(-1, 1)
    
    #genome = np.append(genome, v, axis=1)
    
    #num = len(genome[0, :])
    
    #v = v.repeat(num, axis=1)
    
    #genome = genome * v
    
    
    genome = ppScaler.transform(genome)

    prediction = ppModel.predict(genome)

    viability  = viability - prediction

    time = otherData.loc[:, "cp_time"]

    time = time.to_numpy().reshape(-1, 1)

    otherData =otherData.drop("cp_time", axis=1)
    
    
    if OHE is None:
        
        otherData = ohe.fit_transform(otherData).toarray()

        trainData = np.append(genome, viability, axis=1)
        
        trainData = np.append(trainData, time, axis=1)
        
        trainData = np.append(trainData, otherData, axis=1)

        return (trainData, ohe)
        
        
    else:

        otherData = ohe.transform(otherData).toarray()

        trainData = np.append(genome, viability, axis=1)
        
        trainData = np.append(trainData, time, axis=1)
        
        trainData = np.append(trainData, otherData, axis=1)
        
        return trainData





def classifier(trainData, molecular, nodes, hiddenLayers, batchSize, epochs, testMode=True):
    
    def logloss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred,p_min,p_max)
        return -backend.mean(y_true*backend.log(y_pred) + (1-y_true)*backend.log(1-y_pred))
    
    p_min = 0.001
    p_max = 0.999
    
    Loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.001,
                                              reduction="auto", name="binary_crossentropy")


    ss = StandardScaler()

    model = models.Sequential()
    model.add(layers.Dense(nodes, activation='relu', input_shape=(trainData.shape[-1],)))

    for n in range(hiddenLayers):
        model.add(layers.Dense(nodes, activation='relu'))
        model.add(Dropout(0.2))

    model.add(layers.Dense(molecular.shape[-1], activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss=Loss,
                  metrics=logloss)

    molecular = molecular.to_numpy()

    if testMode:
        
        


        trainData = ss.fit_transform(trainData)
        
        print(trainData.shape)

        trainX, testX, trainY, testY = train_test_split(trainData, molecular, train_size=0.75, shuffle=True)

        history = model.fit(trainX, trainY, batch_size=batchSize, epochs=epochs, validation_data=(testX, testY))

        return (history.history, ss)


    else:

        trainData = ss.fit_transform(trainData)

        model.fit(trainData, molecular, batch_size=batchSize, epochs=epochs)

        return (model, ss)



trainData = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")
trainData2 = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
testData = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
sub = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")

print("data received.")

columns = trainData.columns

subId = sub.loc[:, "sig_id"]

gCol = list()
cCol = list()

for column in columns:

    if column[:2] == "g-":

        gCol.append(column)


    elif column[:2] == "c-":

        cCol.append(column)

trainData = pd.merge(trainData, trainData2, on="sig_id")

molecular = trainData2.drop("sig_id", axis=1)

molCol = molecular.columns

totalGenome = trainData.loc[:, gCol]

totalViability = trainData.loc[:, cCol]

proprocessingData = boolMasking(trainData, "cp_type", "ctl_vehicle")

trainingData = boolMasking(trainData, "cp_type", "trt_cp")

genome = proprocessingData.loc[:, gCol]
viability = proprocessingData.loc[:, cCol]

genome2 = trainingData.loc[:, gCol]
viability2 = trainingData.loc[:, cCol]
molecular2 = trainingData.loc[:, molCol]

gPCA = PCA(n_components=400)
vPCA = PCA(n_components=10)


VEncoder = vEncoder(totalViability, batchSize=100, epochs=50, testMode=False)


Encoder = encoder(totalGenome, batchSize=100, epochs=50, testMode=False)

gPCA.fit(totalGenome - Encoder.predict(totalGenome))

vPCA.fit(totalViability - VEncoder.predict(totalViability))

genome = genome - Encoder.predict(genome)

genome2 = genome2 - Encoder.predict(genome2)


viability = viability - VEncoder.predict(viability)

viability2 = viability2 - VEncoder.predict(viability2)



vPCA.fit(totalViability)


preModel, preSS = viabilityPreprocessing(genome, viability, nodes=2048, hiddenLayers=2, batchSize=10, epochs=20, 
                                         genomePCA=gPCA, viaPCA=vPCA, testMode=False)

otherData = trainingData.loc[:, ["cp_time", "cp_dose"]]

trainingData, OHE = dataPreprocessing(genome2, viability2, otherData, gPCA, vPCA, preModel,preSS, OHE = None)

classModel, classSS = classifier(trainingData, molecular2, nodes=4096, hiddenLayers=4, batchSize=100, epochs=10, testMode=False)


#Prediction Part

preprocessingData = boolMasking(testData, "cp_type", "ctl_vehicle")

testData = boolMasking(testData, "cp_type", "trt_cp")

genome = preprocessingData.loc[:, gCol]
viability = preprocessingData.loc[:, cCol]

otherID = preprocessingData.loc[:, "sig_id"]

genome2 = testData.loc[:, gCol]
viability2 = testData.loc[:, cCol]

sp_id = testData.loc[:, "sig_id"]

otherData2 = testData.loc[:, ["cp_time", "cp_dose"]]

genome2 = genome2 - Encoder.predict(genome2)

testingData = dataPreprocessing(genome2, viability2, otherData2, gPCA, vPCA, preModel, preSS, OHE = OHE)

testingData = classSS.transform(testingData)

print("prediction")

results = classModel.predict(testingData)


sigID = testData.loc[:, "sig_id"]

print("saving")

otherID = otherID.to_frame()

for mol in molCol:
    otherID[mol] = 0



results = pd.DataFrame(results, index=sigID, columns=molCol)

results = results.reset_index()

results = results.append(otherID)

results = results.set_index("sig_id")

results = results.where(results < 1, 1)

results = results.where(results > 0, 0)

results = results.loc[subId, :]

results = results.reset_index()

print(results.shape)
print(sub.shape)

print(results.dtypes)
print(results.isnull().any().any())

results.to_csv("./submission.csv", index=False)