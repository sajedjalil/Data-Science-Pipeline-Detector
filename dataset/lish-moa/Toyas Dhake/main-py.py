import sys 
sys.path.append('/kaggle/input/projectfiles/') 
import pandas as pd
import tensorflow as tf
import warnings
import subprocess
subprocess.run('pip install ../input/strati/iterative_stratification-0.1.6-py3-none-any.whl', shell=True, check=True)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from data import preprocessData, preprocessLabel
from model import NeuralNetworkModel, metric
from plot import plotControlAndTreatedSample, plotGene, plotViability, plotTreatmentDuration

warnings.filterwarnings("ignore")

trainData = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
trainLabel = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
testData = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
sampleSubmission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

# plotControlAndTreatedSample(trainData)
# plotTreatmentDuration(trainData)
# plotViability(trainData)
# plotGene(trainData)

train = preprocessData(trainData)
test = preprocessData(testData)

trainLabel = preprocessLabel(trainLabel)

trainLabel = trainLabel.loc[train['cp_type'] == 0].reset_index(drop=True)
train = train.loc[train['cp_type'] == 0].reset_index(drop=True)

seeds = [453, 23, 2]

results = trainLabel.copy()
sampleSubmission.loc[:, trainLabel.columns] = 0
results.loc[:, trainLabel.columns] = 0

for seed in seeds:
    for i, (trainStrat, testStrat) in enumerate(
            MultilabelStratifiedKFold(n_splits=5, random_state=seed, shuffle=True).split(trainLabel, trainLabel)):
        print(f'Fold {i}')

        model = NeuralNetworkModel(tf, len(train.columns))
        checkpointSaveFile = f'repeat:{seed}_Fold:{i}.hdf5'
        reduceLearningRateLoss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1,
                                                   min_delta=1e-2, mode='min')
        callbackCheckpoint = ModelCheckpoint(checkpointSaveFile, monitor='val_loss', verbose=0, save_best_only=True,
                                             save_weights_only=True, mode='min')

        model.fit(train.values[trainStrat][:, :], trainLabel.values[trainStrat],
                  validation_data=(train.values[testStrat][:, :], trainLabel.values[testStrat]), epochs=20, batch_size=64,
                  callbacks=[reduceLearningRateLoss, callbackCheckpoint], verbose=2)

        model.load_weights(checkpointSaveFile)
        testPrediction = model.predict(test.values[:, :])
        trainPrediction = model.predict(train.values[testStrat][:, :])

        sampleSubmission.loc[:, trainLabel.columns] += testPrediction
        results.loc[testStrat, trainLabel.columns] += trainPrediction

sampleSubmission.loc[:, trainLabel.columns] /= ((i + 1) * len(seeds))
results.loc[:, trainLabel.columns] /= len(seeds)

print(f'Score: {metric(trainLabel, results, trainLabel)}')

sampleSubmission.loc[test['cp_type'] == 1, trainLabel.columns] = 0

sampleSubmission.to_csv('submission.csv', index=False)
