import numpy as np
import pandas as pd
import gc
import random
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from keras import backend as K
from keras.models import Sequential, load_model
from keras.optimizers import Adam, Nadam
from keras.initializers import glorot_uniform
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

def EarlyStop(patience):
    return EarlyStopping(monitor = "val_loss",
                          min_delta = 0,
                          mode = "min",
                          verbose = 1, 
                          patience = patience)

def ModelCheckpointFull(model_name):
    return ModelCheckpoint(model_name, 
                            monitor = 'val_loss', 
                            verbose = 1, 
                            save_best_only = True, 
                            save_weights_only = False, 
                            mode = 'min', 
                            period = 1)

# Probability Density Function Value based on Chris Deotte's comment
pdfs={"var_0":0.18,
"var_1":0.2,
"var_2":0.2,
"var_3":0.14,
"var_4":0.12,
"var_5":0.18,
"var_6":0.2,
"var_7":0,
"var_8":0.125,
"var_9":-0.2,
"var_10":0,
"var_11":0.14,
"var_12":-0.22,
"var_13":-0.22,
"var_14":-0.11,
"var_15":0.14,
"var_16":0.12,
"var_17":0,
"var_18":0.2,
"var_19":0.12,
"var_20":-0.13,
"var_21":-0.18,
"var_22":0.22,
"var_23":-0.14,
"var_24":0.15,
"var_25":0.125,
"var_26":0.2,
"var_27":0,
"var_28":-0.13,
"var_29":0,
"var_30":0,
"var_31":-0.115,
"var_32":0.17,
"var_33":-0.2,
"var_34":-0.19,
"var_35":0.15,
"var_36":-0.15,
"var_37":0.125,
"var_38":0,
"var_39":-0.112,
"var_40":0.18,
"var_41":0,
"var_42":0,
"var_43":-0.16,
"var_44":-0.16,
"var_45":-0.12,
"var_46":0,
"var_47":0.11,
"var_48":0.13,
"var_49":0.18,
"var_50":-0.12,
"var_51":0.17,
"var_52":0.13,
"var_53":0.22,
"var_54":-0.12,
"var_55":0.15,
"var_56":-0.18,
"var_57":-0.12,
"var_58":-0.125,
"var_59":-0.12,
"var_60":0.13,
"var_61":0.115,
"var_62":0.12,
"var_63":-0.13,
"var_64":-0.12,
"var_65":0.11,
"var_66":0.13,
"var_67":0.15,
"var_68":-0.125,
"var_69":0.14,
"var_70":0.14,
"var_71":0.14,
"var_72":-0.12,
"var_73":-0.11,
"var_74":0.12,
"var_75":-0.18,
"var_76":-0.2,
"var_77":-0.12,
"var_78":0.22,
"var_79":0.13,
"var_80":-0.22,
"var_81":-0.22,
"var_82":0.15,
"var_83":-0.15,
"var_84":0.11,
"var_85":-0.16,
"var_86":-0.16,
"var_87":-0.15,
"var_88":-0.135,
"var_89":0.16,
"var_90":0.15,
"var_91":0.18,
"var_92":-0.18,
"var_93":-0.16,
"var_94":0.22,
"var_95":0.16,
"var_96":0.11,
"var_97":0.13,
"var_98":0,
"var_99":0.2,
"var_100":0,
"var_101":-0.12,
"var_102":-0.15,
"var_103":0,
"var_104":-0.12,
"var_105":0.13,
"var_106":0.15,
"var_107":-0.16,
"var_108":-0.2,
"var_109":-0.22,
"var_110":0.2,
"var_111":0.16,
"var_112":0.13,
"var_113":-0.13,
"var_114":-0.15,
"var_115":-0.16,
"var_116":-0.13,
"var_117":0,
"var_118":0.15,
"var_119":0.18,
"var_120":0,
"var_121":-0.18,
"var_122":-0.18,
"var_123":-0.17,
"var_124":0,
"var_125":0.16,
"var_126":0,
"var_127":-0.18,
"var_128":0.15,
"var_129":0,
"var_130":0.18,
"var_131":-0.17,
"var_132":-0.15,
"var_133":0.2,
"var_134":0.15,
"var_135":0.15,
"var_136":0,
"var_137":0.16,
"var_138":0.125,
"var_139":-0.21,
"var_140":0.115,
"var_141":-0.18,
"var_142":-0.13,
"var_143":-0.125,
"var_144":0.13,
"var_145":0.16,
"var_146":-0.21,
"var_147":0.18,
"var_148":-0.18,
"var_149":-0.15,
"var_150":-0.15,
"var_151":0.16,
"var_152":-0.125,
"var_153":-0.115,
"var_154":-0.20,
"var_155":0.18,
"var_156":-0.14,
"var_157":0.16,
"var_158":0,
"var_159":0.115,
"var_160":0,
"var_161":0,
"var_162":0.17,
"var_163":0.17,
"var_164":0.2,
"var_165":-0.2,
"var_166":-0.22,
"var_167":0.14,
"var_168":0.13,
"var_169":-0.16,
"var_170":0.2,
"var_171":0.14,
"var_172":-0.15,
"var_173":0.16,
"var_174":-0.22,
"var_175":0.14,
"var_176":0.11,
"var_177":-0.18,
"var_178":-0.13,
"var_179":0.18,
"var_180":0.17,
"var_181":0.125,
"var_182":0,
"var_183":-0.11,
"var_184":0.2,
"var_185":0,
"var_186":-0.13,
"var_187":0.135,
"var_188":-0.17,
"var_189":0.115,
"var_190":0.18,
"var_191":0.18,
"var_192":-0.13,
"var_193":-0.125,
"var_194":-0.15,
"var_195":0.15,
"var_196":0.15,
"var_197":-0.16,
"var_198":-0.20,
"var_199":0.125 }

# Random Seed
seed = 12345
np.random.seed(seed)
random.seed(seed)

# Constants
epochs = 150
batch_size = 1024
number_of_folds = 15

# Load Data
train = pd.read_csv('../input/train.csv')
labels = train.target
test = pd.read_csv('../input/test.csv')

# Summary
print('====== Dataset Shapes')
print('Train: ' + str(train.shape))
print('Labels: ' + str(labels.shape))
print('Test: ' + str(test.shape))

# Features
feats = [f for f in train.columns if f not in ['target', 'ID_code']]

# Sort Based on PDF per Column
sorted_pdfs = OrderedDict(sorted(pdfs.items(), key = lambda x: x[1]))
feats_sorted = list(sorted_pdfs.keys())
#feats_sorted = train[feats].kurtosis().sort_values(ascending=False) # Use in combination with 'feats_sorted.index' 
print(feats_sorted)
print(train[feats_sorted].head())

# Flip value of feature when PDF is negative
for feature in feats_sorted:
    if sorted_pdfs[feature] < 0:
        train[feature] *= -1
        test[feature] *= -1

# Scaling
scaler = preprocessing.RobustScaler()
train[feats_sorted] = scaler.fit_transform(train[feats_sorted])
test[feats_sorted] = scaler.transform(test[feats_sorted])

# Input Shape
input_shape = train[feats_sorted].shape[1]

# Define CNN 1D model
def create_model():
    model = Sequential()
    model.add(Conv1D(16, 2, activation = 'relu', input_shape=(input_shape, 1), kernel_initializer = glorot_uniform(seed = seed)))
    model.add(BatchNormalization())       
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(16, activation = 'relu', kernel_initializer = glorot_uniform(seed = seed)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid', kernel_initializer = glorot_uniform(seed = seed)))
    model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr=0.0005), metrics = ['accuracy'])
    #print(model.summary())

    return model

# Reshape
train = train[feats_sorted].values.reshape(-1, 200, 1)
test = test[feats_sorted].values.reshape(-1, 200, 1)

# CV Folds
folds = StratifiedKFold(n_splits = number_of_folds, shuffle = True, random_state = seed)

# Arrays to store predictions
oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test.shape[0])

# Loop through folds
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train, labels)):
    train_x, train_y = train[train_idx], labels.iloc[train_idx]
    valid_x, valid_y = train[valid_idx], labels.iloc[valid_idx]

    print('Running Fold: ' + str(n_fold))

    # CNN 1D model
    model = create_model()
    model.fit(train_x, train_y, 
                validation_data = (valid_x, valid_y), 
                epochs = epochs, 
                batch_size = batch_size, 
                callbacks = [EarlyStop(25), ModelCheckpointFull('model.h5')],
                verbose = 2)

    # Delete Model
    del model
    gc.collect()

	# Reload Best Saved Model
    model = load_model('model.h5')

    # OOF Predictions
    oof_preds[valid_idx] = model.predict(valid_x).reshape(-1,)
    
    # Submission Predictions
    predictions = model.predict(test).reshape(-1,)
    sub_preds += predictions / number_of_folds

    # Fold AUC Score
    print('Fold %2d AUC : %.6f' % (n_fold, roc_auc_score(valid_y, oof_preds[valid_idx])))        

    # Cleanup 
    del model, train_x, train_y, valid_y, valid_x
    K.clear_session()
    gc.collect

print('Full AUC score %.6f' % roc_auc_score(labels, oof_preds))

# Generate Submission
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = sub_preds
submission.to_csv('submission.csv', index=False)