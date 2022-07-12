# Small parts of the code are based on Vladimir Bogorod's kernel: https://www.kaggle.com/bogorodvo/lightgbm-baseline-model-using-sparse-matrix
# I tried to take a different approach to increase speed of the data processing and try also a different classifier.

# Import Modules
import numpy as np
import pandas as pd
import gc
import random
import warnings
import os
from scipy.sparse import vstack, csr_matrix, save_npz, load_npz
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

# Random Seed
seed = 422499
np.random.seed(seed)
random.seed(seed)

# Constants
number_of_rows = 8921483
folds = 5

dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }
        
# Load Data
train = pd.read_csv('../input/train.csv', dtype = dtypes, low_memory = True)
test  = pd.read_csv('../input/test.csv', dtype = dtypes, low_memory = True)
labels = train.HasDetections.values

# Get Indexes
train_ids = train.index
test_ids  = test.index

# Drop some columns and 3 Highest Cardinal features
train.drop(['HasDetections', 'MachineIdentifier', 'Census_SystemVolumeTotalCapacity', 'Census_OEMModelIdentifier', 'CityIdentifier'], axis = 1, inplace = True)
test.drop(['MachineIdentifier', 'Census_SystemVolumeTotalCapacity', 'Census_OEMModelIdentifier', 'CityIdentifier'], axis = 1, inplace = True)

# Shapes
print('Train Shape: ' + str(train.shape))
print('Test Shape: ' + str(test.shape))

# Concatenate Dataframes
df = train.append(test).reset_index()

# Cleanup
del train, test
gc.collect()

# Drop index
df.drop(['index'], axis = 1, inplace = True)

# Label Encode all data
print('Perform Label Encoding')
for col in df.columns:
    df[col] = (pd.factorize(df[col], na_sentinel = 0)[0]).astype(np.uint16)
    
# Setup One Hot Encoding
ohe = OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8').fit(df)
features = ohe.get_feature_names(input_features=df.columns)
print('Total Categorical Features: ' + str(len(features)))

# Split back to Train and Test
train = df[:number_of_rows]
test = df[number_of_rows:]

# Cleanup
del df
gc.collect()

# Transform data using small batches to reduce memory usage
print('Perform OHE')
m = 100000
train = vstack([ohe.transform(train[i*m:(i+1)*m]) for i in range(train.shape[0] // m + 1)])
test  = vstack([ohe.transform(test[i*m:(i+1)*m])  for i in range(test.shape[0] // m +  1)])

# Convert to Sparse Column Matrix to use .indptr for values per column 
train = train.tocsc()
test = test.tocsc()     

# Use masking to get index of columns with more then the specified amount of values
values_threshold = 1000
train_index_values = np.diff(train.indptr)
test_index_values = np.diff(test.indptr)
total_index_values = train_index_values + test_index_values
train_index_threshold_mask = train_index_values >= values_threshold
balance_mask = ((train_index_values / total_index_values) > 0.20) & ((train_index_values / total_index_values) < 0.80)
col_index = train_index_threshold_mask & balance_mask
print ('Feature columns with values above threshold: ' + str(col_index.sum()))

# Apply index
train = train[ : ,col_index]
test = test[ : ,col_index]

# Convert back to Sparse Row matrix
train =  train.tocsr()
test = test.tocsr()

# Shapes
print(train.shape)
print(test.shape)

# Save to disk
save_npz('train.npz', train, compressed = True)
save_npz('test.npz',  test,  compressed = True)

# Cleanup
del ohe, test, train
gc.collect()

# CV
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)

# Placeholder Arrays
oof_preds = np.zeros(train_ids.shape[0])
sub_preds = np.zeros(test_ids.shape[0])
    
# Perform CV
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_ids, labels)):
    
    print('Start Data prep Fold ' + str(n_fold))
    
    # Load Train Data and split.
    train = load_npz('train.npz')
    train_x = vstack([train[train_idx[i*m:(i+1)*m]] for i in range(train_idx.shape[0] // m + 1)])
    valid_x = vstack([train[valid_idx[i*m:(i+1)*m]]  for i in range(valid_idx.shape[0] //  m + 1)])
    train_x, valid_x = csr_matrix(train_x, dtype='float32'), csr_matrix(valid_x, dtype='float32')
    train_y, valid_y = labels[train_idx], labels[valid_idx]
    
    # Cleanup
    del train
    gc.collect()
        
    print('Start Fold ' + str(n_fold))

    # Logistic Regression Classifier
    clf = LogisticRegression(
                C = 0.05,
                max_iter = 100,
                tol = 0.0001,
                solver = 'sag',
                fit_intercept = True,
                penalty = 'l2',
                dual = False,
                verbose = 0)

    clf.fit(train_x, train_y)

    # Validation Set
    oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
    print('Fold %2d AUC score: %.5f' % (n_fold, roc_auc_score(valid_y, oof_preds[valid_idx])))        
    
    # Cleanup 
    del train_x, train_y, valid_y, valid_x
    gc.collect()
    
    # Test Set
    test = load_npz('test.npz')
    test = csr_matrix(test, dtype='float32')
    predictions = clf.predict_proba(test)[:, 1]
    sub_preds += predictions / folds.n_splits
    
    # Cleanup 
    del clf, test
    gc.collect()

print('Final AUC score %.5f' % roc_auc_score(labels, oof_preds))

# Generate Final Submission file
submission = pd.read_csv('../input/sample_submission.csv')
submission['HasDetections'] = np.array(sub_preds)
submission.to_csv('submission.csv', index = False)

print('Done..')
