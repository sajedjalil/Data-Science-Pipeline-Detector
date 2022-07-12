#Inspired by:

#Theo Viel's kernel (https://www.kaggle.com/theoviel/load-the-totality-of-the-data) - dtypes
#Vladislav Bogorod's kernel (https://www.kaggle.com/bogorodvo/lightgbm-baseline-model-using-sparse-matrix) - base
#Andrew Lukyanenko's kernel (https://www.kaggle.com/artgor/is-this-malware-eda-fe-and-lgb-updated) - features


import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.sparse import vstack, csr_matrix, save_npz, load_npz
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import gc
gc.enable()

dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float32',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int16',
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
        'UacLuaenable':                                         'float64', # was 'float32'
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float32', # was 'float16'
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float32', # was 'float16'
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float64', # was 'float32'
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float64', # was 'float32'
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float32', # was 'float16'
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float32', # was 'float16'
        'Census_InternalPrimaryDisplayResolutionVertical':      'float32', # was 'float16'
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float64', # was 'float32'
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

print('Download Train and Test Data.\n')
train = pd.read_csv('../input/train.csv', dtype=dtypes, low_memory=True)
train['MachineIdentifier'] = train.index.astype('uint32')
test  = pd.read_csv('../input/test.csv',  dtype=dtypes, low_memory=True)
test['MachineIdentifier']  = test.index.astype('uint32')

#Add some new features
cols = train.columns.tolist()[:-1]
cols.append('ResolutionRatio')
cols.append('primary_drive_c_ratio')
cols.append('non_primary_drive_MB')
cols.append('aspect_ratio')
cols.append('dpi')
cols.append('MegaPixels')
cols.append('ram_per_processor')
cols.append('new_num_0')
cols.append('new_num_1')
cols.append('HasDetections')
train['ResolutionRatio'] = train['Census_InternalPrimaryDisplayResolutionVertical'] / train['Census_InternalPrimaryDisplayResolutionHorizontal']
test['ResolutionRatio'] = test['Census_InternalPrimaryDisplayResolutionVertical'] / test['Census_InternalPrimaryDisplayResolutionHorizontal']
train['primary_drive_c_ratio'] = train['Census_SystemVolumeTotalCapacity']/ train['Census_PrimaryDiskTotalCapacity']
train['non_primary_drive_MB'] = train['Census_PrimaryDiskTotalCapacity'] - train['Census_SystemVolumeTotalCapacity']
test['primary_drive_c_ratio'] = test['Census_SystemVolumeTotalCapacity']/ test['Census_PrimaryDiskTotalCapacity']
test['non_primary_drive_MB'] = test['Census_PrimaryDiskTotalCapacity'] - test['Census_SystemVolumeTotalCapacity']
train['aspect_ratio'] = train['Census_InternalPrimaryDisplayResolutionHorizontal']/ train['Census_InternalPrimaryDisplayResolutionVertical']
test['aspect_ratio'] = test['Census_InternalPrimaryDisplayResolutionHorizontal']/ test['Census_InternalPrimaryDisplayResolutionVertical']
train['dpi'] = ((train['Census_InternalPrimaryDisplayResolutionHorizontal']**2 + train['Census_InternalPrimaryDisplayResolutionVertical']**2)**.5)/(train['Census_InternalPrimaryDiagonalDisplaySizeInInches'])
test['dpi'] = ((test['Census_InternalPrimaryDisplayResolutionHorizontal']**2 + test['Census_InternalPrimaryDisplayResolutionVertical']**2)**.5)/(test['Census_InternalPrimaryDiagonalDisplaySizeInInches'])
train['MegaPixels'] = (train['Census_InternalPrimaryDisplayResolutionHorizontal'] * train['Census_InternalPrimaryDisplayResolutionVertical'])/1e6
test['MegaPixels'] = (test['Census_InternalPrimaryDisplayResolutionHorizontal'] * test['Census_InternalPrimaryDisplayResolutionVertical'])/1e6
train['ram_per_processor'] = train['Census_TotalPhysicalRAM']/ train['Census_ProcessorCoreCount']
test['ram_per_processor'] = test['Census_TotalPhysicalRAM']/ test['Census_ProcessorCoreCount']
train['new_num_0'] = train['Census_InternalPrimaryDiagonalDisplaySizeInInches'] / train['Census_ProcessorCoreCount']
test['new_num_0'] = test['Census_InternalPrimaryDiagonalDisplaySizeInInches'] / test['Census_ProcessorCoreCount']
train['new_num_1'] = train['Census_ProcessorCoreCount'] * train['Census_InternalPrimaryDiagonalDisplaySizeInInches']
test['new_num_1'] = test['Census_ProcessorCoreCount'] * test['Census_InternalPrimaryDiagonalDisplaySizeInInches']
train['Census_IsFlightingInternal'] = train['Census_IsFlightingInternal'].fillna(1)
train['Census_ThresholdOptIn'] = train['Census_ThresholdOptIn'].fillna(1)
train['Census_IsWIMBootEnabled'] = train['Census_IsWIMBootEnabled'].fillna(1)
train['Wdft_IsGamer'] = train['Wdft_IsGamer'].fillna(0)
test['Census_IsFlightingInternal'] = test['Census_IsFlightingInternal'].fillna(1)
test['Census_ThresholdOptIn'] = test['Census_ThresholdOptIn'].fillna(1)
test['Census_IsWIMBootEnabled'] = test['Census_IsWIMBootEnabled'].fillna(1)
test['Wdft_IsGamer'] = test['Wdft_IsGamer'].fillna(0)
train = train[cols]
del cols
gc.collect()

print('Transform all features to category.\n')
for usecol in train.columns.tolist()[1:-1]:

    train[usecol] = train[usecol].astype('str')
    test[usecol] = test[usecol].astype('str')
    
    #Fit LabelEncoder
    le = LabelEncoder().fit(
            np.unique(train[usecol].unique().tolist()+
                      test[usecol].unique().tolist()))

    #At the end 0 will be used for dropped values
    train[usecol] = le.transform(train[usecol])+1
    test[usecol]  = le.transform(test[usecol])+1

    agg_tr = (train
              .groupby([usecol])
              .aggregate({'MachineIdentifier':'count'})
              .reset_index()
              .rename({'MachineIdentifier':'Train'}, axis=1))
    agg_te = (test
              .groupby([usecol])
              .aggregate({'MachineIdentifier':'count'})
              .reset_index()
              .rename({'MachineIdentifier':'Test'}, axis=1))

    agg = pd.merge(agg_tr, agg_te, on=usecol, how='outer').replace(np.nan, 0)
    #Select values with more than 1000 observations
    agg = agg[(agg['Train'] > 1000)].reset_index(drop=True)
    agg['Total'] = agg['Train'] + agg['Test']
    #Drop unbalanced values
    agg = agg[(agg['Train'] / agg['Total'] > 0.2) & (agg['Train'] / agg['Total'] < 0.8)]
    agg[usecol+'Copy'] = agg[usecol]

    train[usecol] = (pd.merge(train[[usecol]], 
                              agg[[usecol, usecol+'Copy']], 
                              on=usecol, how='left')[usecol+'Copy']
                     .replace(np.nan, 0).astype('int').astype('category'))

    test[usecol]  = (pd.merge(test[[usecol]], 
                              agg[[usecol, usecol+'Copy']], 
                              on=usecol, how='left')[usecol+'Copy']
                     .replace(np.nan, 0).astype('int').astype('category'))

    del le, agg_tr, agg_te, agg, usecol
    gc.collect()
          
y_train = np.array(train['HasDetections'])
train_ids = train.index
test_ids  = test.index

del train['HasDetections'], train['MachineIdentifier'], test['MachineIdentifier']
gc.collect()

print("If you don't want use Sparse Matrix choose Kernel Version 2 to get simple solution.\n")

print('--------------------------------------------------------------------------------------------------------')
print('Transform Data to Sparse Matrix.')
print('Sparse Matrix can be used to fit a lot of models, eg. XGBoost, LightGBM, Random Forest, K-Means and etc.')
print('To concatenate Sparse Matrices by column use hstack()')
print('Read more about Sparse Matrix https://docs.scipy.org/doc/scipy/reference/sparse.html')
print('Good Luck!')
print('--------------------------------------------------------------------------------------------------------')

#Fit OneHotEncoder
ohe = OneHotEncoder(categories='auto', sparse=True, dtype='uint8').fit(train)

#Transform data using small groups to reduce memory usage
m = 100000
train = vstack([ohe.transform(train[i*m:(i+1)*m]) for i in range(train.shape[0] // m + 1)])
test  = vstack([ohe.transform(test[i*m:(i+1)*m])  for i in range(test.shape[0] // m +  1)])
save_npz('train.npz', train, compressed=True)
save_npz('test.npz',  test,  compressed=True)

del ohe, train, test
gc.collect()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf.get_n_splits(train_ids, y_train)

lgb_test_result  = np.zeros(test_ids.shape[0])
counter = 0

print('\nLightGBM\n')

for train_index, test_index in skf.split(train_ids, y_train):
    
    print('Fold {}\n'.format(counter + 1))
    
    train = load_npz('train.npz')
    X_fit = vstack([train[train_index[i*m:(i+1)*m]] for i in range(train_index.shape[0] // m + 1)])
    X_val = vstack([train[test_index[i*m:(i+1)*m]]  for i in range(test_index.shape[0] //  m + 1)])
    X_fit, X_val = csr_matrix(X_fit, dtype='float32'), csr_matrix(X_val, dtype='float32')
    y_fit, y_val = y_train[train_index], y_train[test_index]
    
    del train
    gc.collect()

    lgb_model = lgb.LGBMClassifier(max_depth=-1,
                                   n_estimators=25000,
                                   learning_rate=0.05,
                                   num_leaves=2**12-1,
                                   colsample_bytree=0.28,
                                   objective='binary', 
                                   n_jobs=-1)
                                   
                               
    lgb_model.fit(X_fit, y_fit, eval_metric='auc', 
                  eval_set=[(X_val, y_val)], 
                  verbose=100, early_stopping_rounds=100)
                  
    
    del X_fit, X_val, y_fit, y_val, train_index, test_index
    gc.collect()
    
    test = load_npz('test.npz')
    test = csr_matrix(test, dtype='float32')
    lgb_test_result += lgb_model.predict_proba(test)[:,1]
    counter += 1
    
    del test, lgb_model
    gc.collect()
    

submission = pd.read_csv('../input/sample_submission.csv')
submission['HasDetections'] = lgb_test_result / counter
submission.to_csv('lgb_submission.csv', index=False)

print('\nDone.')