print("Running --> Home Credit Default Risk - Pytorch Neural Network")

import numpy as np
import pandas as pd

import datetime
import random
import string

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('mode.chained_assignment', None)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  print("GPU NAME --> ", torch.cuda.get_device_name(0))

# SET HYPERPARAMETERS
hp_test_size = 0.2
hp_epochs = 12
hr_batch_size = 320
hp_lr= 0.000008
hp_emb_drop = 0.04
hp_layers = [800, 350]
hp_ps = [0.001,0.01]

# LOAD DATA
application_train_df = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv').sample(frac = 1)
application_test_df = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')
previous_application_df = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')

application_train_df['CSV_SOURCE'] = 'application_train.csv'
application_test_df['CSV_SOURCE'] = 'application_test.csv'
df = pd.concat([application_train_df, application_test_df])

# MANAGE previous_applications.csv
temp_previous_df = previous_application_df.groupby('SK_ID_CURR', as_index=False).agg({'NAME_CONTRACT_STATUS': lambda x: ','.join(set(','.join(x).split(',')))})
temp_previous_df['has_only_approved'] = np.where(temp_previous_df['NAME_CONTRACT_STATUS'] == 'Approved', '1', '0')
temp_previous_df['has_been_rejected'] = np.where(temp_previous_df['NAME_CONTRACT_STATUS'].str.contains('Refused'), '1', '0')

# JOIN DATA
df = pd.merge(df, temp_previous_df, on='SK_ID_CURR', how='left')

# CREATE CUSTOM COLUMNS
#################################################### total_amt_req_credit_bureau
df['total_amt_req_credit_bureau'] = (
  df['AMT_REQ_CREDIT_BUREAU_YEAR'] * 1 + 
  df['AMT_REQ_CREDIT_BUREAU_QRT'] * 2 + 
  df['AMT_REQ_CREDIT_BUREAU_MON'] * 8 + 
  df['AMT_REQ_CREDIT_BUREAU_WEEK'] * 16 + 
  df['AMT_REQ_CREDIT_BUREAU_DAY'] * 32 +
  df['AMT_REQ_CREDIT_BUREAU_HOUR'] * 64)
df['total_amt_req_credit_bureau_isnull'] = np.where(df['total_amt_req_credit_bureau'].isnull(), '1', '0')
df['total_amt_req_credit_bureau'].fillna(0, inplace=True)

#######################################################################  has_job
df['has_job'] = np.where(df['NAME_INCOME_TYPE'].isin(['Pensioner', 'Student', 'Unemployed']), '1', '0')

#######################################################################  has_children
df['has_children'] = np.where(df['CNT_CHILDREN'] > 0, '1', '0')

####################################################### clusterise_days_employed
def clusterise_days_employed(x):
    days = x['DAYS_EMPLOYED']
    if days > 0:
      return 'not available'
    else:
      days = abs(days)
      if days < 30:
        return 'less 1 month'
      elif days < 180:
        return 'less 6 months'
      elif days < 365:
        return 'less 1 year'
      elif days < 1095:
        return 'less 3 years'
      elif days < 1825:
        return 'less 5 years'
      elif days < 3600:
        return 'less 10 years'
      elif days < 7200:
        return 'less 20 years'
      elif days >= 7200:
        return 'more 20 years'
      else:
        return 'not available'
df['cluster_days_employed'] = df.apply(clusterise_days_employed, axis=1)

#######################################################################  custom_ext_source_3
def clusterise_ext_source(x):
    if str(x) == 'nan':
      return 'not available'
    else:
      if x < 0.1:
        return 'less 0.1'
      elif x < 0.2:
        return 'less 0.2'
      elif x < 0.3:
        return 'less 0.3'
      elif x < 0.4:
        return 'less 0.4'
      elif x < 0.5:
        return 'less 0.5'
      elif x < 0.6:
        return 'less 0.6'
      elif x < 0.7:
        return 'less 0.7'
      elif x < 0.8:
        return 'less 0.8'
      elif x < 0.9:
        return 'less 0.9'
      elif x <= 1:
        return 'less 1'
df['clusterise_ext_source_1'] = df['EXT_SOURCE_1'].apply(lambda x: clusterise_ext_source(x))
df['clusterise_ext_source_2'] = df['EXT_SOURCE_2'].apply(lambda x: clusterise_ext_source(x))
df['clusterise_ext_source_3'] = df['EXT_SOURCE_3'].apply(lambda x: clusterise_ext_source(x))

#######################################################################  house_variables_sum
house_vars = ['APARTMENTS_AVG','APARTMENTS_MEDI','APARTMENTS_MODE','BASEMENTAREA_AVG',
  'BASEMENTAREA_MEDI','BASEMENTAREA_MODE','COMMONAREA_AVG','COMMONAREA_MEDI',
  'COMMONAREA_MODE','ELEVATORS_AVG','ELEVATORS_MEDI','ELEVATORS_MODE','EMERGENCYSTATE_MODE',
  'ENTRANCES_AVG','ENTRANCES_MEDI','ENTRANCES_MODE','FLOORSMAX_AVG','FLOORSMAX_MEDI',
  'FLOORSMAX_MODE','FLOORSMIN_AVG','FLOORSMIN_MEDI','FLOORSMIN_MODE','FONDKAPREMONT_MODE',
  'HOUSETYPE_MODE','LANDAREA_AVG','LANDAREA_MEDI','LANDAREA_MODE','LIVINGAPARTMENTS_AVG',
  'LIVINGAPARTMENTS_MEDI','LIVINGAPARTMENTS_MODE','LIVINGAREA_AVG','LIVINGAREA_MEDI','LIVINGAREA_MODE',
  'NONLIVINGAPARTMENTS_AVG','NONLIVINGAPARTMENTS_MEDI','NONLIVINGAPARTMENTS_MODE','NONLIVINGAREA_AVG',
  'NONLIVINGAREA_MEDI','NONLIVINGAREA_MODE','TOTALAREA_MODE','WALLSMATERIAL_MODE',
  'YEARS_BEGINEXPLUATATION_AVG','YEARS_BEGINEXPLUATATION_MEDI','YEARS_BEGINEXPLUATATION_MODE',
  'YEARS_BUILD_AVG','YEARS_BUILD_MEDI','YEARS_BUILD_MODE']
df['house_variables_sum'] = df[house_vars].sum(axis=1)
df['house_variables_sum_isnull'] = np.where(df['house_variables_sum'].isnull(), '1', '0')
df['house_variables_sum'].fillna(value=df['house_variables_sum'].median(), inplace=True)


# SELECT COLUMNS
numerical_columns = [
  'AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL',
  'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION',
  'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE',
  'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'total_amt_req_credit_bureau',
  'house_variables_sum']
categorical_columns = [
  'CODE_GENDER', 'CSV_SOURCE', 'FLAG_OWN_CAR', 'NAME_EDUCATION_TYPE', 'FLAG_OWN_REALTY', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE',
  'NAME_CONTRACT_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE', 'NAME_TYPE_SUITE',
  'has_only_approved', 'has_been_rejected', 'has_job', 'has_children', 'cluster_days_employed',
  'clusterise_ext_source_1', 'clusterise_ext_source_2', 'clusterise_ext_source_3',
  'total_amt_req_credit_bureau_isnull', 'house_variables_sum_isnull']

target_column = ['TARGET']
df = df[numerical_columns + categorical_columns + target_column]

# MANAGE MISSING VALUES
for numerical_column in numerical_columns:
  if df[numerical_column].isnull().values.any():
    df[numerical_column + '_isnull'] = np.where(df[numerical_column].isnull(), '1', '0')
  df[numerical_column].fillna(value=df[numerical_column].median(), inplace=True)

for categorical_column in categorical_columns:
  df[categorical_column].fillna('NULL', inplace=True)

# STANDARDISE
min_max_scaler = preprocessing.MinMaxScaler()
df[numerical_columns] = pd.DataFrame(min_max_scaler.fit_transform(df[numerical_columns]))

# CONVERT CATEGORICAL COLUMNS INTO TYPE "category"
categorical_columns.remove('CSV_SOURCE')

for column in categorical_columns:
  df[column] = LabelEncoder().fit_transform(df[column].astype(str))
  df[column] = df[column].astype('category')
    
# SPLIT DATA INTO TRAINING vs TRAIN
train_df = df[df['CSV_SOURCE'] == 'application_train.csv']
train_output_df = pd.DataFrame(train_df['TARGET'], columns=['TARGET'])

test_df = df[df['CSV_SOURCE'] == 'application_test.csv']

# REMOVE NOT USEFUL COLUMNS
train_df.drop(columns=['CSV_SOURCE', 'TARGET'], axis=0, inplace=True)
test_df.drop(columns=['CSV_SOURCE', 'TARGET'], axis=0, inplace=True)

# CREATE VALIDATION TEST
x_train, x_validation, y_train, y_validation = train_test_split(train_df, train_output_df, test_size=hp_test_size, random_state=42)

# CREATE TENSORS
print("CREATING TENSORS...")
def create_tensors(input_df):
  stack = []
  for column in input_df.columns:
    if input_df.dtypes[column] == np.int64 or input_df.dtypes[column] == np.float64:
      stack.append(input_df[column].astype(np.float64))
    else:
      stack.append(input_df[column].cat.codes.values)
  return torch.tensor(np.stack(stack, 1), dtype=torch.float)

tensor_x_train_cat = create_tensors(x_train[categorical_columns]).float().to(device)
tensor_x_train_num = create_tensors(x_train[numerical_columns]).float().to(device)
tensor_y_train = torch.tensor(y_train.values).flatten().float().to(device)

tensor_x_valid_cat = create_tensors(x_validation[categorical_columns]).float().to(device)
tensor_x_valid_num = create_tensors(x_validation[numerical_columns]).float().to(device)
tensor_y_valid = torch.tensor(y_validation.values).flatten().float().to(device)

tensor_x_test_cat = create_tensors(test_df[categorical_columns]).float().to(device)
tensor_x_test_num = create_tensors(test_df[numerical_columns]).float().to(device)

# CREATE CATEGORICAL EMBEDDING SIZES
categorical_columns_size = [len(df[column].cat.categories) for column in categorical_columns]
categorical_embedding_sizes = [(col_size, min(50, (col_size + 1) // 2)) for col_size in categorical_columns_size]

# DEFINE NEURAL NETWORK MODEL
class Model(nn.Module):
  def __init__(self, embedding_size, input_size, num_numerical_cols, layers, ps):
    super().__init__()

    self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
    self.emb_drop = nn.Dropout(hp_emb_drop)

    self.bn_cont = nn.BatchNorm1d(num_numerical_cols)

    layerlist = []
    for i, elem in enumerate(layers):
      layerlist.append(nn.Linear(input_size, elem))
      layerlist.append(nn.ReLU(inplace=True))
      layerlist.append(nn.BatchNorm1d(layers[i]))
      layerlist.append(nn.Dropout(ps[i]))
      input_size = elem
    layerlist.append(nn.Linear(layers[-1], 1))

    self.layers = nn.Sequential(*layerlist)

  def forward(self, x_c, x_n):

    embeddings = [e(x_c[:,i].long()) for i, e in enumerate(self.all_embeddings)]

    x = torch.cat(embeddings, 1)
    x = self.emb_drop(x)

    x_n = self.bn_cont(x_n)

    x = torch.cat([x, x_n], 1)
    x = self.layers(x)

    return x

# INSTANCIATE MODEL
print("INSTANTIATING MODEL...")
num_numerical_cols = tensor_x_train_num.shape[1]

num_categorical_cols = sum((nf for ni, nf in categorical_embedding_sizes))
initial_input_size = num_categorical_cols + num_numerical_cols

model = Model(categorical_embedding_sizes, initial_input_size, num_numerical_cols, layers=hp_layers, ps=hp_ps)
sigmoid = nn.Sigmoid()
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hp_lr)
model.to(device)

# TRAIN NEURAL NETWORK MODEL
print("TRAINING MODEL...")
train_tensor_dataset = TensorDataset(tensor_x_train_cat, tensor_x_train_num, tensor_y_train)
train_loader = DataLoader(dataset=train_tensor_dataset, batch_size=hr_batch_size, shuffle=True)

model.train()

tot_y_train_in = []
tot_y_train_out = []

for epoch in range(hp_epochs):
  train_losses = []
  for x_cat, x_num, y in train_loader:
    y_train = model(x_cat, x_num)
    single_loss = loss_function(sigmoid(y_train.squeeze()), y)
    single_loss.backward() 
    optimizer.step()

    train_losses.append(single_loss.item())
    tot_y_train_in.append(y)
    tot_y_train_out.append(y_train)
  epoch_loss = 1.0 * sum(train_losses) / len(train_losses)
  epoch_auc = roc_auc_score(torch.cat(tot_y_train_in).cpu().numpy(), torch.cat(tot_y_train_out).cpu().detach().numpy())
  tot_y_train_in = []
  tot_y_train_out = []
  print("\tepoch: " + str(epoch) + "\tloss: " + str(epoch_loss) + "\tauc: " + str(epoch_auc))

# VALIDATE NEURAL NETWORK MODEL
print("VALIDATING MODEL...")
validation_tensor_dataset = TensorDataset(tensor_x_valid_cat, tensor_x_valid_num, tensor_y_valid)
validation_loader = DataLoader(dataset=validation_tensor_dataset, batch_size=hr_batch_size, shuffle=True)

valid_losses = []

model.eval()

tot_y_valid_in = []
tot_y_valid_out = []

with torch.no_grad():
  for x_cat, x_num, y in validation_loader:
    y_valid = model(x_cat, x_num)
    validation_loss = loss_function(sigmoid(y_valid.squeeze()), y)
    valid_losses.append(validation_loss.item())

    tot_y_valid_in.append(y_valid)
    tot_y_valid_out.append(y)

  valid_loss = round(1.0 * sum(valid_losses) / len(valid_losses), 5)
  print("\tloss: " + str(valid_loss))
  valid_auc = roc_auc_score(torch.cat(tot_y_valid_out).cpu(), torch.cat(tot_y_valid_in).cpu())
  print("\tauc: " + str(valid_auc))
    
# MAKE PREDICTIONS
print("MAKING PREDICTIONS...")
with torch.no_grad():
  y_test = model(tensor_x_test_cat, tensor_x_test_num)

# GENERATE SUBMISSION.csv
print("GENERATING SUBMISSIONS...")
nn_prediction_df = pd.DataFrame(y_test).astype("float")
x_scaled = min_max_scaler.fit_transform(nn_prediction_df)
nn_prediction_df = pd.DataFrame(x_scaled)
nn_prediction_df = pd.concat([nn_prediction_df, application_test_df['SK_ID_CURR']], axis=1)
nn_prediction_df.columns = ['TEMP_TARGET', 'SK_ID_CURR']
nn_prediction_df['TARGET'] = nn_prediction_df['TEMP_TARGET']
nn_prediction_df = nn_prediction_df[['SK_ID_CURR', 'TARGET']]
nn_prediction_df.to_csv('submission.csv', index=False)

print("EXECUTION COMPLETED.")