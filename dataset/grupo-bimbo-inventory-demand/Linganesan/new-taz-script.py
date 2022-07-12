import pandas as pd
import numpy as np

train_types = {'Agencia_ID':np.uint16, 'Ruta_SAK':np.uint16, 'Cliente_ID':np.uint32, 
               'Producto_ID':np.uint16, 'Demanda_uni_equil':np.uint32}

test_types = {'Agencia_ID':np.uint16, 'Ruta_SAK':np.uint16, 'Cliente_ID':np.uint32, 
              'Producto_ID':np.uint16, 'id':np.uint32}

print("Reading data...")
df_train = pd.read_csv('../input/train.csv', usecols=train_types.keys(), dtype=train_types)
df_test = pd.read_csv('../input/test.csv',usecols=test_types.keys(), dtype=test_types)


## transform target variable to log(1 + demand)
df_train['log_demand'] = 1.006999 * np.log1p(df_train['Demanda_uni_equil'] + 0.01159) - 0.01159
#overall mean
mean_total = np.mean(df_train['log_demand'])                         

#mean by product
mean_P =  pd.DataFrame({'MP':df_train.groupby('Producto_ID')['log_demand'].mean()}).reset_index()
#mean by cliente
mean_C =  pd.DataFrame({'MC':df_train.groupby('Cliente_ID')['log_demand'].mean()}).reset_index()
#mean by product and agencia
mean_PA = pd.DataFrame({'MPA':df_train.groupby(['Producto_ID','Agencia_ID'])['log_demand'].mean()}).reset_index()
#mean by product and ruta
mean_PR = pd.DataFrame({'MPR':df_train.groupby(['Producto_ID','Ruta_SAK'])['log_demand'].mean()}).reset_index()
#mean by product, client, agencia
mean_PCA = pd.DataFrame({'MPCA':df_train.groupby(['Producto_ID','Cliente_ID','Agencia_ID'])['log_demand'].mean()}).reset_index()

#merge product, client, agencia 
submit = df_test.merge(mean_PCA,how='left',on=["Producto_ID","Cliente_ID","Agencia_ID"])
#merge product and ruta
submit = submit.merge(mean_PR,how='left',on=["Producto_ID","Ruta_SAK"])
#merge product and agencia
submit = submit.merge(mean_PA,how='left',on=["Producto_ID", "Agencia_ID"])
#merge cliente
submit = submit.merge(mean_C,how='left',on=["Cliente_ID"])
#merge product
submit = submit.merge(mean_P,how='left',on=["Producto_ID"])

# Fit an equation for predictors
submit['Pred'] = np.expm1(submit['MPCA']) * 0.7173 + np.expm1(submit['MPR']) * 0.1849 + 0.126

# Computing Nan columns
submit.loc[pd.isnull(submit['Pred']), 'Pred'] = np.expm1(submit.loc[pd.isnull(submit['Pred']), 'MPR']) * 0.741 + 0.192
submit.loc[pd.isnull(submit['Pred']), 'Pred'] = np.expm1(submit.loc[pd.isnull(submit['Pred']), 'MC']) * 0.822 + 0.855
submit.loc[pd.isnull(submit['Pred']), 'Pred'] = np.expm1(submit.loc[pd.isnull(submit['Pred']), 'MPA']) * 0.53 + 0.95
submit.loc[pd.isnull(submit['Pred']), 'Pred'] = np.expm1(submit.loc[pd.isnull(submit['Pred']), 'MP']) * 0.49 + 1
submit.loc[pd.isnull(submit['Pred']), 'Pred'] = np.expm1(mean_total) - 0.9

# Round to 4 decimal places
submit['Pred'] = submit['Pred'].round(decimals=4)

# Relabel columns ready for creating submission
submit.rename(columns={'Pred': 'Demanda_uni_equil'}, inplace=True)
submit_use = submit[['id', 'Demanda_uni_equil']]

print("Creating Submission...")
# Saving the submission into csv
submit_use.to_csv('dummy_submission.csv', index=False, columns=['id', 'Demanda_uni_equil'])