#2. hét hétfő 1. feladat vázlat @13:15
import pandas as pd
import numpy as np
import re, time
start_cpu = time.clock()
start_real = time.time()

print("Reading data")
train = pd.read_csv('../input/train.csv',usecols=['Cliente_ID','Producto_ID','Agencia_ID','Ruta_SAK','Demanda_uni_equil'],nrows=3000000)
test  = pd.read_csv('../input/test.csv' ,usecols=['Cliente_ID','Producto_ID','Agencia_ID','Ruta_SAK'],nrows=300)

print("Computing means")
#transform target variable to log(1 + demand) - this makes sense since we're 
#trying to minimize rmsle and the mean minimizes rmse:
train['log_demand']=np.log1p(train['Demanda_uni_equil'])

mean_total=train['log_demand'].mean() #overall mean
print(mean_total)
#mean by product
mean_P = train.groupby(['Producto_ID'],as_index=False).log_demand.mean().rename(columns={'log_demand':'MP'})              #.loc[:,'log_demand']
#mean by product and ruta
mean_PR = train.groupby(['Producto_ID','Ruta_SAK'],as_index=False).log_demand.mean().rename(columns={'log_demand':'MPR'}) #.loc[:,'log_demand']
#mean by product, client, agencia
mean_PCA = train.groupby(['Producto_ID','Cliente_ID','Agencia_ID'],as_index=False).log_demand.mean().rename(columns={'log_demand':'MPCA'}) #.loc[:,'log_demand']

print("Merging means with test set")
submit = pd.merge(test,mean_PCA,how='left',on=['Producto_ID', 'Cliente_ID', 'Agencia_ID'])
submit = pd.merge(submit,mean_PR,how='left',on=['Producto_ID', 'Ruta_SAK'])
submit = pd.merge(submit,mean_P,how='left',on=['Producto_ID'])
# Now create Predictions column
#submit['Pred']=np.expm1(submit['MPCA'])*0.826+0.45
def seged0(row):
    return np.expm1(row['MPCA'])*0.826+0.45

submit['Pred'] = submit.apply(lambda row:seged0(row),axis=1)
#print (submit[pd.isnull(submit.Pred)].loc[:,'MPR'])
#submit[pd.isnull(submit.Pred)]=submit[pd.isnull(submit.Pred)].loc[:,'MPR']
def seged1(row):
	if pd.notnull(row['Pred']):
		return row['Pred']
	else:
		return np.expm1(row['MPR'])*0.79+0.09
submit['Pred'] = submit.apply(lambda row:seged1(row),axis=1)
def seged2(row):
	if pd.notnull(row['Pred']):
		return row['Pred']
	else:
		return np.expm1(row['MP'])*1.05+0.6
submit['Pred'] = submit.apply(lambda row:seged2(row),axis=1)
def seged3(row):
	if pd.notnull(row['Pred']):
		return row['Pred']
	else:
	    return np.expm1(mean_total)+1.15
submit['Pred'] = submit.apply(lambda row:seged3(row),axis=1)
print(submit)
print("Write out submission file")
submit['id']=submit.index
submit=submit.rename(columns={'Pred':'Demanda_uni_equil'})
submit=submit[['id','Demanda_uni_equil']]
submit=submit.set_index('id')
submit.to_csv('submit_mean_by_Agency_Ruta_Client.csv')
print("Done!")

elapsed_cpu = time.clock() - start_cpu
elapsed_real = time.time() - start_real
print('CPU idő:%s, Valós idő:%s'%(elapsed_cpu,elapsed_real) )