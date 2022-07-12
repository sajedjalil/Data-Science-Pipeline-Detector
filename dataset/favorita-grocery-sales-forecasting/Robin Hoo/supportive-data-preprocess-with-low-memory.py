# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import joblib, pickle

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
'''
PLEASE RUN IT ON YOUR LOCAL MACHINE WITH CSV FILES UNZIPPED
'''

# Any results you write to the current directory are saved as output.
####################################################################
#
# Modified base on the script from ArjanGroen 
# Improved the Nan column handling
# https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65/code
#
####################################################################
def reduce_mem_usage(props,prompt=True):
	nan_cols=props.columns[props.isnull().any()].tolist()
	if prompt:
		start_mem_usg = props.memory_usage().sum() / 1024**2 
		print("Memory usage of properties dataframe is :",start_mem_usg," MB")
	for col in props.columns:
		if props[col].dtype != object:  # Exclude strings
			if prompt:
				# Print current column type
				print("******************************")
				print("Column: ",col)
				print("dtype before: ",props[col].dtype)
			
			if col in nan_cols:
				if prompt: 
					print('Column: %s has NAN values'%col)
				props.loc[:,col] = props.loc[:,col].astype(np.float32)
			else:
				# make variables for Int, max and min
				IsInt = False
				mx = props[col].max()
				mn = props[col].min()
				
				# Integer does not support NA, therefore, NA needs to be filled
				

				# test if column can be converted to an integer
				asint = props[col].astype(np.int64)
				result = (props[col] - asint)
				result = result.sum()
				if result > -0.01 and result < 0.01:
					IsInt = True

				
				# Make Integer/unsigned Integer datatypes
				if IsInt:
					if mn >= 0:
						if mx < 2**8:
							props.loc[:,col] = props.loc[:,col].astype(np.uint8)
						elif mx < 2**16:
							props.loc[:,col] = props.loc[:,col].astype(np.uint16)
						elif mx < 2**32:
							props.loc[:,col] = props.loc[:,col].astype(np.uint32)
						else:
							props.loc[:,col] = props.loc[:,col].astype(np.uint64)
					else:
						if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
							props.loc[:,col] = props.loc[:,col].astype(np.int8)
						elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
							props.loc[:,col] = props.loc[:,col].astype(np.int16)
						elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
							props.loc[:,col] = props.loc[:,col].astype(np.int32)
						elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
							props.loc[:,col] = props.loc[:,col].astype(np.int64)	
				
				# Make float datatypes 32 bit
				else:
					props.loc[:,col] = props.loc[:,col].astype(np.float32)

			if prompt:
				# Print new column type
				print("dtype after: ",props[col].dtype)
				print("******************************")
	
	if prompt:
		# Print final result
		print("___MEMORY USAGE AFTER COMPLETION:___")
		mem_usg = props.memory_usage().sum() / 1024**2 
		print("Memory usage is: ",mem_usg," MB")
		print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
	return props

def encoder(data,group=False,allow_nan=False):
	results={}
	paces={}
	cols=data.columns
	for col in cols:
		results[col]=np.array(data[col])
		items=sorted(list(np.unique(results[col])))
		for item in items:
			results[col][results[col]==item]=items.index(item)
		if len(cols)<2:
			return results[col]
		paces[col]=len(items)
	if group:
		paces=sorted(paces.items(), key=lambda d: d[1])
		if allow_nan:
			col,stp=paces[0]
			results[col]=results[col]+1
			paces[0]=(col,stp+1)
		ary=np.zeros(len(data),dtype=np.int)
		factor=1
		for i in range(len(paces)):
			col,stp=paces[i]
			ary=ary+factor*results[col]
			factor*=stp
		return ary
	return results.values()

def dump(obj,file_name,level=6):
	joblib.dump(value=obj,filename=file_name,compress=level,protocol=pickle.HIGHEST_PROTOCOL)
	return

def load(file_name):
	return joblib.load(file_name)

MIN_DATE='2012-03-02'
MAX_DATE='2017-12-26'
dump_fold=''
read_fold='../input/'
print('Building Calendar Dataframe...')
data=pd.DataFrame(data={'date':pd.date_range(MIN_DATE,MAX_DATE)})
data['year']=data['date'].dt.year
data['quarter']=data['date'].dt.quarter
data['month']=data['date'].dt.month
data['day']=data['date'].dt.day
data['dow']=data['date'].dt.weekday
data['woy']=data['date'].dt.weekofyear
data['wom']=(data['day']-data['dow']+6)//7
data['doy']=data['date'].dt.dayofyear
data['day_idx']=data.index
data['date']=data['date'].astype(str)
data=reduce_mem_usage(data)
print('Building Calendar Dataframe Done!\n')
print('Dumpping Calendar Dataframe...')
dump(data,'%scalendar.dataframe.gz'%dump_fold)
print('Dumpping Calendar Dataframe Done!')

print('Loading initial data dataframe...')
print('loading items data..')
data = pd.read_csv('%sitems.csv'%read_fold)
data['family'],data['class']=encoder(data[['family','class']])
print('Loading full initial dataframe done!')
data=reduce_mem_usage(data)

print('Dumpping Products Dataframe...')
dump(data,'%sproducts.dataframe.gz'%dump_fold)
print('Dumpping Products Dataframe Done!')

print('Loading initial data dataframe...')
print('loading stores data..')
data = pd.read_csv('%sstores.csv'%read_fold)
data['type'],data['city'],data['state']=encoder(data[['type','city','state']])
data.rename(columns={'type':'store_type'},inplace=True)
print('Loading full initial dataframe done!')
data=reduce_mem_usage(data)

print('Dumpping Stores Dataframe...')
dump(data,'%sstores.dataframe.gz'%dump_fold)
print('Dumpping Stores Dataframe Done!')

print('Loading initial data dataframe...')
print('loading transactions data..')
data = pd.read_csv('%stransactions.csv'%read_fold)
data['date']=data['date'].astype(str)
data=reduce_mem_usage(data.merge(load('%scalendar.dataframe.gz'%dump_fold)[['date','day_idx']],how='left',on='date').drop(['date'],axis=1).rename(columns={'day_idx':'date'}))
print('packing data...')
print('Loading initial dataframe done!')

print('Dumpping Data Initial Dataframe...')
dump(data,'%stransactions.dataframe.gz'%dump_fold)
print('Dumpping Data Initial Dataframe Done!')

print('Building Calendar Dataframe...')
data=pd.read_csv('%soil.csv'%read_fold)
data=reduce_mem_usage(data[data['dcoilwtico'].notnull()].merge(load('%scalendar.dataframe.gz'%dump_fold)[['date','day_idx']],how='left',on='date').drop('date',axis=1).rename(columns={'day_idx':'date'}))
print('Building Oil Price Dataframe Done!\n')

print('Dumpping Oil Price Dataframe...')
dump(data,'%soil.dataframe.gz'%dump_fold)
print('Dumpping Oil Price Dataframe Done!')

print('Loading initial data dataframe...')
print('loading holidays_events data..')
data = pd.read_csv('%sholidays_events.csv'%read_fold)
data['date']=data['date'].astype(str)
data.loc[(data['transferred'].isnull())|(data['transferred']==False),'transferred']=0
data.loc[data['transferred']==True,'transferred']=1
data.rename(columns={'type':'holiday_type'},inplace=True)
print('Loading full initial dataframe done!')

stores = pd.read_csv('%sstores.csv'%read_fold)[['store_nbr','city','state']]
holidays = pd.concat([data[data['locale']=='Local'].merge(stores,how='left',left_on='locale_name',right_on='city'),
                      data[data['locale']=='Regional'].merge(stores,how='left',left_on='locale_name',right_on='state')]).drop(['locale_name','city','state'],axis=1)


stores['locale']='National'
holidays=pd.concat([holidays,data[data['locale']=='National'].merge(stores[['store_nbr','locale']],how='inner',on='locale').drop(['locale_name'],axis=1)])


holidays['holiday']=encoder(holidays[['transferred','holiday_type','locale','description']],group=True,allow_nan=True)
holidays.drop(['holiday_type','locale','description','transferred'],axis=1,inplace=True)
holidays=reduce_mem_usage(holidays.merge(load('%scalendar.dataframe.gz'%dump_fold)[['date','day_idx']],how='left',on='date').drop('date',axis=1).rename(columns={'day_idx':'date'}))

print('Dumpping Holidays Dataframe...')
dump(holidays,'%sholidays.dataframe.gz'%dump_fold)
print('Dumpping Holidays Dataframe Done!')

print('loading train,test data..')
data = pd.concat([pd.read_csv('%strain.csv'%read_fold),pd.read_csv('%stest.csv'%read_fold)])
data['unique']=data['item_nbr']*(data['store_nbr'].max()+1)+data['store_nbr']
data['date']=data['date'].astype(str)
data.loc[(data['onpromotion'].isnull())|(data['onpromotion']==False),'onpromotion']=0
data.loc[data['onpromotion']==True,'onpromotion']=1
print('Loading full initial dataframe done!')
data=data.merge(load('%sproducts.dataframe.gz'%dump_fold),how='left',on='item_nbr')
print('products merging is done!')
data = reduce_mem_usage(data)

print('Building data dataframe...')
print('Encapuslating full initial dataframe...')
data=data.merge(load('%scalendar.dataframe.gz'%dump_fold),how='left',on='date').drop('date',axis=1).rename(columns={'day_idx':'date'})
print('calendar merging is done!')
data=data.merge(load('%sholidays.dataframe.gz'%dump_fold),how='left',on=['store_nbr','date'])
data.loc[data['holiday'].isnull(),'holiday']=0
print('holidays merging is done!')
data=data.merge(load('%stransactions.dataframe.gz'%dump_fold),how='left',on=['store_nbr','date'])
print('transactions merging is done!')
data=data.merge(load('%sstores.dataframe.gz'%dump_fold),how='left',on='store_nbr')
print('stores merging is done!')
data=data.merge(load('%soil.dataframe.gz'%dump_fold),how='left',on='date')
print('oil merging is done!')
print('Encapuslating full initial dataframe done!')
data = reduce_mem_usage(data)
print("Memory usage of dataframe is :%f MB"%(1.0*data.memory_usage().sum()/1024**2))   
print('Building data dataframe done!')
print('Dumpping Initial Data Dataframe...')
dump(data,'%sdata.initial.dataframe.gz'%dump_fold)
print('Dumpping Initial Data Dataframe Done!\n\n')