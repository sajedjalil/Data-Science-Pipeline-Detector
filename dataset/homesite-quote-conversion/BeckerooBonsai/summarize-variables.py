'''
Author : Rebecca Jones

This summarizes some basic features of each variable and exports the summary to an easy to read csv file.

	Features summarized are:
		- type of variable (int,float,object)
		- count of distinct levels for each variable
		- count of distinct levels containing at least 0.1% of the data
		- count of distinct levels containing at least 1% of the data
		- max value observed (for numeric variables)
		- min value observed (for numeric variables)
		- mean value observed (for numeric variables)
		- percent of missing data 
		- rate of quote conversion for group of missing data 
		- max quote conversion rate among all groups with at least 1% of the data
		- min quote conversion rate among all groups with at least 1% of the data
		- values of levels for top two groups based on group size
		- values of levels for top two groups based on quote conversion rate
		- percent of observations at the top level (same as mode of variable)
		- mean of variable for observations where quote bound (for numeric variables)
		- mean of variable for observations where quote did not bind (for numeric variables)

'''



import pandas as pd
import numpy as np

fileIn = '../input/train.csv'
fileOut = 'variableSummary.csv'

####################################################
# import data to dataframe and do basic cleaning
####################################################

data = pd.read_csv(fileIn)
data = data.fillna(-1)

data = data.drop('QuoteNumber', axis=1)
data['Date'] = pd.to_datetime(pd.Series(data['Original_Quote_Date']))
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['dayofweek'] = data['Date'].dt.dayofweek
data['IsWeekend'] = data.dayofweek > 4
data['Quarter'] = data['Date'].dt.quarter
data = data.drop('Original_Quote_Date', axis=1)
data = data.drop('Date', axis=1)

#######################################################

totalObs = len(data)
min_group_size = 0.001*totalObs # minimum group size with at least 0.1% of observations
gen_group_size = 0.01*totalObs # group size with at least 1% of observations



varSummary = []
varNames = []

for colName in data.columns[1:]:

	typeOfData = data[colName].dtype#.name
	numLevels =  len(data[colName].unique())
	listLevels = np.array(list(data[colName].unique()))
	listLvlsObsCnt = np.array(list(data[colName].value_counts()))
	LvlsObsCntNoLowCnt = sum(listLvlsObsCnt > min_group_size)
	LvlsObsCntGroups = sum(listLvlsObsCnt > gen_group_size)
	maxValue = max(listLevels) if typeOfData !='O' else 'NA'
	minValue = min(listLevels) if typeOfData !='O' else 'NA'
	meanValue = data[colName].mean() if typeOfData !='O' else 'NA'	
	if (typeOfData!='O'):
		if len(listLvlsObsCnt[listLevels==-1])>0 :
			percentMissing = listLvlsObsCnt[listLevels==-1][0]/totalObs
			missingGrpBindRate = data['QuoteConversion_Flag'][data[colName]==-1].mean()
		else:
			percentMissing = 0
			missingGrpBindRate = 'NA'
	elif '-1' in list(listLevels):
		percentMissing = listLvlsObsCnt[listLevels=='-1'][0]/totalObs
		missingGrpBindRate = data['QuoteConversion_Flag'][data[colName]=='-1'].mean()
	else:
		percentMissing = 0
		missingGrpBindRate = 'NA'  

	selectVars = list(listLevels[listLvlsObsCnt>gen_group_size])
	if selectVars==[]:
		temp = data[[colName,'QuoteConversion_Flag']]
	else:
		temp = data[[colName,'QuoteConversion_Flag']][data[colName].isin(selectVars)]
	temp2 = temp.groupby([colName]).QuoteConversion_Flag.mean()
	minGrpBindRate = min(temp2.values)
	maxGrpBindRate = max(temp2.values)

	percentAtTopLvl = listLvlsObsCnt[0]/totalObs
	if len(listLevels)>1:
		top2lvlsBySize = list(listLevels[:2]) #" / ".join(str(e) for e in listLevels[:2]) 
		top2lvlsByBindRate = list(temp2.sort_values(ascending=False).index[:2])
		
	if (typeOfData!='O'):
		meanOfBound = data[colName][((data['QuoteConversion_Flag']==1)&(data[colName]!=-1))].mean()
		meanOfUnBound = data[colName][((data['QuoteConversion_Flag']==0)&(data[colName]!=-1))].mean()
	else:
		meanOfBound = 'NA'
		meanOfUnBound = 'NA'
		

	d = { 'typeOfData' : typeOfData, 
		   'numLevels' : numLevels, 
		   'numLvlsNoLowCT' : LvlsObsCntNoLowCnt, 
		   'numLvlsBigGrps' : LvlsObsCntGroups,
		   'maxValue' : maxValue, 
		   'minValue' : minValue, 
		   'meanValue' : meanValue,
		   'pctMissing' : percentMissing, 
		   'missgBindRate' : missingGrpBindRate , 
		   'top2bySize' : top2lvlsBySize,
		   'top2byBindRate' : top2lvlsByBindRate, 
		   'percentAtTopLvl' : percentAtTopLvl,  
		   'minGrpBindRate' : minGrpBindRate, 
		   'maxGrpBindRate' : maxGrpBindRate, 
		   'meanOfBound' : meanOfBound, 
		   'meanOfUnBound' : meanOfUnBound}
	d = pd.Series(d, index = [ 'typeOfData' , 'numLevels' , 'numLvlsNoLowCT' , 'numLvlsBigGrps' ,
								'maxValue', 'minValue' , 'meanValue',
								'pctMissing' , 'missgBindRate' ,
								'top2bySize', 'top2byBindRate' , 'percentAtTopLvl',
								'minGrpBindRate','maxGrpBindRate', 'meanOfBound' ,'meanOfUnBound' ])
	varSummary = varSummary + [ d ]
	varNames = varNames + [colName]
	

summaryFrame = pd.DataFrame( varSummary , index = varNames)
		
summaryFrame.to_csv(fileOut)		