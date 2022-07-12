# %% [code]
import warnings
warnings.filterwarnings('ignore')#Ignoring warnings

import pandas as pd #importing pandas to handle data via DataFrames
import numpy as np #importing numpy for numerical functionality
import matplotlib.pyplot as plt #importing matplotlib.pyplot for plotting functionality

# %% [code]
def enterData(df,columnName,originalData,placeList):
    for place in placeList:
        provinceState=originalData["Province_State"]#Defining our province/state
        countryRegion=originalData["Country_Region"]#Or our country/region, for use if province/state is unavailable
        
        #Checking if the data entry has a provincial entry, and making a selection on the appropriate location type
        if(place in provinceState.values):
            #Selecting data for province
            selection=originalData[provinceState==place]
        
        elif(place in countryRegion.values):
            #Selecting data for country
            preselection=originalData[countryRegion==place]
            selection=preselection[preselection["Province_State"]==0]#We select the country entry where province state is 0
        
        
        #Writing values to respective dataframes, by country
        df[place]=selection[columnName].values
        
        #Setting date as the index
        df.set_index(selection["Date"],inplace=True)

# %% [code] {"scrolled":false}
def WeekSplit(daysInWeek,df):

    dataWeeks=np.int(len(casesTrain)/daysInWeek) #70/7=10 weeks worth of data
    df["Week"]=0 #Setting up a week column for the dataframe
    colLength=len(df["Week"])#Defining length of the week column
    
    for week in range(dataWeeks):#Iterating over the weeks worth of data
        for iteration in range(daysInWeek):#For each day
            df["Week"][week*daysInWeek+iteration]=week#We assign a number to the day, depending on what week it is in
      
    return dataWeeks#Returning the number of weeks worth of data

# %% [code] {"scrolled":false}
def MinMaxScaler(df):
    if("Week" in df.columns):#Checking to see if the dataframe has a week column
        
        columns=df.columns.drop("Week")#Defining country column names (without week)
    else:
        columns=df.columns#Otherwise there is no need to drop anything
        
    for column in columns:#Iterating over columns
        
        dataCol=df[column]#Getting each column
        
        colMax=dataCol.max()#Maximum value of column
        
        #If our maximum value is greater than 0, then we scale it
        if(colMax>0):
            dataCol=dataCol/colMax#Performing scaling
        
            df[column]=dataCol#Overwriting to dataFrame

# %% [code] {"scrolled":false}
#Split data by country/week
def Split(country,week,df):
    
    byCountry=df[[country,"Week"]]#Splitting off a country's column, along with the week column
    
    byWeek=byCountry[byCountry["Week"]==week]#Splitting by week
    
    byWeek.drop(columns="Week",inplace=True) #Dropping the now unnecessary week column
    
    return byWeek #Returning the dataframe

# %% [code]
def Pipeline(caseOrFatal,numberOfSamples,originalData,countryList,normalise=True,weekSplit=True):
    df=pd.DataFrame()#Creating placeholder dataframe

    #Entering Data
    enterData(df,caseOrFatal,originalData,countryList)
    
    #Splitting by week(optional)
    if(weekSplit):
        dataWeeks=WeekSplit(numberOfSamples,df)
    
    #Normalising(optional)
    if(normalise):
        MinMaxScaler(df)
    
    return df#Returns a dataframe

# %% [code] {"scrolled":false}
from sklearn.linear_model import LinearRegression
linReg=LinearRegression()

# %% [code] {"scrolled":false}
data = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")#defining a path via os and reading in data
test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")#defining a path via os and reading in data

#Distinguishing between the two Georgias for both dataframes
data["Country_Region"].replace({"Georgia":"GeorgiaC"},inplace=True)
test["Country_Region"].replace({"Georgia":"GeorgiaC"},inplace=True)

# %% [code] {"scrolled":false}
#Filling null values with zeros
data["Province_State"].fillna(value=0,inplace=True);
test["Province_State"].fillna(value=0,inplace=True);

# %% [code] {"scrolled":false}
testentries=len(test)
dataentries=len(data)
#Declaring empty arrays for unique dates

testdates=[]
datadates=[]
#Extracting Test dates
for entry in range(testentries):
    dataEntry=test.iloc[entry,:]
    date=dataEntry["Date"]
    if((date not in testdates)):
        testdates.append(date)
#Extracting Train dates
for entry in range(dataentries):
    dataEntry=data.iloc[entry,:]
    date=dataEntry["Date"]
    if((date not in datadates)):
        datadates.append(date)
        
#Defining these dates as indices
datadates=pd.Index(datadates)
testdates=pd.Index(testdates)

#Finding their intersection i.e. dates we must extract from the overall prediction table we're going to generate
dateIntersect=list(set(testdates)&set(datadates))

# %% [code] {"scrolled":false}
#Next, we want a list of countries/provinces to pass to the pipeline. First we require a list of these column names:
#We extract the data for a single date, as we confirmed previously that each date has an entry for every unique place name
march31=data[data["Date"]=="2020-03-31"]

#Extracting unique province names
uniqueProvinces=march31["Province_State"].unique()[1:]#0 will be the first entry, so we select all but the first entry
colNames=uniqueProvinces#Storing uniquenames in a list called colnames

#Deleting the provinces already collected, from the overall dataframe, to reveal our country rows

deleteProvinceRows=march31[~march31["Province_State"].isin(uniqueProvinces)]#Deleting the unecessary province rows

colNames=np.append(colNames,deleteProvinceRows["Country_Region"].values);#Appending the remaining dataframe

colNames=list(colNames)#Converting to list

# %% [code] {"scrolled":false}
#The split of 7 is now arbitrary; we're no longer relying on week numbers. This is why we turn off weekSplit as well.
dataCases=Pipeline("ConfirmedCases",7,data,colNames,normalise=False,weekSplit=False)
dataFatal=Pipeline("Fatalities",7,data,colNames,normalise=False,weekSplit=False)

# %% [code] {"scrolled":false}
#Getting maximum values of each column i.e. last value of each column
maxCases=dataCases.iloc[-1,:]
maxFatal=dataFatal.iloc[-1,:]

MinMaxScaler(dataCases)
MinMaxScaler(dataFatal)

# %% [code] {"scrolled":true}
#Non-intersecting dates
nonIntersectTrainDates=list(set(datadates)-set(dateIntersect))
#Sorting chronologically
nonIntersectTrainDates.sort()

#Making a copy of the full training set, just in case we need it for comparison purposes
trainCases=dataCases.copy()
trainFatal=dataFatal.copy()


#Extracting those non-intersecting date for predictions from the original training set
dataCases=dataCases.loc[nonIntersectTrainDates]
dataFatal=dataFatal.loc[nonIntersectTrainDates]

# %% [code] {"scrolled":false}
def Predict(df,place,model,iterations,predPoints,daysPerIteration,clampValue=1.5):#Defining our default clamp as 1.5 
    #predPoints is our number of samples we need to take for prediction
    #days per iteration is the number we increase by each iteration, when selecting our index for predictions
    #These two values are exactly the same, but we have labelled them differently to distinguish their purposes in the function
    
    placeCol=df[place]#Extracting column by place
    
    for days in range(iterations):#Iterations is the number of days we wish to predict
        
        #Counting days from the bottom of the list, and selecting the number of daysPerIteration, with an offset of predicted points
        X=pd.DataFrame(placeCol[-(predPoints+daysPerIteration):-daysPerIteration])
        #for the y labels, we simply select the last days
        y=pd.DataFrame(placeCol[-daysPerIteration:])
        
        #Fitting the model
        model.fit(X,y)
        
        #Defining the new X for predicting the future data
        yPred=pd.DataFrame(y.iloc[-1,:])
        
        #Future data
        new=model.predict(yPred)
        
        #Clamp function
        if(new>yPred.values*clampValue): #Upper Limit
            new=yPred.values*clampValue
        elif(new<=yPred.values): #Lower Limit
            new=yPred.values
            
        #Storing the new value, in dataframe format
        new=pd.DataFrame(new)
        
        #Appending the value to the column
        placeCol=placeCol.append(new)
        
        
    return placeCol#Returns the pandas column

# %% [code] {"scrolled":false}
def PredictAll(df,predictionIncrement=5,clampValue=1): #Predictions made in 5 day increments, for increased accuracy
    
    daysToPredict=len(testdates)#Days to predict is given by the length of our test column
    allPredictions=pd.DataFrame()#Empty dataframe for predictions
    
    #Making predictions for each country
    for place in colNames:
        predictions=Predict(df,place,linReg,daysToPredict,predictionIncrement,predictionIncrement,clampValue)
        allPredictions=pd.concat([allPredictions,predictions],axis=1)
    
    
    #Removing the unecessary early dat
    earlyDatesLength=len(datadates)-len(dateIntersect)
    allPredictions=allPredictions.iloc[earlyDatesLength:,:]
    
    #Shortening the prediction number to the appropriate length for the test columns, in case too many were made
    while len(allPredictions)>len(testdates):
        allPredictions=allPredictions.iloc[:-1,:]
    
    #Renaming columns as integers, to give them all different names that will compute quickly
    allPredictions.columns=range(len(allPredictions.columns))
    
    #Setting the index to be the dates
    allPredictions.index=testdates
    return allPredictions

# %% [code]
#Setting clamp value to 1.1
finalCases=PredictAll(dataCases,predictionIncrement=5,clampValue=1.1)
finalFatal=PredictAll(dataFatal,predictionIncrement=5,clampValue=1.1)

# %% [code]
#De-normalising
finalCases=finalCases*maxCases.values
finalFatal=finalFatal*maxFatal.values

#Assigning column names
finalCases.columns=finalFatal.columns=colNames

# %% [code]
#Empty columns
test["ConfirmedCases"]=0
test["Fatalities"]=0

#Adding confirmed cases predictions column to test data
for country in colNames:#Iterating over countries
    predColumn=finalCases[country]
    for date in testdates:#And iterating over dates
        
        entry=predColumn.loc[date]#Getting the value we want to insert
        
        #Booleans dataframes for searching
        countrySearch=test["Country_Region"]==country
        dateSearch=test["Date"]==date
        
        #Searching for the index of the correct country/date entry
        testIndex=test[countrySearch][dateSearch].index
        
        #Setting the indexed entry to the value we want to insert
        test["ConfirmedCases"].loc[testIndex]=entry
        
#Repeating for fatalities
for country in colNames:
    predColumn=finalFatal[country]
    for date in testdates:
        entry=predColumn.loc[date]
        
        countrySearch=test["Country_Region"]==country
        dateSearch=test["Date"]==date
        
        testIndex=test[countrySearch][dateSearch].index
        
        test["Fatalities"].loc[testIndex]=entry

# %% [code]
#Creating the submission in the correct format
submission=test.drop(columns=["Province_State","Country_Region","Date"])

# %% [code]
#Printing some rows of the submission dataframe, to ensure we have the correct format
submission

# %% [code]
#Saving as a .csv file, without the index
submission.to_csv("submission.csv",index=False)