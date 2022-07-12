import pandas as pd
from pandas import Series,DataFrame
from collections import Counter
from pandas import Series

#%pylab inline

trainPath = r'../input/train.csv'
testPath = r'../input/test.csv'

train_df = pd.read_csv(trainPath)
test_df = pd.read_csv(testPath)

#deal character data
train_df['Year']  = train_df['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
train_df['Month'] = train_df['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
train_df['Week']  = train_df['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))

test_df['Year']  = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
test_df['Month'] = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
test_df['Week']  = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))

train_df.drop(['Original_Quote_Date'], axis=1,inplace=True)
test_df.drop(['Original_Quote_Date'], axis=1,inplace=True)

train_x_df = train_df.drop("QuoteConversion_Flag",axis=1)
train_y_df = train_df["QuoteConversion_Flag"]

#get character data columns
notNumTypeCol = [col for col in train_x_df.columns if train_x_df[col].dtype == 'O']

notNumValueCounter = {}
for key in notNumTypeCol:
    notNumValueCounter[key] = Counter(train_x_df[key])
    
def charToInt(data_df,convDict):
    #create map dict
    for colName in convDict.keys():
        valuesCounter = convDict[colName]
        valuesCounterSize = len(valuesCounter)
        mapDict = dict(zip(valuesCounter.keys(),range(valuesCounterSize)))
        
        temp = []
        s = data_df[colName]
        colValueSize = len(s)
        
        for i in range(colValueSize):
            temp.append(mapDict[s[i]])
        data_df[colName] = Series(temp)
        
charToInt(train_x_df,notNumValueCounter)
        
def charToIntAndFillNAN(data_df,charColValueCounter,colValueCounter):
    for col in data_df.columns:
        charKeys = charColValueCounter.keys()
        if col in charKeys:
            currentColValCount = charColValueCounter[col]
            currentCountSize = len(currentColValCount)
            mapDict = dict(zip(charKeys,range(currentCountSize)))
            
            temp = []
            s = data_df[col]
            recordsLen = len(s)
            for i in range(recordsLen):
                temp.append(mapDict[s[i]])
            data_df[col] = Series(temp)
            
            nanFillValue,count = colValueCounter[col].most_commom()(1)[0]
            data_df[col].fillna(nanFillValue,inplace = True)
        else:
            nanFillValue,count = colValueCounter[col].most_commom()(1)[0]
            data_df[col].fillna(nanFillValue,inplace = True)

# colValueCounterfor = {}
# for col in train_x_df.columns:
#     colValueCounterfor[col] = Counter(train_x_df[col])

# nanFillDict = {}
# for key,value in  colValueCounterfor.items():
#     value,counter = value.most_commom()(1)[0]
#     nanFillDict[key] = value

nanFillDict = {}
for col in train_x_df.columns:
    counter =train_x_df[col].value_counts()
    nanFillDict[col] = counter.index[0]

train_x_df.fillna(nanFillDict)

#charToIntAndFillNAN(train_x_df,notNumValueCounter,colValueCounterfor)

#train_x_df[:,]

colMinMax = {}

for col in train_x_df.colunms:
    colMinMax[col] = tuple((train_x_df[col].min,train_x_df[col].max))
    
size = len(train_x_df)    
for col in train_x_df.columns:
    minVal,maxVal = colMinMax[col]
    if minVal != maxVal:
        train_x_df[col] = (train_x_df[col] - Series(size * [minVal]))/(Series(size * [maxVal] - Series(size * [minVal])