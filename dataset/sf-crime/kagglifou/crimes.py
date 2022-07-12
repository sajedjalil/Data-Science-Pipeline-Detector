import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

#print(train["DayOfWeek"].unique()) # Convertir directement en classes numeriques
train.loc[train["DayOfWeek"] == "Monday", "DayOfWeek"] = 0
train.loc[train["DayOfWeek"] == "Tuesday", "DayOfWeek"] = 1
train.loc[train["DayOfWeek"] == "Wednesday", "DayOfWeek"] = 2
train.loc[train["DayOfWeek"] == "Thursday", "DayOfWeek"] = 3
train.loc[train["DayOfWeek"] == "Friday", "DayOfWeek"] = 4
train.loc[train["DayOfWeek"] == "Saturday", "DayOfWeek"] = 5
train.loc[train["DayOfWeek"] == "Sunday", "DayOfWeek"] = 6
#print(train["DayOfWeek"].unique()) # Convertir directement en classes numeriques

#print(train["PdDistrict"].unique()) # Convertir directement en classes numeriques 
# + Verifier qu'il n'y a pas de donnees abscentes !


#according to https://en.wikipedia.org/wiki/White-collar_crime#Blue-collar_crime
white_crime=["FRAUD", "FORGERY/COUNTERFEITING", "BAD CHECKS" , "EXTORTION", "EMBEZZLEMENT", "SUSPICIOUS OCC",
              "BRIBERY"]

blue_crime=["VANDALISM", "LARCENY/THEFT", "STOLEN PROPERTY", "ROBBERY", "DRIVING UNDER THE INFLUENCE",
             "DISORDERLY CONDUCT", "LIQUOR LAWS", "VEHICLE THEFT", "ASSAULT", "KIDNAPPING", "TRESPASS", 
             "ARSON", "RECOVERED VEHICLE", "BURGLARY" , "PROSTITUTION"]
             
other_crime=["MISSING PERSON", "RUNAWAY", "FAMILY OFFENSES", "SEX OFFENSES NON FORCIBLE",
             "PORNOGRAPHY/OBSCENE MAT", "WEAPON LAWS", "DRUNKENNESS", "SUICIDE", "TREA",
             "DRUG/NARCOTIC", "SEX OFFENSES FORCIBLE",  "LOITERING","GAMBLING","SECONDARY CODES",
             "WARRANTS", "OTHER OFFENSES" ,"NON-CRIMINAL" ]
             
             

# On regroupe les donnees par nouvelles categoriee 0,1,2 pour white collar crime, blue collar crime et other crime
train.loc[train["Category"].isin(white_crime) == True , "Category"] = 0 #"White_collar"
train.loc[train["Category"].isin(blue_crime) == True , "Category"] = 1 #"Blue_collar"
train.loc[train["Category"].isin(other_crime) == True , "Category"] = 2 #"Other_crimes"
train["Category"]= train["Category"].fillna(2)
train["DayOfWeek"]= train["DayOfWeek"].fillna(0)

#print(pd.value_counts(train["Category"]))

# A function to get the title from a name.
def get_hour_of_day(date):
    # Use a regular expression to search for the hour of the day
    search = int(date[11:13]) #re.search(' ([A-Za-z]+)\.', date)
    # If the title exists, extract and return it.
    if search:
        return search #.group(1)
    return 12

# Get all the titles and print how often each one occurs.
hours = train["Dates"].apply(get_hour_of_day)
"""
hours [hours <= 10 ] = 0 #Morning
hours[(hours > 10) &  (hours<= 14) ] = 1 # Noon
hours[(hours > 14) & (hours <= 23) ] = 0  #After noon
hours[hours > 23] = 2  # late at night
"""
train["Dates"] = hours

#print(pd.value_counts(train['Dates']))
#

train.loc[train['Resolution'] == "NONE" , 'Resolution'] = 0 # No resolution
train.loc[train['Resolution'] == "ARREST, BOOKED"  , 'Resolution'] = 1 # No resolution
train.loc[train['Resolution'] == "ARREST, CITED" , 'Resolution'] = 2 # No resolution
train.loc[ (train['Resolution'] != 0) & (train['Resolution'] != 1) & (train['Resolution'] != 2) , 'Resolution'] = 3 # No resolution


pd_district = { "SOUTHERN":1,"MISSION":2, "NORTHERN":3,"BAYVIEW":4,"CENTRAL":5,"TENDERLOIN":6,"INGLESIDE":7,"TARAVAL":8,"PARK":9,"RICHMOND":10}

for pdst,v in pd_district.items() :
    train.loc[train['PdDistrict'] == pdst , 'PdDistrict'] = v 


descript = { "GRAND THEFT FROM LOCKED AUTO":0,
"LOST PROPERTY":1,
"BATTERY":2,
"STOLEN AUTOMOBILE":3,
"DRIVERS LICENSE, SUSPENDED OR REVOKED":4,
"WARRANT ARREST":5,
"SUSPICIOUS OCCURRENCE":6,
"AIDED CASE, MENTAL DISTURBED":7,
"PETTY THEFT FROM LOCKED AUTO":8,
"MALICIOUS MISCHIEF, VANDALISM OF VEHICLES":9,
"TRAFFIC VIOLATION":10,
"PETTY THEFT OF PROPERTY":11,
"MALICIOUS MISCHIEF, VANDALISM":12,
"THREATS AGAINST LIFE":13,
"FOUND PROPERTY":14,
"ENROUTE TO OUTSIDE JURISDICTION":15,
"GRAND THEFT OF PROPERTY":16,
"POSSESSION OF NARCOTICS PARAPHERNALIA":17,
"PETTY THEFT FROM A BUILDING":18 }

train.loc[train['Descript'].isin(descript) == False , 'Descript'] = 19 # Other descriptions

for dscrt,v in descript.items() :
    train.loc[train['Descript'] == dscrt , 'Descript'] = v 

address={
"800 Block of BRYANT ST":0,
"800 Block of MARKET ST":1,
"2000 Block of MISSION ST":2,
"1000 Block of POTRERO AV":3,
"900 Block of MARKET ST":4,
"0 Block of TURK ST":5,
"0 Block of 6TH ST":6,
"300 Block of ELLIS ST":7,
"400 Block of ELLIS ST":8,
"16TH ST / MISSION ST":9,
"1000 Block of MARKET ST":10,
"1100 Block of MARKET ST":11,
"2000 Block of MARKET ST":12,
"100 Block of OFARRELL ST":13,
"700 Block of MARKET ST":14,
"3200 Block of 20TH AV":15,
"100 Block of 6TH ST":16,
"500 Block of JOHNFKENNEDY DR":17,
"TURK ST / TAYLOR ST":18,
"200 Block of TURK ST":19,
"0 Block of PHELAN AV":20,
"0 Block of UNITEDNATIONS PZ":21,
"0 Block of POWELL ST":22,
"100 Block of EDDY ST":23,
"1400 Block of PHELPS ST":24,
"300 Block of EDDY ST":25,
"100 Block of GOLDEN GATE AV":26,
"100 Block of POWELL ST":27,
"200 Block of INTERSTATE80 HY":28,
"MISSION ST / 16TH ST":29 
}

train.loc[train['Address'].isin(address) == False , 'Address'] = 30 # Other descriptions

for add,v in address.items() :
    train.loc[train['Address'] == add , 'Address'] = v 

predictors = ["DayOfWeek","X","Y", "Resolution","PdDistrict","Dates",'Descript','Address']
#predictors = ["DayOfWeek","Y", "Resolution","PdDistrict","Dates",'Descript','Address']

# Score Regression logistique faible : 66% random state = 1
# Learning on train set
alg = RandomForestClassifier(random_state=1, n_estimators=30, min_samples_split=6, min_samples_leaf=3)
#alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
#scores = cross_validation.cross_val_score(alg, train[predictors], train["Category"], cv=5)
#print(scores.mean())

from sklearn.cross_validation import KFold
kf = KFold(train.shape[0], n_folds=5, random_state=1)

predictions = []
for trainn, testt in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (train[predictors].iloc[trainn,:])
    # The target we're using to train the algorithm.
    train_target = train["Category"].iloc[trainn]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(train[predictors].iloc[testt,:])
    predictions.append(test_predictions)

print(predictions)
import numpy as np

predictions = np.concatenate(predictions, axis=0)
accuracy=0

for i in range(len(train)):
        if(predictions[i]==train["Category"].iloc[i]):
            accuracy+=1
accuracy =accuracy / len(predictions)

print(accuracy)
