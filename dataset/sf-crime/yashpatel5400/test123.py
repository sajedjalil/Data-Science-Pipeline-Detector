"""
Author: Yash Patel
Name: SFCrime.py
Description: Maps, for the public SF crime data set,
the crimes as a function of the month (see general trends
a typical "average" year)
"""

import pandas as pd
import numpy as np
import csv as csv

from sklearn.ensemble import RandomForestClassifier
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt

def diff(a, b):
    b = set(b)
    return [aa for aa in a if aa not in b]

def analyzeData():
    print("Reading values...")
    train = pd.read_csv('../input/train.csv', dtype={'Age': np.float_})

    # Denote, in the given string format, locations (start/end) of month
    START_MONTH = 5
    END_MONTH = 7

    train['Months'] = train['Dates'].map(lambda x: x[START_MONTH:END_MONTH]).astype(int)
    train = train.drop(['Dates', 'Descript', 'DayOfWeek', 'Resolution', 
        'Address', 'X', 'Y'], axis=1)
    uniqueCategories = train['Category'].unique()
    uniqueCategoriesMap = dict(zip(uniqueCategories, 
        range(len(uniqueCategories))))
    print(train)

    # Num_months offset due to loop counter
    NUM_MONTHS = 13

    print("Analyzing files...")
    month_crimes = {}
    for month in range(1, NUM_MONTHS):
        crimes = train.loc[train['Months'] == month]
        crimesUnique = crimes['Category'].unique()

        # Finds those crimes not committed in this particular month and their
        # indicies in the lookup table (used to default their values to 0)
        notCommitted = diff(uniqueCategories, crimesUnique)
        notCommittedIndices = list(map(lambda x: 
            uniqueCategories.tolist().index(x), notCommitted))

        numCrimes = len(crimesUnique)
        crimeCounts = [len(crimes.loc[crimes['Category'] == \
            crimesUnique[i]]) for i in range(numCrimes)]

        NO_CRIME = 0
        for index in notCommittedIndices:
            if index >= len(crimeCounts): crimeCounts.append(NO_CRIME) 
            else: crimeCounts.insert(index, NO_CRIME) 
        crimeLookup = dict(zip(uniqueCategories, crimeCounts))
        month_crimes[month] = crimeLookup

    crime_per_month = {}
    for crime in uniqueCategories:
        crime_months = [month_crimes[month][crime] for month in range(1, NUM_MONTHS)]
        crime_per_month[crime] = crime_months

    toPlotCounter = 0
    numPlots = 0
    NUM_PLOTS = 3

    print("Graphing output...")
    xArray = range(1, NUM_MONTHS)
    for crime in crime_per_month:
        yArray = crime_per_month[crime]
        plt.plot(yArray, label=crime)
        toPlotCounter += 1
        if toPlotCounter % NUM_PLOTS == 0:
            numPlots += 1
            plt.legend(bbox_to_anchor=(0., 1.0, 1., .10), loc=2,
               ncol=2, mode="expand", borderaxespad=0.)

            plt.xlabel("Month")
            plt.ylabel("Crime Frequency")

            plt.savefig('Crimes_Months_{}.png'.format(numPlots))
            plt.close()

if __name__ == "__main__":
    analyzeData()