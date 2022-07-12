"""
    Hi there! Everything below is a joke!
    Don't take it seriuos, it's just funny, that random numbers
    can take not the last position on the leaderboard.
    
    You can just check your luck with this method - change the seed and submit.
"""

import pandas as pd
import numpy as np
import zipfile

np.random.seed(42) # The heart of the algorithm. Ypu can get 4.09140 with seed=42, for example.


# Getting test ids
z = zipfile.ZipFile('../input/test.csv.zip')
test_ids = pd.read_csv(z.open('test.csv'), usecols=["Id"], squeeze=True)

# Creating names for our data-frame columns
cols = ['Id', 'ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY',\
       'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC',\
       'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION', 'FAMILY OFFENSES',\
       'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING',\
       'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON',\
       'NON-CRIMINAL', 'OTHER OFFENSES', 'PORNOGRAPHY/OBSCENE MAT',\
       'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY',\
       'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE',\
       'STOLEN PROPERTY', 'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS',\
       'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS']

# DataFrame of random numbers from 0 to 1 rounded to 0.xx (to reduce csv size)
Random_numbers = np.random.rand(39*884262).reshape((884262, 39)).round(2)
df = pd.concat([test_ids, pd.DataFrame(Random_numbers)], axis=1)
df.columns = cols

df.to_csv('all_ones.csv', index = False)
print("Your random Data frame is ready for submission")