#This script looks at the survival rate for those with the most common names

import numpy as np
import pandas as pd

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
dfTitanic = train

#Extract the first name from passenger name
dfTitanic['FirstName'] = dfTitanic['Name'].str.extract('(Mr\. |Miss\. |Master. |Mrs\.[A-Za-z ]*\()([A-Za-z]*)')[1]

#pull out the passengers that have popular names (> 10 occurances)
dfPassengersWithPopularNames = dfTitanic[dfTitanic['FirstName'].isin(dfTitanic['FirstName'].value_counts()[dfTitanic['FirstName'].value_counts() > 10].index)]

#calculate the surival rate by popular name
ax = (dfPassengersWithPopularNames.groupby('FirstName').Survived.sum()/dfPassengersWithPopularNames.groupby('FirstName').Survived.count()).order(ascending=False).plot(kind='bar',y='Survival rate',fontsize=8)

#set y axis label and save to png for display below
ax.set_ylabel("Survival Rate")
fig = ax.get_figure()
fig.savefig('figure.png')



