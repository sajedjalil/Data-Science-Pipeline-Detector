# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

animals = pd.read_csv('../input/train.csv')
#print(animals.head())

y = animals.shape

x = animals.describe()
print(x)



ot = animals["OutcomeType"].value_counts(normalize = True)
otv = animals["OutcomeType"].value_counts()
print(otv)

#for drawing up a pie chart of the proportions of outcomes
series = pd.Series(3 * np.random.rand(5), index=['Adoption', 'Return to Owner', 'Transfer', 'Euthanasia', 'Died'], name='Outcome')
series.plot.pie(figsize=(6, 6))


#amount of each subtype
ost = animals["OutcomeSubtype"].value_counts()
#print(ost)

#amount of each animal of different ages
a = animals["AgeuponOutcome"].value_counts()
#print(a)


# SEEING HOW THE AGE AFFECTS THE OUTCOME
#making all of the values into days
def calc_age(x):
    x = str(x)
    if x == 'nan': return 0
    elif 'day ' in x:
        return 1
        
    elif 'days' in x:
        x = x.replace("days","")
        x = int(x)
        return x
        
    elif 'week ' in x:
        return 7
        
    elif 'weeks' in x:
        x = x.replace("weeks", "")
        x = 7*int(x)
        return x
    
    elif 'month ' in x:
        return 30
        
    elif 'months' in x:
        x = x.replace("months", "")
        x = 30*int(x)
        return x 
        
    elif 'year ' in x:
        return 365
        
    elif 'years' in x:
        x = x.replace("years","")
        x = 365*int(x)
        return x
        
    else:
        return 0


animals["Age in Days"] = float('NaN')
animals["Age in Days"]= animals.AgeuponOutcome.apply(calc_age)



animals["Dogs"] = float('NaN')
animals["Dogs"][animals["AnimalType"] == 'Dog'] = 1
animals["Dogs"][animals["AnimalType"] == 'Cat'] = 0



#separating puppies/kittens from dogs/cats
#animals["Puppy"] = float('NaN')
#animals["Puppy"][animals["AgeuponOutcome"] =< 1] = 1
#animals["Puppy"][animals["AgeuponOutcome"] > 1] = 0





