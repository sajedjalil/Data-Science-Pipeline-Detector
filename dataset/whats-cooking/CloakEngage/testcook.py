import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem import WordNetLemmatizer
from collections import Counter
import json

# Reading the data
train = pd.read_json('../input/train.json')

stemmer = WordNetLemmatizer()

# the function for cleaning the pattern in json.file
def clean_recipe(recipe):
    # set to the lowercase
    recipe = [ str.lower(i) for i in recipe ]
    # Remove some special characters
    def replacing(i):
        i = i.replace('&', '').replace('(', '').replace(')','')
        i = i.replace('\'', '').replace('\\', '').replace(',','')
        i = i.replace('.', '').replace('%', '').replace('/','')
        i = i.replace('"', '')
        return i
    # Replacing characters
    recipe = [ replacing(i) for i in recipe ]
    # Remove digits
    recipe = [ i for i in recipe if not i.isdigit() ]
    # Stem ingredients
    recipe = [ stemmer.lemmatize(i) for i in recipe ]
    return recipe

train['ingredients'] = train.ingredients.apply(lambda x: clean_recipe(x))
# The number of times each ingredient is used is stored in the 'sumbags' dictionary
dictWord = [ Counter(clean_recipe(recipe)) for recipe in train.ingredients ]

# collect the ingredients and counting the value
totaldictWord = sum(dictWord, Counter())
print (totaldictWord)
# change the dictionary to list 
result = list()
for key, value in totaldictWord.items():
    result.append((value, key))
#print (result)




for ingredient in list(totaldictWord.keys()):
    train[ingredient] = train.ingredients.apply(lambda x: ingredient in x)
  
cuisine = train.drop(['ingredients', 'id'], axis=1).groupby('cuisine')
cuisine = cuisine.aggregate(np.sum)

my_colors = 'rgbkymc'

fig, axes = plt.subplots(nrows=10, ncols=2, sharex=False, sharey=False, figsize=(20,40))
for i, c in enumerate(list(cuisine.index)):
    cuisine.loc[c].sort(inplace=False, ascending=False)[:20].plot(kind='bar', ax=axes[int(i/2), int(i%2)], title=c,color = my_colors)
    axes[int(i/2), int(i%2)]

fig.tight_layout()
fig.savefig('20 top most ingredients.jpg')





with open('../input/train.json') as data_file:    
    data = json.load(data_file)
    

allword = []
totalCon = dict()
for item in data:
    for itemIn in item['ingredients']:
        allword.append((item['cuisine'],itemIn))
        
for x in allword:
    if x not in totalCon:
        totalCon[x] = 1
    else:
        totalCon[x] = totalCon[x]+1
#print(totalCon)
