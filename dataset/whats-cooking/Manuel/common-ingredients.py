#
# Script printing the common ingredients to all the cuisines.
#
# It seems reasonable to remove them from the dataset
#

import pandas as pd
#from nltk.corpus import stopwords
from itertools import chain
import numpy as np

# Reading the data
train = pd.read_json('../input/train.json')

# Some cleaning
#cachedStopWords = stopwords.words("english")
def clean_ingredient(ingredient):
    # To lowercase
    ingredient = str.lower(ingredient)

    # Remove some special characters
    ingredient = ingredient.replace('&', '').replace('(', '').replace(')','')
    ingredient = ingredient.replace('\'', '').replace('\\', '').replace(',','')
    ingredient = ingredient.replace('.', '').replace('%', '').replace('/','')

    # Remove digits
    ingredient = ''.join([i for i in ingredient if not i.isdigit()])
    
    # Remove stop words
    #ingredient = ' '.join([i for i in ingredient.split(' ') if i not in cachedStopWords])

    return ingredient
    
# Get all the ingredients
all_ingredients = {clean_ingredient(ingredient) for ingredient in chain(*train.ingredients)}

# fill the dataset with a column per ingredient
for ingredient in all_ingredients:
    train[ingredient] = train.ingredients.apply(lambda x: ingredient in x)
    
# Group by cuisine
cuisine = train.groupby('cuisine')

# Summ the ingredients
cuisine = cuisine.aggregate(np.sum)

# And finally take the common ingredients
common_ingredients = {ingredient if all(item > 0 for item in cuisine[ingredient]) else 0 for ingredient in all_ingredients}
common_ingredients.remove(0)

print('Common ingredients to all the cuisines:', len(common_ingredients))
print(common_ingredients)