#
# This script creates a figure showing the 10 most used ingredients per cuisine
#
# As you will see, the salt is the Queen of the Cuisine :-)
#
# The ingredients need to be cleaned, making 'low fat mozzarella' and 
# 'reduced fat mozzarella' the same ingredient. Ideas are welcome.
# 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Reading the data
train = pd.read_json('../input/train.json')

stemmer = WordNetLemmatizer()

# Auxiliar function for cleaning
def clean_recipe(recipe):
    # To lowercase
    recipe = [ str.lower(i) for i in recipe ]

    # Remove some special characters
    # Individuals replace have a very good performance
    # http://stackoverflow.com/a/27086669/670873
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
bags_of_words = [ Counter(clean_recipe(recipe)) for recipe in train.ingredients ]
sumbags = sum(bags_of_words, Counter())

# This plot will show the number of receipes per cuisine. Is everybody cooking Italian food?
plt.style.use(u'ggplot')
fig = train.cuisine.value_counts().plot(kind='bar')
fig = fig.get_figure()
fig.tight_layout()
fig.savefig('Number_of_recipes_per_cuisine.jpg')

# Finally, plot the 10 most used ingredients per cuisine
# Filling the dataset with a column per ingredient
for ingredient in list(sumbags.keys()):
    train[ingredient] = train.ingredients.apply(lambda x: ingredient in x)

cuisine = train.drop(['ingredients', 'id'], axis=1).groupby('cuisine')
cuisine = cuisine.aggregate(np.sum)

fig, axes = plt.subplots(nrows=10, ncols=2, sharex=False, sharey=False, figsize=(30,60))
for i, c in enumerate(list(cuisine.index)):
    cuisine.loc[c].sort(inplace=False, ascending=False)[:10].plot(kind='barh', ax=axes[int(i/2), int(i%2)], title=c)
    axes[int(i/2), int(i%2)].invert_yaxis()

fig.tight_layout()
fig.savefig('10_most_used_ingredients_per_cuisine.jpg')
