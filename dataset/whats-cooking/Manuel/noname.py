#
# This script creates a plot with the 10 most used ingredients.
#
# The original recipe, contained in the 'ingredients' column, is cleaned as follow:
#
# - to lowecase
# - replacing symbols
# - removing digits
# - stemming the words using the WordNetLemmatizer
#
# The ingredients should be cleaned mote, making 'low fat mozzarella' and 
# 'reduced fat mozzarella' the same ingredient. Ideas are welcome.
# 

import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Reading the data
train = pd.read_json('../input/train.json')

stemmer = WordNetLemmatizer()
#cachedStopWords = stopwords.words("english")

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

# The number of times each ingredient is used is stored in the 'sumbags' dictionary
bags_of_words = [ Counter(clean_recipe(recipe)) for recipe in train.ingredients ]
sumbags = sum(bags_of_words, Counter())

# Finally, plot the 10 most used ingredients
plt.style.use(u'ggplot')
fig = pd.DataFrame(sumbags, index=[0]).transpose()[0].sort(ascending=False, inplace=False)[:10].plot(kind='barh')
fig.invert_yaxis()
fig = fig.get_figure()
fig.tight_layout()
fig.savefig('10_most_used_ingredients.jpg')
