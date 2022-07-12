import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Reading the data
train = pd.read_json('../input/train.json')
train.dtypes
stemmer = WordNetLemmatizer()
#cachedStopWords = stopwords.words("english")

# Auxiliar function for cleaning
def clean_cuisine(cuisine):
    # To lowercase
    cuisine = [ str.lower(i) for i in cuisine ]

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
    cuisine = [ replacing(i) for i in cuisine ]
    
    # Remove digits
    cuisine = [ i for i in cuisine if not i.isdigit() ]
    
    # Stem ingredients
    cuisine = [ stemmer.lemmatize(i) for i in cuisine ]
    
    return cuisine

# The number of times each ingredient is used is stored in the 'sumbags' dictionary
bags_of_words = [ Counter(clean_cuisine(cuisine)) for cuisine in train.ingredients ]
sumbags = sum(bags_of_words, Counter())

# Finally, plot the 10 most used ingredients
plt.style.use(u'ggplot')
pd.DataFrame(sumbags, index=[0]).transpose()[0].sort(ascending=False, inplace=False)[:10].plot(kind='barh')