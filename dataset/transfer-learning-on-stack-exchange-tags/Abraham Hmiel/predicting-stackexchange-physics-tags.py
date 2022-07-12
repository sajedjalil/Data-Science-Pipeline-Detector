# Python 3
# On first several passes, just try the simplest things and we'll add complexity later

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

# Input data files are available in the "../input/" directory.
biologyTable = pd.read_csv('../input/biology.csv',header=0)
cookingTable = pd.read_csv('../input/cooking.csv',header=0)
cryptoTable  = pd.read_csv('../input/crypto.csv',header=0)
diyTable     = pd.read_csv('../input/diy.csv',header=0)
roboticsTable= pd.read_csv('../input/robotics.csv',header=0)
travelTable  = pd.read_csv('../input/travel.csv',header=0)
physicsTable = pd.read_csv('../input/test.csv',header=0)

#clean data using BeautifulSoup and Regex. First define functions:

#cleans title of punctuation and sends to lower case (and optionally, stop words)
def titles_to_wordlist(title, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    # No need to remove HTML
    #  
    # 1. Remove non-letters (and delete hyphens for now... this will have to be revised later)
    title_text = re.sub("[^a-zA-Z0-9]"," ", title)
    #
    # 2. Convert words to lower case and split them
    words = title_text.lower().split()
    #
    # 3. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 4. Return a list of words
    return(words)

#cleans content of html (and optionally, stop words)
def content_to_wordlist(content, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    content_text = BeautifulSoup(content, 'lxml').get_text()
    #  
    # 2. Remove non-letters (and delete hyphens for now... this will have to be revised later)
    content_text = re.sub("[^a-zA-Z0-9]"," ", content_text)
    #
    # 3. Convert words to lower case and split them
    words = content_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)
    
#splits tags into list of words
def tags_to_wordlist(tags):
    # Tags are already in lower case and grouped
    # by using hyphens to incidate multiple words
    #
    # 1. Split the tags
    words = tags.split()
    #
    # 2. Return a list of words
    return(words)


clean_content = content_to_wordlist( biologyTable["content"][0],remove_stopwords=True )

clean_titles = []
clean_content = []
clean_tags = []

#form one large data frame with all the training set info to make text cleaning easier
learningTables = [biologyTable, cookingTable, cryptoTable, diyTable, roboticsTable, travelTable]

print("Cleaning and parsing the training set StackExchange questions...\n")


for table in learningTables:
    num_records = table.shape[0]
    for record in table.index:
        # If the index is evenly divisible by 1000, print a message
        if( (record+1)%1000 == 0 ):
            print("Question %d of %d\n" % ( record+1, num_records ) )
        clean_titles.append( titles_to_wordlist( table["title"][record], remove_stopwords=True ))
        clean_content.append( content_to_wordlist( table["content"][record], remove_stopwords=True ))
        clean_tags.append( tags_to_wordlist( table["tags"][record] ))

#not yet
#allTables = pd.concat(learningTables)

#begin parsing the learning tables into bag-of-words matrices
count_vect = CountVectorizer()







