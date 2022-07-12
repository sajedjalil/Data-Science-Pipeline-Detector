import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import string

dataframes = {
    "cooking": pd.read_csv("../input/cooking.csv"),
    "crypto": pd.read_csv("../input/crypto.csv"),
    "robotics": pd.read_csv("../input/robotics.csv"),
    "biology": pd.read_csv("../input/biology.csv"),
    "travel": pd.read_csv("../input/travel.csv"),
    "diy": pd.read_csv("../input/diy.csv"),
}


uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

def stripTagsAndUris(x):
    if x:
        # BeautifulSoup on content
        soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text =  soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)
    else:
        return ""
        
        
        
for df in dataframes.values():
    df["content"] = df["content"].map(stripTagsAndUris)

print(dataframes["robotics"].iloc)

def removePunctuation(x):
    # Lowercasing all words
    x = x.lower()
    # Removing non ASCII chars
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
    # Removing (replacing with empty spaces actually) all the punctuations
    return re.sub("["+string.punctuation+"]", " ", x)
    
for df in dataframes.values():
    df["title"] = df["title"].map(removePunctuation)
    df["content"] = df["content"].map(removePunctuation)
    
print(dataframes["robotics"].iloc[1])

stops = set(stopwords.words("english"))
def removeStopwords(x):
    # Removing all the stopwords
    filtered_words = [word for word in x.split() if word not in stops]
    return " ".join(filtered_words)
    
for df in dataframes.values():
    df["title"] = df["title"].map(removeStopwords)
    df["content"] = df["content"].map(removeStopwords)
    
print(dataframes["robotics"].iloc[1])

for df in dataframes.values():
    # From a string sequence of tags to a list of tags
    df["tags"] = df["tags"].map(lambda x: x.split())
    
print(dataframes["robotics"].iloc[1])

for name, df in dataframes.items():
    # Saving to file
    df.to_csv(name + "_light.csv", index=False)