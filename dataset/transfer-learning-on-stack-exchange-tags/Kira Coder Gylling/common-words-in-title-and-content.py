# -*- coding: utf-8 -*-

import pandas as pd
import re
from nltk.corpus import stopwords
import numpy as np

# Strip html tags from content
def strip_tags(df):
    df['content'] = [re.sub('<[^<]+?>', '', text) for text in df['content']]

# Strip non-alphanumeric characters from content and title
def strip_nonalphanum(df):
    df['content'] = [re.sub(r'([^\s\w]|_)+', '', text) for text in df['content']]
    df['title'] = [re.sub(r'([^\s\w]|_)+', '', text) for text in df['title']]

# Turn content and title to lowercase
def lowercase(df):
    df['content'] = df['content'].str.lower()
    df['title'] = df['title'].str.lower()

# Remove stop words from content
def rm_stop_words(df):
    stop = set(stopwords.words('english'))
    df_split = [text.replace(',', '').split() for text in df['content']]
    for i in range(len(df_split)):
        df.loc[i, 'content'] = ' '.join(word for word in df_split[i] if word not in stop)

def main():
    # Prepare data
    test = pd.read_csv('../input/test.csv')
    strip_tags(test)
    strip_nonalphanum(test)
    lowercase(test)
    rm_stop_words(test)

    # Extract possible tags
    N = len(test)
    test['tags'] = np.zeros(N)
    for i in range(N):
        test.loc[i, 'tags'] = ' '.join(set(test.loc[i, 'title'].split()) & set(test.loc[i, 'content'].split()))
    
    # Write csv
    test.to_csv('common_words.csv', columns = ['id', 'tags'], index=False, doublequote=False)
            
if __name__ == "__main__":
    print('Starting script...')
    main()