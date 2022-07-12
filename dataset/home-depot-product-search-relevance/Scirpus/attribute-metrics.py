import math
import pandas as pd
import re
import cgi
from nltk.stem.porter import *
import string
from collections import Counter
from difflib import SequenceMatcher


WORD = re.compile(r'\w+')
porter = PorterStemmer()
punct = string.punctuation
punct_re = re.compile('[{}]'.format(re.escape(punct)))
tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')


def porterStemmer(x):
    porterwords = []
    for word in x.split():
        porterword = porter.stem(word)
        porterwords.append(porterword)
    return ' '.join(porterwords).lower()


def get_cleantext(x):
    if isinstance(x, str):
        no_tags = tag_re.sub('', x, re.M | re.I)
        no_tags = cgi.escape(no_tags)
        no_tags = punct_re.sub(' ', no_tags, re.M | re.I)
        ready_for_web = re.sub('[^a-zA-Z0-9]+', ' ', no_tags, re.M | re.I)
        return porterStemmer(ready_for_web.lower())
    else:
        return "null"


def text_to_vector(text):
    words = WORD.findall(text.lower())
    return Counter(words)


def get_similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_hamdist(str1, str2):
    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def get_levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n
    current = range(n+1)
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return current[n]


if __name__ == "__main__":
    print('Started!')
    train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
    train.fillna('', inplace=True)
    attributes = \
        pd.read_csv('../input/attributes.csv',
                    encoding="ISO-8859-1")[['product_uid', 'value']]
    attributes.fillna('', inplace=True)
    attributestats = \
        train.iloc[:100][['id', 'product_uid', 'search_term']].copy()
    attributestats = pd.merge(attributestats,
                              attributes,
                              how='left',
                              on='product_uid')
    attributestats.search_term = \
        attributestats.apply(lambda x: get_cleantext(x['search_term']),
                             axis=1)
    attributestats.value = \
        attributestats.apply(lambda x: get_cleantext(x['value']),
                             axis=1)
    attributestats['attributecosine'] = \
        attributestats.apply(lambda x: get_cosine(text_to_vector(x['search_term']),
                             text_to_vector(x['value'])), axis=1)
    attributestats['attributeham'] = \
        attributestats.apply(lambda x: get_hamdist(x['search_term'],
                             x['value']), axis=1)
    attributestats['attributesimilar'] = \
        attributestats.apply(lambda x: get_similar(x['search_term'],
                             x['value']),
                             axis=1)
    attributestats['attributelevenshtein'] = \
        attributestats.apply(lambda x: get_levenshtein(x['search_term'],
                             x['value']), axis=1)
    attributestats = \
        attributestats.groupby('id')[['attributecosine',
                                      'attributeham',
                                      'attributesimilar',
                                      'attributelevenshtein']].sum().reset_index()
    print(attributestats.head())
