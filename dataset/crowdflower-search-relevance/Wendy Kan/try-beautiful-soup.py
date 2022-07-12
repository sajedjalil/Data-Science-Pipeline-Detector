
import pandas as pd

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# Use Pandas to read in the training and test data
train = pd.read_csv("../input/train.csv").fillna("")
test  = pd.read_csv("../input/test.csv").fillna("")


train.product_description = list(map(lambda row: BeautifulSoup(row).get_text(), train.product_description))
train['wordlist'] = list(map(lambda row: row.lower().split(), train.product_description))
print(train[train['query']=='boyfriend jeans'].product_description)

