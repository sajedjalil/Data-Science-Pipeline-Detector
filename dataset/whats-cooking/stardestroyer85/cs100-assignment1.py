import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Reading the data
train = pd.read_json('../input/train.json')
train.describe()
