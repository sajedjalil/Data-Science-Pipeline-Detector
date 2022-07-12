# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

## A silly excuse to play around with doc2vec on the description field.
## Builds a model and runs a small RF 
## You could export and use the vectors as features? can it help with your ensemble?




import multiprocessing
import gensim.models.doc2vec
import pandas as pd
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# Download the punkt tokenizer for sentence splitting
#nltk.download() #download if not already done. "~/nltk_data"

# Multithread
cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"


def mlogloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.

    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


def description_to_wordlist(description_text, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.

    # Split them
    words = description_text.split()

    # Optionally remove stop words.
    # Word2Vec works best keeping stop words because the algorithm relies on the broader context of the sentence.
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    # Return a list of words
    return words

########################################################################################################################

data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
twosigmaconnect_doc2vec_model_filename = '../input/twosigmaconnect_doc2vec.d2v'
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)

train_df = train_df.set_index('listing_id')
test_df = test_df.set_index('listing_id')

# Cut the dataset to only cols needed.
train_df = train_df[['description', 'interest_level']]
test_df = test_df[['description']]
train_df['tr'] = 0
test_df['tr'] = 1
test_df['interest_level'] = 'NULL'

biggly_d = pd.concat([train_df, test_df], axis =0)

# A little bit of housekeeping
biggly_d['description'] = biggly_d['description'].str.lower()

biggly_d['description'] = biggly_d['description'].str.replace(r'^\s+$', 'AFULLEMPYSTRING')

biggly_d['description'] = biggly_d['description'].str.replace(r'<p><a  website_redacted ', ' ')
biggly_d['description'] = biggly_d['description'].str.replace(r'!<br /><br />', ' ')
biggly_d['description'] = biggly_d['description'].str.replace(r"<br', '/><br", ' ')
biggly_d['description'] = biggly_d['description'].str.replace(r" st ", ' street ')
biggly_d['description'] = biggly_d['description'].str.replace(r" st.", ' street ')
biggly_d['description'] = biggly_d['description'].str.replace(r" dr ", ' drive ')
biggly_d['description'] = biggly_d['description'].str.replace(r"-", ' ')
biggly_d['description'] = biggly_d['description'].str.replace(r"$amp;", '&')
biggly_d['description'] = biggly_d['description'].str.replace(r"!", '')
biggly_d['description'] = biggly_d['description'].str.replace(r"*", '')
biggly_d['description'] = biggly_d['description'].str.replace(r"\(", '')
biggly_d['description'] = biggly_d['description'].str.replace(r"\)", '')
biggly_d['description'] = biggly_d['description'].str.replace(r"+", '')
biggly_d['description'] = biggly_d['description'].str.replace(r"\t", '')
biggly_d['description'] = biggly_d['description'].str.replace(r"\r", '')
biggly_d['description'] = biggly_d['description'].str.replace(r"hr", ' hour')


sentences = []
print("Building TaggedDocuments")
for index, row in biggly_d.iterrows():
    sentences.append(TaggedDocument(description_to_wordlist(row['description']), [index]))
print("Finished")

model = Doc2Vec(min_count=5, window=10, size=100, sample=1e-5, negative=5, workers=4)
model.build_vocab(sentences)
idx = list(range(len(sentences)))
np.random.seed(345)
np.random.shuffle(idx)
perm_sentences = [sentences[i] for i in idx]
model.train(perm_sentences)


############################## Join the learnt vectors with the original dataset #######################
## Bit of a hack
num_train = len(sentences)
tr_array = np.zeros((num_train, 100))
for i in list(range(num_train)):
    tr_array[i] = model.docvecs[i]

tr_array = pd.DataFrame(tr_array, index=biggly_d.index, columns=["embed" + str(i) for i in range(0, 100)])
d = biggly_d.merge(tr_array, left_index=True, right_index=True)

train_df = d[d['tr'] == 0]
y = train_df['interest_level']
target_num_map = {'high': 0, 'medium': 1, 'low': 2}
y = y.apply(lambda x: target_num_map[x])
train_df = train_df.drop(['interest_level', 'description', 'tr'], axis=1)

xtrain, xvalid, ytrain, yvalid = train_test_split(train_df, y, random_state=42)

foo = RandomForestClassifier(n_jobs=4, n_estimators=100, random_state=42)
foo.fit(xtrain, ytrain)
bar = foo.predict_proba(xvalid)

val = mlogloss(yvalid.values, bar)
print(val)
