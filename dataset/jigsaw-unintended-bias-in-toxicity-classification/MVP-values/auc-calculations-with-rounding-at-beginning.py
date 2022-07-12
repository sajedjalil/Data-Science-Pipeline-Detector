# Source for a lot of the preprocessing + initial model code: https://www.kaggle.com/taindow/simple-cudnngru-python-keras

import gc
import operator
import os
import re

from gensim.models import KeyedVectors
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, Input, Dense, CuDNNGRU,concatenate, Bidirectional, SpatialDropout1D, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn import model_selection

print(os.listdir("../input")) # To see input files
tqdm.pandas() # Run so we can see Pandas progress -- SO FUN!

# Read in the data
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))
train.head()
test.head()

# While coding, let's use just some of the training data to make things faster
#train = train.head(10000)
#print("New train shape: {}".format(train.shape))

# CHANGE TARGET
# Let's change target to be: 0 = non toxic non identity, 1 = non toxic identity, 2 = toxic non identity, 3 = toxic identity

toxicity_info = ['severe_toxicity','obscene','identity_attack','insult','threat']
#identities = ['asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu', 'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability','jewish', 'latino', 'male', 'muslim', 'other_disability','other_gender', 'other_race_or_ethnicity', 'other_religion','other_sexual_orientation', 'physical_disability','psychiatric_or_mental_illness', 'transgender', 'white']
identities = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'white', 'black', 'psychiatric_or_mental_illness']

# Round target and identity columns to 0/1
train['target'] = train['target'].round()
train[identities] = train[identities].round()

# Then make a column to record whether any identity is mentioned
train['mentions_identity'] = train[identities].any(axis=1)
    
def col_0(c):
    return 1 - c['target'] if c['mentions_identity'] == 0 else 0
        
def col_1(c):
    return 1 - c['target'] if c['mentions_identity'] == 1 else 0
        
def col_2(c):
    return c['target'] if c['mentions_identity'] == 0 else 0
        
def col_3(c):
    return c['target'] if c['mentions_identity'] == 1 else 0
    
print("is it getting here lmao")

# Make 4 target columns
train['t0'] = train.progress_apply(col_0, axis=1)
train['t1'] = train.progress_apply(col_1, axis=1)
train['t2'] = train.progress_apply(col_2, axis=1)
train['t3'] = train.progress_apply(col_3, axis=1)

print("4 targets made.")

# Import embeddings
ft_common_crawl = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
# TODO: Limit just for coding purposes! (limit = 500000)
# This is a dictionary that maps common words to their vector representations
embeddings_index = KeyedVectors.load_word2vec_format(ft_common_crawl) # Takes ~2.5 min when limited
print("Embeddings index done!")

# Now that we have an embeddings dictionary, we want to ensure that as many of our training data words
# are actually mapped to embeddings as possible.

# Function to build vocabulary index that maps words to number of times they occur
def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

# Check how many of the words in our data are in the embeddings index
def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    num_kw = 0
    num_uw = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            num_kw += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            num_uw += vocab[word]
            pass
    print("Found embeddings for {:.3%} of all vocab".format(len(known_words)/len(vocab)))
    print("Found embeddings for {:.3%} of all text".format(num_kw/(num_kw + num_uw)))
    # Sort unknown words by how often they occur
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]
    return unknown_words
    
# Make all the words lower case
# TODO: Do we actually want to do this? Does it mess up proper nouns?
train['comment_text'] = train['comment_text'].progress_apply(lambda x:x.lower())

vocab = build_vocab(train['comment_text'])
oov = check_coverage(vocab, embeddings_index)
oov[:15] # Check 15 most common out of vocab words

# FIX CONTRACTIONS
# An issue is contractions! For ex, "don't" is the most commonly missed word. Let's fix that!

# To keep memory low, delete stuff as we go. We will remake vocab + oov.
del(vocab,oov)
gc.collect()

# Map of many many common contractions to the full words
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

# Function to find known contractions in the embedding
def known_contractions(embed):
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known

print("Known contractions: \n{}".format(known_contractions(embeddings_index)))

# Replace weird apostrophes with recognized one and then replace contractions with full term if in contraction_mapping
def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text
    
# Do that!
train['comment_text'] = train['comment_text'].progress_apply(lambda x:clean_contractions(x, contraction_mapping))

# How much did our coverage improve?
vocab = build_vocab(train['comment_text'])
oov = check_coverage(vocab, embeddings_index)
oov[:15] # Check 15 most common out of vocab words

# PUNCTUATION
# The next obvious problem is punctuation... for example, 'it,'

del(vocab,oov)
gc.collect()

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

# It turns out that our embeddings index contains punctuation itself
# Is it missing any of the symbols in punct?

def missing_punct(embed, punct):
    missing = ""
    for p in punct:
        if p not in embed:
            missing += p
            missing += ' '
    return missing
    
print("Missing: {}".format(missing_punct(embeddings_index, punct)))
missing = {"_":" ", "`":" "} # We will just map the missing punctuation to spaces on our own

# Replace missing punctuation with spaces, and add spaces around punctuation in our mapping
def clean_punct(text, missing, punct_mapping):
    for p in missing:
        text.replace(p, missing[p])
    for p in punct_mapping:
        text = text.replace(p, f' {p} ') # Add spaces around punctuation
    return text
    
train['comment_text'] = train['comment_text'].progress_apply(lambda x:clean_punct(x, missing, punct)) # Now do it!

# How much did our coverage improve?
vocab = build_vocab(train['comment_text'])
oov = check_coverage(vocab, embeddings_index)
oov[:15] # Check 15 most common out of vocab words

# SMALL TEXT
# It looks like small text (e.g. 'ʜᴏᴍᴇ') is a problem. Let's replace that with normal characters.

# Interestingly, tho, it seems like a lot of this is spam. Which I guess is the point -- it gets around comment spam filters.
# In our data, is it categorized as *toxic*, though?
small_text = {'ᴀ':'a', 'ʙ':'b', 'ᴄ':'c', 'ᴅ':'d', 'ᴇ':'e', 'ғ':'f','ɢ':'g', 'ʜ':'h', 'ɪ':'i', 'ᴊ':'j', 'ᴋ':'k', 'ʟ':'l', 'ᴍ':'m', 'ɴ':'n', 
'ᴏ':'o', 'ᴘ':'p', 'ʀ':'r', 's':'s', 'ᴛ':'t', 'ᴜ':'u', 'ᴠ':'v', 'ᴡ':'w', 'x':'x', 'ʏ':'y'}

#students = train[train['comment_text'].str.contains("sᴛᴜᴅᴇɴᴛs")]
#print("{:.3%} of these comments are marked as toxic".format(len(students[students['target'] > 0.5].index)/len(students.index)))

#andframe = train[train['comment_text'].str.contains("ᴀɴᴅ")]
#print("{:.3%} of these comments are marked as toxic".format(len(andframe[andframe['target'] > 0.5].index)/len(andframe.index)))

#e = train[train['comment_text'].str.contains("ᴇ")]
#print("{:.3%} of these comments are marked as toxic".format(len(e[e['target'] > 0.5].index)/len(e.index)))

# In these three examples, the spammy comments are NEVER marked toxic.
# So it seems like toxicity markers did not mark spam as toxic.
# Therefore, instead of converting them to normal font, maybe we should convert it to a small font marker.

# TODO: Somehow incorporate this into model?

#re.sub(".*[ᴀʙᴄᴅᴇғɢʜɪᴊᴋʟᴍɴᴏᴘʀᴛᴜᴠᴡʏ].*", "smalltext", 'sᴛᴀʀᴛ')
#re.sub(".*[ᴀʙᴄᴅᴇғɢʜɪᴊᴋʟᴍɴᴏᴘʀᴛᴜᴠᴡʏ].*", "smalltext", 'sᴛᴀʀᴛ bob')

del(vocab,oov)
gc.collect()

# FURTHER PREP:

# Split into training and validating data sets
train_df, validate_df = model_selection.train_test_split(train, test_size=0.1)

# Tokenize the comments
MAX_NUM_WORDS = 10000
TOXICITY_COLUMN = 'complex_target'
TEXT_COLUMN = 'comment_text'

tk = Tokenizer(num_words=MAX_NUM_WORDS)
tk.fit_on_texts(train_df[TEXT_COLUMN])

# We need to make sure all of the comments are the same length (truncate or pad if too long/too short)
MAX_SEQUENCE_LENGTH = 256
def pad_text(texts, tokenizer):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)
    
gc.collect()

# Create our embeddings matrix
EMBEDDINGS_DIMENSION = 300 # That's how big the imported embedding vectors are
embedding_matrix = np.zeros((len(tk.word_index) + 1, EMBEDDINGS_DIMENSION))

# Fill in embeddings matrix
num_words_in_embedding = 0
for word, i in tk.word_index.items(): # If it's in our tokenized training data...
    if word in embeddings_index.vocab: # Look it up in imported vectors
        embedding_vector = embeddings_index[word] # Get vector
        embedding_matrix[i] = embedding_vector # Put it in embedding_matrix
        num_words_in_embedding += 1
        
# Prep train and validation text/labels
targs = ['t0', 't1', 't2', 't3']
train_text = pad_text(train_df[TEXT_COLUMN], tk)
train_labels = train_df[['t0', 't1', 't2', 't3']].values
validate_text = pad_text(validate_df[TEXT_COLUMN], tk)
validate_labels = validate_df[['t0', 't1', 't2', 't3']].values

gc.collect()

# MAKE THE MODEL!

sequence_input = Input(shape = (MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer = Embedding(len(tk.word_index) + 1,
                            EMBEDDINGS_DIMENSION, 
                            weights=[embedding_matrix], 
                            input_length=MAX_SEQUENCE_LENGTH, 
                            trainable=False)
x = embedding_layer(sequence_input)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)

avg_pool1 = GlobalAveragePooling1D()(x)
max_pool1 = GlobalMaxPooling1D()(x)

x = concatenate([avg_pool1, max_pool1])

preds = Dense(4, activation='softmax')(x)

model = Model(sequence_input, preds)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])
print("Model compiled.")

# Let's train this binch

BATCH_SIZE = 1024
NUM_EPOCHS = 100

model.fit(train_text, train_labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(validate_text, validate_labels),callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=3)])

# VALIDATION WITH BIAS MEASURES

# Calculate overall AUC

# Add actual and predicted validation values to validate_df
validate_predictions = model.predict(validate_text)
validate_df['predicted'] = list(pd.DataFrame(validate_predictions)[2] + pd.DataFrame(validate_predictions)[3])
validate_df['binary_prediction'] = 0
validate_df.loc[validate_df['predicted'] >= 0.5, 'binary_prediction'] = 1
validate_df['binary_true'] = np.where(validate_df['target'] >= 0.5, 1, 0)

# Calculate overall AUC
overall_auc = metrics.roc_auc_score(validate_df['binary_true'].values, validate_df['binary_prediction'].values)
print("Overall AUC is: {:.3%}".format(overall_auc))

# Wrapper method for computing AUC to catch exceptions
def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

auc_df = pd.DataFrame(columns=['identity','AUC', 'BPSN_AUC', 'BNSP_AUC'])
auc_df.loc[0, 'identity'] = 'overall'
auc_df.loc[0, 'AUC'] = overall_auc

idx = 1

# Calculate AUC for each subgroup
for identity in identities:
    auc_df.loc[idx, 'identity'] = identity
    
    identity_examples = validate_df[validate_df[identity] == 1.0]
    identity_auc = metrics.roc_auc_score(identity_examples['binary_true'].values, identity_examples['binary_prediction'].values)
    print("AUC for identity {}: {:.3%}".format(identity, identity_auc))
    auc_df.loc[idx, 'AUC'] = identity_auc
    
    # Restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not
    # A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not, 
    # likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.
    test_set = validate_df[((validate_df[identity] == 1.0) & (validate_df['binary_true'] == 0.0)) | ((validate_df[identity] == 0.0) & (validate_df['binary_true'] == 1.0))]
    bpsn_auc = metrics.roc_auc_score(test_set['binary_true'].values, test_set['binary_prediction'].values)
    print("BPSN AUC for identity {}: {:.3%}".format(identity, bpsn_auc))
    auc_df.loc[idx, 'BPSN_AUC'] = bpsn_auc
    
    # Restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not
    # A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not, 
    # likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.
    test_set = validate_df[((validate_df[identity] == 1.0) & (validate_df['binary_true'] == 1.0)) | ((validate_df[identity] == 0.0) & (validate_df['binary_true'] == 0.0))]
    bnsp_auc = metrics.roc_auc_score(test_set['binary_true'].values, test_set['binary_prediction'].values)
    print("BNSP AUC for identity {}: {:.3%}".format(identity, bnsp_auc))
    auc_df.loc[idx, 'BNSP_AUC'] = bpsn_auc
    
    idx = idx + 1
    
auc_df.to_csv("scores.csv", index=False)
    
# MAKE AND SUBMIT PREDICTIONS

# Read in sample submission file
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
submission.reset_index(drop=False, inplace=True) # idk what this does lol
print(submission.head())

output = model.predict(pad_text(test[TEXT_COLUMN], tk))

# Change prediction to sum of last two output rows (combined probability of toxicity)
for index, row in submission.iterrows():
    submission.loc[index, 'prediction'] = output[index][2] + output[index][3]

# Use pandas to write output csv file
submission.to_csv("submission.csv", index=False, quoting=3)