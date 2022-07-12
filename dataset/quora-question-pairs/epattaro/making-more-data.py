## Having more data is key to getting better results
## Enough (good) data can beat the best feature engineering / model tunning & stacking
##
## This notebook provides 2 ways to engineer more data
## I am currently using one, and implementing the other in my model.
##
## I will try to comment it as best as I can

################################################################################################
############# Step 1: Importing libraries & deffining some key parameters ######################
################################################################################################

import pandas as pd
import numpy as np
import re
import string
from string import printable
punctuation = set(string.punctuation)
from tqdm import tqdm

from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk

SW_1 = get_stop_words('english')
SW_2 = stopwords.words('english')
SW = sorted (set (SW_1 + SW_2))

from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

porter = PorterStemmer()
snowball = SnowballStemmer('english')

verbs = ['VB','VBD','VBG','VBN','VBP','VBZ']
adverbs = ['RB','RBR','RBS','WRB']
nouns = ['NN','NNP','NNPS','NNS']
pronouns = ['PRP','PRP$','WP','WP$']

word_types = [verbs, adverbs, nouns, pronouns]
types_names = ['verbs', 'adverbs', 'nouns', 'pronouns']

from nltk.tag.perceptron import PerceptronTagger
tagger = PerceptronTagger()
tagset = None

################################################################################################
############################# Step 2: Pre defined functions  ###################################
################################################################################################

def clean_text (string, clean_SW=False):
    
    string = string.replace('-',' ') ## break words with "-"
    
    cleanr = re.compile('<.*?>')
       
    cleantext = re.sub(cleanr, '', string) ## removes web/html notation
       
    cleantext = [x.lower() if x!=x.upper() else x for x in cleantext.split()] ## make all lower case unless word is all upper
    cleantext = ' '.join([x for x in cleantext if x not in SW])
    
    cleantext = cleantext.replace('\n',' ') ## removes skip lines
    
    cleantext = cleantext.replace('  ',' ').replace('  ',' ').replace('  ',' ') ## removes extra spaces between words
       
    cleantext = cleantext.strip() ## removes extra spaces in the end/beggining of words
       
    cleantext = ''.join(ch for ch in cleantext if ch not in punctuation) ## removes punctuation
    
    return (cleantext)
    
def stem_str(x,stemmer=SnowballStemmer('english')):
    x = text.re.sub("[^a-zA-Z0-9]"," ", x)
    x = (" ").join([stemmer.stem(z) for z in x.split(" ")])
    x = " ".join(x.split())
    return x
    
################################################################################################
############################# Step 2.5 loads up the data:  #####################################
################################################################################################

# so this can run on kaggle, a smaller set is being read through nrows, one can simply change it

df_test = pd.read_csv('../input/test.csv', encoding='utf-8', nrows=10000).fillna('').drop('test_id',1)
df_train = pd.read_csv('../input/train.csv', encoding='utf-8', nrows=10000).fillna('').drop(['id'],1)
df_train['train_set'] = 1
df_test['test_set'] = 1

df = pd.concat([df_train,df_test], 0)
df.index = np.arange(len(df))

################################################################################################
#################### Step 3:  different version of questions 1 & 2:  ###########################
################################################################################################

#
# Basically instead of just using q1 & q2 we are now going to have tuples of questions to work with
#
# (q1 & q2), (q1_without_verbs, q2_without_verbs), (q1_only_verbs, q2_only_verbs), (q1_clean,q2_clean), ....
# 
# all new features will have to be built on top of those tuples, hence, there will be a ton of new features
#

df['question1'] = df['question1'].fillna('')
df['question2'] = df['question2'].fillna('')

df['question1_original'] = df['question1']
df['question2_original'] = df['question2']

df['question1'] = df['question1'].map(lambda x: clean_text(x)) ## takes a while
df['question2'] = df['question2'].map(lambda x: clean_text(x)) ## takes a while

df['question1_porter'] = df['question1'].apply(lambda x:stem_str(x.lower(),porter))
df['question2_porter'] = df['question2'].apply(lambda x:stem_str(x.lower(),porter))

temp1 = df['question1'].apply(lambda x:  word_tokenize(x))
temp2 = df['question2'].apply(lambda x:  word_tokenize(x))

temp1_tags = temp1.apply(lambda x: nltk.tag._pos_tag(x, None, tagger))
temp2_tags = temp2.apply(lambda x: nltk.tag._pos_tag(x, None, tagger))

df['question1_tags'] = temp1_tags
df['question2_tags'] = temp2_tags

for wtype, name in tqdm(zip(word_types, types_names)):
    
    df['question1_only_%s' %name] = df['question1_tags'].map(lambda string: ' '.join([x for (x,y) in string if y in wtype]))    
    df['question2_only_%s' %name] = df['question2_tags'].map(lambda string: ' '.join([x for (x,y) in string if y in wtype]))
    
    df['question1_not_%s' %name] = df['question1_tags'].map(lambda string: ' '.join([x for (x,y) in string if y not in wtype]))    
    df['question2_not_%s' %name] = df['question2_tags'].map(lambda string: ' '.join([x for (x,y) in string if y not in wtype]))
    
df = df.drop(['question1_tags','question2_tags'],1)

################################################################################################
########################### Step 4: New non-duplicated questions:  #############################
################################################################################################

#
# The idea here is that we can make an (almost) arbitrary number of new non duplicated questions
# How so?
# If we take 2 random questions from our total questions sampling pool, 
# the odds that they have the same meaning is so small we can disconsider it.
#

columns_1 = [x for x in df.columns if '1' in x]
columns_2 = [x for x in df.columns if '2' in x]

temp_df_1 = df[columns_1]
temp_df_2 = df[columns_2]

## making new questions == to the size of the df. One can change the value.
## also; this code makes new question by combining question1 & question2. 
## a better approach would be to put all questions (both 1 & 2) in a same set, and sample from there

row_list_1 = np.random.choice(df.index, len(df))
row_list_2 = np.random.choice(df.index, len(df))

temp_df_1 = temp_df_1.ix[row_list_1]
temp_df_1.index = np.arange(len(temp_df_1))
temp_df_2 = temp_df_2.ix[row_list_2]
temp_df_2.index = np.arange(len(temp_df_2))

temp_concat = pd.concat([temp_df_1,temp_df_2],1)
temp_concat['is_duplicate'] = 0
temp_concat = temp_concat.fillna('')

temp_concat['MUND'] = 1

df = pd.concat([df,temp_concat],0)
df.index = np.arange(len(df))

################################################################################################
############################# Step 5: New duplicated questions:  ###############################
################################################################################################

## we have repeated questions being compared to different questions
## if:
## A = B
## B = C
## we can imply that:
## A = C
## Hence we can make a limited amount of new duplicated questions
## Its also valuable to notice that we can make
## B = A
## C = B
## C = A
## Why would we do that? Well, many features run on compared tuples (question1&question2) but many features run
## on just one of those tuples, hence, having the mirroed version of the questions, will help the code learn that it
## is not having a single feature that helps, but that having it in pairity with its tuple.

df_duplicates = df.loc[df['is_duplicate']==1][['qid1','qid2']]
originals = df_duplicates

df_duplicates = pd.concat([df_duplicates,df_duplicates.rename(columns={'qid1':'qid2','qid2':'qid1'})])
df_duplicates = np.array(pd.merge(df_duplicates, df_duplicates, on='qid1', how='left'))
df_duplicates = pd.DataFrame(np.concatenate((df_duplicates[:,:2], df_duplicates[:,1:3]))).drop_duplicates()
df_duplicates = df_duplicates.loc[df_duplicates[0]!=df_duplicates[1]]
df_duplicates.columns = ['qid1','qid2']
df_duplicates = pd.DataFrame([(x,y) for (x,y) in df_duplicates[['qid1','qid2']].get_values() if (x,y) not in originals.get_values()])
df_duplicates.columns = ['qid1','qid2']

tempdf1 = df[columns_1].loc[df['qid1'].isnull()==False]
tempdf2 = df[columns_2].loc[df['qid2'].isnull()==False]

tempdf1.columns = [col.replace('1','N') for col in tempdf1.columns]
tempdf2.columns = [col.replace('2','N') for col in tempdf2.columns]

tempdf = pd.concat([tempdf1,tempdf2], 0).drop_duplicates()
tempdf1 = tempdf.copy()
tempdf2 = tempdf.copy()
tempdf1.columns = [col.replace('N','1') for col in tempdf1.columns]
tempdf2.columns = [col.replace('N','2') for col in tempdf2.columns]

df_duplicates = pd.merge(df_duplicates, tempdf1, how='left', on='qid1')
df_duplicates = pd.merge(df_duplicates, tempdf2, how='left', on='qid2')

df_duplicates['is_duplicate'] = 0

df_duplicates = df_duplicates.fillna('')

df_duplicates['MUD'] = 1

df = pd.concat([df, df_duplicates],0)
df.index = np.arange(len(df))

################################################################################################
###################################### Step 6: saving!  ########################################
################################################################################################

#
# the original dataframe was already massive. its now even bigger. saving each set separately will help
# to properly process it when making new features. (these lines are commented out so they wont be run on kaggle)
#
# I hope it helps! If so, please share/upvote!


# train_index = df.loc[df['train_set']==1].index
# test_index = df.loc[df['test_set']==1].index
# madeup_non_duplicates = df.loc[df['MUND']==1].index
# madeup_duplicates = df.loc[df['MUD']==1].index

# indexes = [train_index, test_index, madeup_non_duplicates, madeup_duplicates]
# names = ['TRAIN','TEST','MUND','MUD']

# for file_name, index in zip(names,indexes):
    
#     temp = df.ix[index]
#     temp.to_csv('INPUT_DATAFRAMES/%s.csv' %file_name, index=False, encoding='utf-8')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory