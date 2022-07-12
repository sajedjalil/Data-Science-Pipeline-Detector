# %% [code]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
import string 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import spacy

from tqdm import trange
import random
from spacy.util import compounding,minibatch
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from plotly.subplots import make_subplots
import plotly.graph_objects as go

stop = stopwords.words('english')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# %% [code]
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

# %% [code]
print('train shape:',train.shape)
print('test shape:',test.shape)
train.head()

# %% [code]
print('Sentiment of text : {} \nOur training text :\n{}\nSelected text to predict:\n{}'.format(train['sentiment'][1],train['text'][1],train['selected_text'][1]))


# %% [code]
train.isnull().sum()

# %% [code]
train.dropna(inplace=True)
print(train)

# %% [code]
train.sentiment.describe()

# %% [code]

temp=train.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)
temp.style.background_gradient(cmap='Blues')

# %% [code]
fig=make_subplots(1,2,subplot_titles=('Train set','Test set'))
x=train.sentiment.value_counts()
fig.add_trace(go.Bar(x=x.index,y=x.values,marker_color=['#3368d4','#32ad61','#f24e4e'],name='train'),row=1,col=1)
x=test.sentiment.value_counts()
fig.add_trace(go.Bar(x=x.index,y=x.values,marker_color=['#3368d4','#32ad61','#f24e4e'],name='test'),row=1,col=2)

# %% [code]
def jaccard_similarity(str1,str2):
    A = set(str1.lower().split())
    B = set(str2.lower().split())
    C = A.intersection(B)
    return  float(len(C))/(len(A)+len(B)-len(C))

# %% [code]
str1 = 'MY NAME IS KEVIN'
str2 = 'MYSELF KEVIN'
jaccard_score = jaccard_similarity(str1,str2)
print('JACCARD SCORE :',jaccard_score)

# %% [code]
def jaccard_similarity(df):
    A = set(df['text'].lower().split())
    B = set(df['selected_text'].lower().split())
    C = A.intersection(B)
    return float(len(C))/(len(A)+len(B)-len(C))

# %% [code]
train['jaccard_score'] = train.apply(jaccard_similarity,axis=1)


# %% [code]
train['NO_WORDS_ST'] = train.selected_text.apply(lambda x: len(str(x).split()))
train['NO_WORDS_T'] = train.text.apply(lambda x: len(str(x).split()))
train['DIFF_WORDS']  = train['NO_WORDS_T'] - train['NO_WORDS_ST'] 

# %% [code]
train.head()

# %% [code]
#DISTRIBUTION OF LENGTH B/W SELECTED_TEXT AND TEXT'
plt.hist(train['NO_WORDS_ST'],bins=20,label='selected_text')
plt.hist(train['NO_WORDS_T'],bins=20,label='text')
plt.title('DISTRIBUTION OF LENGTH B/W SELECTED_TEXT AND TEXT')
plt.legend()
plt.show()

# %% [code]
plt.figure(figsize=(8,6))
sns.kdeplot(train['NO_WORDS_ST'],shade=True,COLOR='B')
sns.kdeplot(train['NO_WORDS_T'],shade=True,COLOR='R')
plt.title('DISTRIBUTION OF LENGTH')
plt.show()

# %% [code]
plt.figure(figsize=(8,6))
sns.kdeplot(train[train['sentiment']=='positive']['DIFF_WORDS'],shade=True,COLOR='B',label='DIFF_WORDS_POS')
sns.kdeplot(train[train['sentiment']=='negative']['DIFF_WORDS'],shade=True,COLOR='R',label='DIFF_WORDS_NEG')
plt.title('DISTRIBUTION OF DIFFERNCE IN LENGTH OF POSITIVE WORDS & NEGATIVE WORDS')
plt.show()

# %% [code]
plt.figure(figsize=(8,6))
sns.kdeplot(train[train['sentiment']=='positive']['jaccard_score'],shade=True,COLOR='B',label='jaccard_score_pos')
sns.kdeplot(train[train['sentiment']=='negative']['jaccard_score'],shade=True,COLOR='R',label='jaccard_score_neg')
plt.title('DISTRIBUTION OF JACCARD SCORE OF POSITIVE WORDS , NEGATIVE WORDS & NEUTRAL WORDS')
plt.show()

# %% [code]
train[train['sentiment']=='neutral']['jaccard_score'].describe()

# %% [code]
plt.figure(figsize=(12,6))
sns.boxplot(train[train['sentiment']=='neutral']['jaccard_score'])
plt.show()

# %% [code]
plt.plot(train[train['sentiment']=='neutral']['jaccard_score'],'r+')
plt.show()

# %% [code]
def CLEAN_TEXT(text):

    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\s+|www\.\s+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# %% [code]
def CLEAN_TEXT1(text):

    # TOKENIZE TEXT AND REMOVE PUNCUTATION
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # REMOVE WORDS THAT CONTAIN NUMBERS
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # REMOVE STOP WORDS
    text = [x for x in text if x not in stop]
    # REMOVE EMPTY TOKENS
    text = [t for t in text if len(t) > 0]
    # REMOVE WORDS WITH ONLY ONE LETTER
    text = [t for t in text if len(t) > 1]
    # JOIN ALL
    text = " ".join(text)
    return(text)

# %% [code]
train['text'] = train['text'].apply(str).apply(lambda x: CLEAN_TEXT(x))
train['selected_text'] = train.selected_text.apply(str).apply(lambda x: CLEAN_TEXT(x))

# %% [code]
train['CLEANED_TEXT'] = train['text'].apply(lambda x: CLEAN_TEXT1(x))
train['CLEANED_SELECTED_TEXT'] = train.selected_text.apply(lambda x: CLEAN_TEXT1(x))

# %% [code]
train.head(3)

# %% [code]
word_token = word_tokenize("".join(train['CLEANED_SELECTED_TEXT']))
print(word_token[:50])

# %% [code]
most_comman_token_15 = Counter(word_token).most_common(15)
most_comman_token_15_df = pd.DataFrame(most_comman_token_15)
most_comman_token_15_df.columns = ['word','count']
most_comman_token_15_df.style.background_gradient(cmap='Blues')

# %% [code]
def plot_wordcloud(text,mask=None,max_words=400,max_font_size=100,figure_size=(24.0,16.0),title=None,title_size=40,image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords={'u',"im"}
    stopwords=stopwords.union(more_stopwords)
    
    wordcloud = WordCloud(background_color='White',
                         stopwords = stopwords,max_words=max_words,
                         max_font_size=max_font_size,random_state=42,mask=mask)
    
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = imagegenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors),interpolation="BILINEAR");
        plt.title(title,fontdict={'SIZE':title_size,
                                  'VERTICALALIGNMENT':'bottom'})
    else:
            plt.imshow(wordcloud);
            plt.title(title,fontdict={'SIZE':title_size,'COLOR':'RED',
                                     'VERTICALALIGNMENT':'bottom'})
            plt.axis('OFF');
    plt.tight_layout()  
    
D = '/kaggle/input/imagetc/'


# %% [code]
positive_sentiment = train[train['sentiment']=='positive']
negative_sentiment = train[train['sentiment']=='negative']
neutral_sentiment = train[train['sentiment']=='neutral']

# %% [code]
plt.figure(figsize=(8,6))
sns.kdeplot(neutral_sentiment['NO_WORDS_ST'],shade=True,COLOR='B',label='NEU_NO_WORDS_ST')
sns.kdeplot(neutral_sentiment['NO_WORDS_T'],shade=True,COLOR='R',label='NEU_NO_WORDS_T')
plt.title('DISTRIBUTION OF NUMBER OF WORDS IN SELECTED TEXT & TEXT IN NEUTRAL DATAFRAME')
plt.show()

# %% [code]
word_token_pos = word_tokenize("".join(positive_sentiment['CLEANED_SELECTED_TEXT']))
print(word_token_pos[:50])

# %% [code]
most_comman_token_15_pos = Counter(word_token_pos).most_common(15)
most_comman_token_15_pos_df = pd.DataFrame(most_comman_token_15_pos)
most_comman_token_15_pos_df.columns = ['word','count']
most_comman_token_15_pos_df.style.background_gradient(cmap='Blues')

# %% [code]
twitter_mask=np.array(Image.open(D+'twitter.png'))
plot_wordcloud(positive_sentiment.text,mask=twitter_mask,max_font_size=80,title_size=30,title="WORDCLOUD FOR POSITIVE TWEETS")

# %% [code]
word_token_neg = word_tokenize("".join(negative_sentiment['CLEANED_SELECTED_TEXT']))
print(word_token_neg[:50])

# %% [code]
most_comman_token_15_neg = Counter(word_token_neg).most_common(15)
most_comman_token_15_neg_df = pd.DataFrame(most_comman_token_15_neg)
most_comman_token_15_neg_df.columns = ['word','count']
most_comman_token_15_neg_df.style.background_gradient(cmap='Reds')

# %% [code]

twitter_mask=np.array(Image.open(D+'twitter.png'))
plot_wordcloud(negative_sentiment.text,mask=twitter_mask,max_font_size=80,title_size=30,title="WORDCLOUD FOR NEGATIVE TWEETS")

# %% [code]

word_token_neu = word_tokenize("".join(neutral_sentiment['CLEANED_SELECTED_TEXT']))
print(word_token_neu[:50])

# %% [code]

most_comman_token_15_neu = Counter(word_token_neu).most_common(15)
most_comman_token_15_neu_df = pd.DataFrame(most_comman_token_15_neu)
most_comman_token_15_neu_df.columns = ['word','count']
most_comman_token_15_neu_df.style.background_gradient(cmap='Greens')

# %% [code]

twitter_mask=np.array(Image.open(D+'twitter.png'))
plot_wordcloud(neutral_sentiment.text,mask=twitter_mask,max_font_size=80,title_size=30,title="WORDCLOUD FOR NEUTRAL TWEETS")

# %% [code]
def get_top_n_words(corpus,n_grams=None):
    vec = CountVectorizer(ngram_range=(n_grams,n_grams)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    
    sum_of_words = bag_of_words.sum(axis=0)
    word_freq = [(word, sum_of_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    word_freq = sorted(word_freq, key = lambda x: x[1], reverse=True)
    return word_freq[:15]

# %% [code]
top_n_bigrams = get_top_n_words(train['text'].dropna(),2)
x,y = map(list,zip(*top_n_bigrams))
plt.figure(figsize=(9,7))
sns.barplot(x=y,y=x)
plt.show()

# %% [code]
top_n_bigrams = get_top_n_words(train['selected_text'].dropna(),2)
x,y = map(list,zip(*top_n_bigrams))
plt.figure(figsize=(9,7))
sns.barplot(x=y,y=x)
plt.show()

# %% [code]

top_n_trigrams = get_top_n_words(train['text'].dropna(),3)
x,y = map(list,zip(*top_n_trigrams))
plt.figure(figsize=(9,7))
sns.barplot(x=y,y=x)
plt.show()

# %% [code]
top_n_trigrams = get_top_n_words(train['selected_text'].dropna(),3)
x,y = map(list,zip(*top_n_trigrams))
plt.figure(figsize=(9,7))
sns.barplot(x=y,y=x)
plt.show()

# %% [code]
top_n_trigrams_pos = get_top_n_words(positive_sentiment['text'].dropna(),3)
x,y = map(list,zip(*top_n_trigrams_pos))
plt.figure(figsize=(9,7))
sns.barplot(x=y,y=x)
plt.show()

# %% [code]
top_n_trigrams_neg = get_top_n_words(negative_sentiment['text'].dropna(),3)
x,y = map(list,zip(*top_n_trigrams_neg))
plt.figure(figsize=(9,7))
sns.barplot(x=y,y=x)
plt.show()

# %% [code]
top_n_trigrams_neu = get_top_n_words(neutral_sentiment['text'].dropna(),3)
x,y = map(list,zip(*top_n_trigrams_neu))
plt.figure(figsize=(9,7))
sns.barplot(x=y,y=x)
plt.show()

# %% [code]
data_copy = train.copy()
data_train = data_copy[data_copy['NO_WORDS_T']>=3]

# %% [code]
data_train.head()

# %% [code]
def get_training_data(sentiment):
    train_data=[]
    
    '''
    RETURNS TRAINING DATA IN THE FORMAT NEEDED TO TRAIN SPACY NER
    '''
    for index,row in data_train.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.CLEANED_SELECTED_TEXT
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            train_data.append((text, {"entities": [[start,end,'selected_text']]}))
    return train_data

# %% [code]
def training(train_data, output_dir, n_iter=20, model=None):
    """LOAD THE MODEL,SET UP THE PIPELINE AND TRAIN THE ENTITY RECOGNIZER"""
    if model is not None:
        nlp=spacy.load(model) #LOAD EXISTING SPACY MODEL
        print("LOADED MODEL '%S'" %model)
    else:
        nlp = spacy.blank("en") #CREATE BLANK LANGUAGE CLASS
        print("CREATED BLANK 'en' MODEL ")
        
        # THE PIPELINE EXECUTION
        # CREATE THE BUILT-IN PIPELINE COMPONENTS AND THEM TO THE PIPELINE
        # NLP.CREATE_PIPE WORKS FOR BUILT-INS THAT ARE REGISTERED IN THE SPACY
        
        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner,last=True)
            
             # OTHERWISE, GET IT SO WE CAN ADD LABELS
        
        else:
            ner = nlp.get_pipe("ner")
            
        # ADD LABELS 
        for _, annotations in train_data:
                for ent in annotations.get("entities"):
                    ner.add_label(ent[2])
                    # GET NAMES OF OTHER PIPES TO DISABLE THEM DURING TRAINING
        
        pipe_exceptions = ["ner","trf_wordpiecer","trf_tok2vec"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
        
        with nlp.disable_pipes(*other_pipes): # TRAINING OF ONLY NER
            
             # RESET AND INTIALIZE THE WEIGHTS RANDOML - BUT ONLY IF WE'RE
            # TRAINING A MODEL
            
            if model is None:
                nlp.begin_training()
            else:
                nlp.resume_training()
            
            for itn in trange(n_iter):
                random.shuffle(train_data)
                losses={}
                # BATCH UP THE EXAMPLE USING SPACY'S MNIBATCH
                batches = minibatch(train_data,size=compounding(4.0,1000.0,1.001))
                #PRINT(BATCHES)
                for batch in batches:
                    texts , annotations = zip(*batch)
                    nlp.update(
                        texts, #BATCH OF TEXTS
                        annotations, # BATCH OF ANNOTATIONS
                        drop = 0.5,  # DROPOUT - MAKE IT HARDER TO MEMORISE DATA
                         losses = losses,
                )
            print("losses", losses)
        save_model(output_dir, nlp, 'st_ner')
        
        

# %% [code]
def get_model_path(sentiment):
    model_out_path = None 
    if sentiment == 'positive':
        model_out_path = 'models/model_pos'
    elif sentiment == 'negative':
        model_out_path = 'models/model_neg'
    return model_out_path

# %% [code]
def save_model(output_dir,nlp,new_model_name):
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("SAVED MODEL TO",output_dir)

# %% [code]
sentiment ='positive'
train_data = get_training_data(sentiment)
model_path = get_model_path(sentiment)
training(train_data,model_path,n_iter=3,model=None)

# %% [code]
sentiment ='negative'
train_data = get_training_data(sentiment)
model_path = get_model_path(sentiment)
training(train_data,model_path,n_iter=3,model=None)

# %% [code]
model_path = '/kaggle/working/models/'
model_path_pos = model_path + 'model_pos'
model_path_neg = model_path + 'model_neg'

# %% [code]
def predict(text,model):
    docx = model(text)
    ent_arr=[]
    for ent in docx.ents:
        #PRINT(ENT.TEXT)
        start = text.find(ent.text)
        end = start + len(ent.text)
        entity_arr = [start,end,ent.label_]
        if entity_arr not in ent_arr:
            ent_arr.append(entity_arr)
    selected_text = text[ent_arr[0][0]:ent_arr[0][1]] if len(ent_arr)>0 else text
    return selected_text

# %% [code]
selected_text=[]
if model_path is not None:
    print("LOADING MODELS  FROM ", model_path)
    model_pos = spacy.load(model_path_pos)
    model_neg = spacy.load(model_path_neg)
    for index,row in test.iterrows():
        text = row.text.lower()
        if row.sentiment == 'neutral':
            selected_text.append(text)
        elif row.sentiment == 'positive':
            selected_text.append(predict(text,model_pos))
        else:
            selected_text.append(predict(text,model_neg))       

# %% [code]
assert len(test.text) == len(selected_text)
submission['selected_text'] = selected_text
submission.to_csv('submission.csv',index=False)

# %% [code]
from IPython.core.display import HTML
def multi_table(table_list):
    ''' ACCEPS A LIST OF IPYTABLE OBJECTS AND RETURNS A TABLE WHICH CONTAINS EACH IPYTABLE IN A CELL
    '''
    return HTML(
        '<TABLE><TR STYLE="BACKGROUND-COLOR:WHITE;">' + 
        ''.join(['<TD>' + table._repr_html_() + '</TD>' for table in table_list]) +
        '</TR></TABLE>'
    )

# %% [code]
multi_table([test.head(10),submission.head(10)])

# %% [code]
