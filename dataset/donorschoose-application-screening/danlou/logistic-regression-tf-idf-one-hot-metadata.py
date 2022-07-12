import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score


def extract_texts(df):
    """ Combines the text fields of all submissions. """
    submission_texts = {}
    for idx, submission in df.iterrows():
        sub_id = submission.name
        sub_text = submission['project_title']
        sub_text += '. ' + submission['project_subject_categories']
        sub_text += '. ' + submission['project_subject_subcategories']    
        sub_text += '. ' + submission['project_essay_1']
        sub_text += ' ' + submission['project_essay_2']
        sub_text += ' ' + submission['project_essay_3']
        sub_text += ' ' + submission['project_essay_4']
        sub_text += ' ' + submission['project_resource_summary']
        sub_text += ' ' + submission['description']

        # clean whitespace
        sub_text = sub_text.replace('\\n', ' ')
        sub_text = sub_text.replace('\\r', ' ')
        sub_text = ' '.join(sub_text.split())

        submission_texts[sub_id] = sub_text

    return submission_texts


def tokenize_texts(submission_texts, tokenizer='spacy'):
    """
    Tokenizes all extracted texts.
    We provide implementations for two rule-based tokenizers:
    spacy - better, handles many exceptions
    regex - faster, based on spacing out punctuation and splitting by whitespace
    """    
    submission_texts_tokenized = {}
    
    if tokenizer == 'spacy':
        import spacy
        nlp = spacy.load('en', disable=['parser'])

        ids, texts = submission_texts.keys(), submission_texts.values()
        for sub_id, doc in zip(ids, nlp.pipe(texts, n_threads=16, batch_size=10000)):
            doc_tokens = []
            for token in doc:
                if not token.is_stop:
                    doc_tokens.append(token.text.lower())
            submission_texts_tokenized[sub_id] = ' '.join(doc_tokens)
            
            # takes roughly one hour for all texts ...
            if len(submission_texts_tokenized) % 1000 == 0:
                print(datetime.now(), 'at ', len(submission_texts_tokenized))
    
    elif tokenizer == 'regex':
        import re
        from nltk.corpus import stopwords
        stopwords_en = set(stopwords.words('english'))

        for sub_id, text in submission_texts.items():
            # text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub('([.,!?()])', r' \1 ', text)

            tokens = text.lower().split()
            tokens = [tok for tok in tokens if tok not in stopwords_en]
            submission_texts_tokenized[sub_id] = ' '.join(tokens)

    return submission_texts_tokenized


print(datetime.now(), 'Loading Data ...')

#base_path = '~/.kaggle/competitions/donorschoose-application-screening'
base_path = '../input'


train = pd.read_csv(base_path + "/train.csv", index_col="id", low_memory=False).fillna("NA")
test = pd.read_csv(base_path + "/test.csv", index_col="id", low_memory=False).fillna("NA")
#train = train.sample(10000, random_state=42)  # for quick runs

resources = pd.read_csv(base_path + "/resources.csv", index_col="id").fillna("NA")
resources_agg = resources.reset_index().groupby('id').agg(
    dict(quantity='sum', price='sum', description=lambda x: ' '.join(x)))

df = pd.concat([train.drop("project_is_approved", axis=1), test], axis=0)
df = pd.merge(df, resources_agg, left_index=True, right_index=True, how="inner")
df = df.fillna("NA")

sum_prices = resources_agg.get('price').to_dict()
sum_nitems = resources_agg.get('quantity').to_dict()


print(datetime.now(), 'Compiling Submission Texts ...')
load_processed = False  # just for spacy (which takes much longer than regex)
if load_processed:
    submission_texts = pickle.load(open('processed_texts.p', 'rb'))
else:
    tokenizer = 'regex'
    submission_texts = extract_texts(df)
    submission_texts = tokenize_texts(submission_texts, tokenizer)
    
    if tokenizer == 'spacy':
        pickle.dump(submission_texts, open('processed_texts.p', 'wb'))


# unfiltered; 4,938,845 unique ngrams
# min_df = 0.001; 28,946 unique ngrams
# min_df = 0.0001; 247,854 unique ngrams
print(datetime.now(), "Preparing Text Features ...")
text_vectorizer = TfidfVectorizer(max_features=None,
                                  min_df=0.001,
                                  ngram_range=(1, 2),
                                  tokenizer=lambda x: x.split(),
                                  dtype=np.float32)
text_vectorizer.fit([submission_texts[idx] for idx in train.index])
train_features_text = text_vectorizer.transform([submission_texts[i] for i in train.index])
print(datetime.now(), "# Ngrams:", train_features_text.shape[1])


print(datetime.now(), "Preparing Price Features ...")
def price_binner(prices):
    return np.digitize(prices, [10, 100, 500, 1000, 5000, 10000])

price_encoder = LabelBinarizer(sparse_output=False)
prices_binned = price_binner([sum_prices[i] for i in train.index])
train_features_price = price_encoder.fit_transform(prices_binned)


print(datetime.now(), "Preparing Quantity Features ...")
def quantity_binner(quantities):
    return np.digitize(quantities, [1, 5, 10, 100, 500])

nitems_encoder = LabelBinarizer(sparse_output=True)
nitems_binned = quantity_binner([sum_nitems[i] for i in train.index])
train_features_nitems = nitems_encoder.fit_transform(nitems_binned)


print(datetime.now(), "Preparing Location Features ...")
sub_states = df.get('school_state').to_dict()
state_encoder = LabelBinarizer(sparse_output=True)
train_features_state = state_encoder.fit_transform([sub_states[idx] for idx in train.index])


print(datetime.now(), "Preparing Grade Features ...")
sub_grades = df.get('project_grade_category').to_dict()
grade_encoder = LabelBinarizer(sparse_output=True)
train_features_grade = grade_encoder.fit_transform([sub_grades[idx] for idx in train.index])


print(datetime.now(), "Preparing Category Features ...")
sub_cats = df.get('project_subject_categories').to_dict()
cats_encoder = LabelBinarizer(sparse_output=True)
train_features_cats = cats_encoder.fit_transform([sub_cats[idx] for idx in train.index])


print(datetime.now(), "Preparing SubCategory Features ...")
sub_subcats = df.get('project_subject_subcategories').to_dict()
subcats_encoder = LabelBinarizer(sparse_output=True)
train_features_subcats = subcats_encoder.fit_transform([sub_subcats[idx] for idx in train.index])


print(datetime.now(), "Stacking and Scaling Features ...")
from scipy.sparse import hstack as sp_hstack

train_features = sp_hstack([train_features_text,
                            train_features_price,
                            train_features_nitems,
                            train_features_cats,
                            train_features_subcats,
                            train_features_state,
                            train_features_grade])

# train_features = scale(train_features.tocsc(), with_mean=False)  # not required for LogReg


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

classifiers = {}
classifiers['LogReg'] = LogisticRegression(solver="sag", max_iter=1000, n_jobs=6)
classifiers['SVM'] = LinearSVC()

for clf_name, clf in classifiers.items():
    print(datetime.now(), "Building %s Classifier ..." % clf_name)

    clf.fit(train_features, train["project_is_approved"])
    cv_scores = cross_val_score(clf, train_features, train["project_is_approved"],
                                cv=3, scoring='roc_auc')
    
    print(datetime.now(), "%s Train AUC: %.3f" % (clf_name, np.mean(cv_scores)))


print(datetime.now(), "Preparing Test Features ...")
test_features_text    = text_vectorizer.transform([submission_texts[i] for i in test.index])
test_features_price   = price_encoder.transform(price_binner([sum_prices[i] for i in test.index]))
test_features_nitems  = nitems_encoder.transform(quantity_binner([sum_nitems[i] for i in test.index]))
test_features_state   = state_encoder.transform([sub_states[idx] for idx in test.index])
test_features_grade   = grade_encoder.transform([sub_grades[idx] for idx in test.index])
test_features_cats    = cats_encoder.transform([sub_cats[idx] for idx in test.index])
test_features_subcats = subcats_encoder.transform([sub_subcats[idx] for idx in test.index])

test_features = sp_hstack([test_features_text,
                           test_features_price,
                           test_features_nitems,
                           test_features_cats,
                           test_features_subcats,
                           test_features_state,
                           test_features_grade])

#test_features = scale(test_features.tocsc(), with_mean=False)  # not required for LogReg

# LogReg consistently performs best
preds = classifiers['LogReg'].predict_proba(test_features)[:, 1]
sub = pd.DataFrame(preds, columns=["project_is_approved"], index=test.index)
sub.to_csv("sub_logreg.csv", index=True)
