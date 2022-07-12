import pandas as pd
# import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv("../input/train.tsv", sep="\t")
test = pd.read_csv('../input/test.tsv', sep="\t")
sub = pd.read_csv('../input/sampleSubmission.csv', sep=",")

tokenizer = TweetTokenizer()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)
full_text = list(train['Phrase'].values) + list(test['Phrase'].values)
vectorizer.fit(full_text)
train_vectorized = vectorizer.transform(train['Phrase'])
test_vectorized = vectorizer.transform(test['Phrase'])


y = train['Sentiment']
logreg = LogisticRegression()
ovr = OneVsRestClassifier(logreg)
ovr.fit(train_vectorized, y)
pred_test1 = ovr.predict(test_vectorized)
sub['Sentiment'] = pred_test1
sub.to_csv("submission_1.csv", index=False)


