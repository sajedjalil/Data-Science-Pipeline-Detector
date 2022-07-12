"""this is my first competition submission on kaggle with not much time on my hands,
and still getting acquainted with python.
I claim it  to be neither beautiful nor efficient. Hopefully it will get a decent
prediction though. The kernel is pretty simple, a naive Bayes classifier derived from
the book "Data Science from Scratch" by Joel Grus, where it was described for spam filtering."""

import math
import re
import random
from collections import defaultdict
import numpy as np
import pandas as pd

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

def tokenize(text):
    text = text.lower()
    all_words = re.findall("[a-z0-9]+", text)
    return set(all_words)
    
def count_words(training_set):
    counts = defaultdict(lambda: {'EAP': 0, 'HPL': 0, 'MWS': 0})
    for item in training_set:
        for word in tokenize(item['text']):
            counts[word][item['author']] += 1
    return counts

def word_probabilities(counts, total_eap, total_hpl, total_mws, k=0.33):
    probs = [{'word': w,
            'eap': (d['EAP'] + k) / (total_eap + 3 * k),
            'hpl': (d['HPL'] + k) / (total_hpl + 3 * k),
            'mws': (d['MWS'] + k) / (total_mws + 3 * k)}
            for w, d in counts.items()]
    return probs
    
def author_probability(word_probs, text):
    text_words = tokenize(text)
    log_prob_if_eap = 0.0
    log_prob_if_hpl = 0.0
    log_prob_if_mws = 0.0
    for d in word_probs:
        if d['word'] in text_words:
            log_prob_if_eap += math.log(d['eap'])
            log_prob_if_hpl += math.log(d['hpl'])
            log_prob_if_mws += math.log(d['mws'])
        else:
            log_prob_if_eap += math.log(1.0 - d['eap'])
            log_prob_if_hpl += math.log(1.0 - d['hpl'])
            log_prob_if_mws += math.log(1.0 - d['mws'])
    prob_if_eap = math.exp(log_prob_if_eap)
    prob_if_hpl = math.exp(log_prob_if_hpl)
    prob_if_mws = math.exp(log_prob_if_mws)
    if (prob_if_eap + prob_if_hpl + prob_if_mws) > 0:
        divisor = prob_if_eap + prob_if_hpl + prob_if_mws
    else:
        divisor = 1 #the result of the division is zero anyways
    eap = prob_if_eap/divisor
    hpl = prob_if_hpl/divisor
    mws = prob_if_mws/divisor
    #return max(probs, key=probs.get), (probs[max(probs, key=probs.get)]/divisor)
    probs = {"EAP": eap, "HPL": hpl, "MWS": mws}
    return probs
    
class NaiveBayesClassifier:
    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []
        
    def train(self, training_set):
        num_eap = len([idx for idx, row in training_set.iterrows() if row.author == "EAP"])
        num_hpl = len([idx for idx, row in training_set.iterrows() if row.author == "HPL"])
        num_mws = len([idx for idx, row in training_set.iterrows() if row.author == "MWS"])
        word_counts = count_words([row for idx, row in training_set.iterrows()])
        self.word_probs = word_probabilities(word_counts, num_eap, num_hpl, num_mws, self.k)
        
    def classify(self, text):
        return author_probability(self.word_probs, text)

training_data = pd.read_csv("../input/train.csv")
#mask = np.random.rand(len(training_data)) < 0.75
#train_data = training_data[mask]
#test_data = training_data[~mask]
classifier = NaiveBayesClassifier(0.333)
#classifier.train(train_data)
classifier.train(training_data)
#classified = [(row.text, row.author, classifier.classify(row.text)) for row in 
#                [row for idx, row in test_data.iterrows()]]
#correct = 0
#incorrect = 1
#for text, author, classification in classified:
#    if author == classification[0]:
#        correct += 1
#    else:
#        incorrect += 1

#print("Correct percentage:", correct / (correct + incorrect))

predict_data = pd.read_csv("../input/test.csv")
prediction = [
    (   row.id, 
        classifier.classify(row.text)['EAP'],
        classifier.classify(row.text)['HPL'], # I know, this really is a sin.
        classifier.classify(row.text)['MWS']  # again
    ) for row in [row for idx, row in predict_data.iterrows()]]
    
file = open("./out.txt","w") 
 
file.write("\"id\",\"EAP\",\"HPL\",\"MWS\"\n")
for ln in prediction:
    file.write("\""+str(ln[0])+"\","+str(ln[1])+","+str(ln[2])+","+str(ln[3])+"\n")
file.close()