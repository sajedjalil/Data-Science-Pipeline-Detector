import csv
import nltk
import numpy as np
from collections import defaultdict
from nltk.util import ngrams
import operator
tot_ngrams=[]

num_gram=4
with open('../input/train.csv', newline='') as csvfile:
     # csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     csvreader = csv.DictReader(csvfile)
     for row in csvreader:
         Id = row['Id']
         sequence = row['Sequence']
         tokens = sequence.split(',')
         n_grams = nltk.ngrams(tokens,num_gram)
         for n in n_grams:
            tot_ngrams.append(n)


def return_defaultdict():
    return defaultdict(int)

counter_dict = defaultdict(return_defaultdict)
max_dict = dict()

total_num_ngrams = len(tot_ngrams)

for n in tot_ngrams:
    n_last = n[0:-1]
    last = n[-1]
    n_last_tup = tuple(n_last)
    counter_dict[n_last_tup][last] = counter_dict[n_last_tup][last] + 1


for second_last, last_counts in counter_dict.items():
    max_dict[second_last] = max(last_counts.items(), key=operator.itemgetter(1))[0]




last_words=[]
Ids = []
num_gram=4
with open('../input/test.csv', newline='') as csvfile:
     csvreader = csv.DictReader(csvfile)
     for row in csvreader:
         # print row
         Id = row['Id']

         Ids.append(Id)
         sequence = row['Sequence']
         tokens = sequence.split(',')

         last_words.append(tokens[-num_gram+1:])




predictions=[]
for ii,seq in enumerate(last_words):
    test_seq = tuple(seq)
    if test_seq in max_dict:
        predictions.append(max_dict[test_seq])
    else:
        predictions.append('0')

with open('some.csv', 'w',newline='') as f:
    fieldnames = ['Id', 'Last']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i,Id in enumerate(Ids):
        writer.writerow({'Id': Id, 'Last': predictions[i]})






