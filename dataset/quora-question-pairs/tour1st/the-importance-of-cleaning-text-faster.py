# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Borrowed from https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text/notebook

import pandas as pd
import numpy as numpy
import nltk
from nltk.corpus import stopwords
from string import punctuation
import re
import time
from multiprocessing import Process
import _pickle as pickle
import os

PATH = '../input/'

def timeit(f):
	''' Timing decorator '''
	def timed(*args, **kw):
		ts = time.time()
		result = f(*args, **kw)
		te = time.time()
		print('* func:{0!s} took: {1:2.4f} sec\n'.format(f.__name__, te-ts))
		return result
	return timed


@timeit
def read_data(loc):
	''' Read data '''
	df = pd.read_csv(loc)
	print('df shape:', df.shape)
	print('\n/// Missing value checks')
	print(df.isnull().sum()) # Check missing counts
	df.fillna('empty') # Fill missing
	# Sample data
	print('\n/// Sample data')
	for q1, q2 in zip(df.question1[:5], df.question2[:5]):
		print(q1)
		print(q2+'\n')
	print('-'*100)
	return df


def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
	''' Clean text '''
	text = re.sub(r"[^A-Za-z0-9]", " ", text)
	text = re.sub(r"what's", "", text)
	text = re.sub(r"What's", "", text)
	text = re.sub(r"\'s", " ", text)
	text = re.sub(r"\'ve", " have ", text)
	text = re.sub(r"can't", "cannot ", text)
	text = re.sub(r"n't", " not ", text)
	text = re.sub(r"I'm", "I am", text)
	text = re.sub(r" m ", " am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)
	text = re.sub(r"60k", " 60000 ", text)
	text = re.sub(r"\be g\b", " eg ", text)
	text = re.sub(r"\bb g\b", " bg ", text)
	text = re.sub(r"\0s", "0", text)
	text = re.sub(r"\b9 11\b", "911", text)
	text = re.sub(r"e-mail", "email", text)
	text = re.sub(r"\s{2,}", " ", text)
	text = re.sub(r"quikly", "quickly", text)
	text = re.sub(r"\busa\b", " America ", text)
	text = re.sub(r"\bUSA\b", " America ", text)
	text = re.sub(r"\bu s\b", " America ", text)
	text = re.sub(r"\buk\b", " England ", text)
	text = re.sub(r"\bUK\b", " England ", text)
	text = re.sub(r"india", "India", text)
	text = re.sub(r"switzerland", "Switzerland", text)
	text = re.sub(r"china", "China", text)
	text = re.sub(r"chinese", "Chinese", text) 
	text = re.sub(r"imrovement", "improvement", text)
	text = re.sub(r"intially", "initially", text)
	text = re.sub(r"quora", "Quora", text)
	text = re.sub(r"\bdms\b", "direct messages ", text)  
	text = re.sub(r"demonitization", "demonetization", text) 
	text = re.sub(r"actived", "active", text)
	text = re.sub(r"kms", " kilometers ", text)
	text = re.sub(r"KMs", " kilometers ", text)
	text = re.sub(r"\bcs\b", " computer science ", text) 
	text = re.sub(r"\bupvotes\b", " up votes ", text)
	text = re.sub(r"\biPhone\b", " phone ", text)
	text = re.sub(r"\0rs ", " rs ", text) 
	text = re.sub(r"calender", "calendar", text)
	text = re.sub(r"ios", "operating system", text)
	text = re.sub(r"gps", "GPS", text)
	text = re.sub(r"gst", "GST", text)
	text = re.sub(r"programing", "programming", text)
	text = re.sub(r"bestfriend", "best friend", text)
	text = re.sub(r"dna", "DNA", text)
	text = re.sub(r"III", "3", text) 
	text = re.sub(r"the US", "America", text)
	text = re.sub(r"Astrology", "astrology", text)
	text = re.sub(r"Method", "method", text)
	text = re.sub(r"Find", "find", text) 
	text = re.sub(r"banglore", "Bangalore", text)
	text = re.sub(r"bangalore", "Bangalore", text)
	text = re.sub(r"\bJ K\b", " JK ", text)
	
	# Remove punctuation from text
	text = ''.join([c for c in text if c not in punctuation])
	
	# Optionally, remove stop words
	if remove_stop_words:
		text = text.split()
		text = [w for w in text if not w in stop_words]
		text = ' '.join(text)
	
	# Optionally, shorten words to their stems
	if stem_words:
		text = text.split()
		stemmer = SnowballStemmer('english')
		stemmed_words = [stemmer.stem(word) for word in text]
		text = ' '.join(stemmed_words)
	
	# Return a list of words
	return(text)


def process_questions(i, question_list, questions, question_list_name):
	''' Transform questions and display progress '''
	print('processing {}: process {}'.format(question_list_name, i))
	for question in questions:
		question_list.append(text_to_wordlist(str(question)))
	with open('temp_list_'+str(i)+'.pkl', 'wb') as fl_out:
		pickle.dump(question_list, fl_out)


@timeit
def multi(n_cores, tq, qln):
	''' Using multiple processes '''
	procs = []
	clean_txt = {}
	# manager = Manager()
	for i in range(n_cores):
		# clean_txt[i] = manager.list()
		clean_txt[i] = []
 
	for index in range(n_cores):
		tq_indexed = tq[index*len(tq)//n_cores:(index+1)*len(tq)//n_cores]
		proc = Process(target=process_questions, args=(index, clean_txt[index], tq_indexed, qln, ))
		procs.append(proc)
		proc.start()
 
	for proc in procs:
		proc.join()

	clean_txt_list = []		
	for i in range(n_cores):
		with open('temp_list_'+str(i)+'.pkl', 'rb') as fl_in:
			clean_txt[i] = pickle.load(fl_in)
		clean_txt_list += clean_txt[i]
		os.remove('temp_list_'+str(i)+'.pkl')

	print('{} records processed from {}'.format(len(clean_txt_list), qln))
	print('-'*100)
	return(clean_txt_list)


if __name__=='__main__':
	train = read_data(PATH+'train.csv')
	test = read_data(PATH+'test.csv')
	stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that',\
			'these','those','then','just','so','than','such','both','through','about','for','is','of',\
			'while','during','to','What','Which','Is','If','While','This']

	CORES = 4
	train['question1'] = multi(CORES, train.question1, 'train_question1')
	train['question2'] = multi(CORES, train.question2, 'train_question2')
	
	test['question1'] =  multi(CORES, test.question1, 'test_question1')
	test['question2'] =  multi(CORES, test.question2, 'test_question2')

	print('\n/// Sample changes')
	for q1, q2 in zip(train.question1[:5], train.question2[:5]):
		print(q1)
		print(q2+'\n')