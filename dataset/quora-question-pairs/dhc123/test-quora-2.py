# author: alijs
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import operator
import timeit
from sklearn.cross_validation import train_test_split
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from pylab import plot, show, subplot, specgram, imshow, savefig
import nltk
from nltk.stem import SnowballStemmer
import re
from string import punctuation
RS = 12357
ROUNDS = 230

print("Started")
np.random.seed(RS)
input_folder = '../input/'

stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']

def train_xgb(X, y, params):
	print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))
	x, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RS)

	xg_train = xgb.DMatrix(x, label=y_train)
	xg_val = xgb.DMatrix(X_val, label=y_val)

	watchlist  = [(xg_train,'train'), (xg_val,'eval')]
	return xgb.train(params, xg_train, ROUNDS, watchlist)

def predict_xgb(clr, X_test):
	return clr.predict(xgb.DMatrix(X_test))

def create_feature_map(features):
	outfile = open('xgb.fmap', 'w')
	i = 0
	for feat in features:
		outfile.write('{0}\t{1}\tq\n'.format(i, feat))
		i = i + 1
	outfile.close()

def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.

    # Clean the text
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
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
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
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

def process_questions(question_list, questions):
    '''transform questions and display progress'''
    for question in questions:
        question_list[question]=text_to_wordlist(question)

def main():
	params = {}
	params['objective'] = 'binary:logistic'
	params['eval_metric'] = 'logloss'
	params['eta'] = 0.11
	params['max_depth'] = 8
	params['silent'] = 1
	params['seed'] = RS

	df_train = pd.read_csv(input_folder + 'train.csv')
	df_test  = pd.read_csv(input_folder + 'test.csv')
	print("Original data: X_train: {}, X_test: {}".format(df_train.shape, df_test.shape))

	print("Features processing, be patient...")

	# If a word appears only once, we ignore it completely (likely a typo)
	# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
	def get_weight(count, eps=10000, min_count=2):
		return 0 if count < min_count else 1 / (count + eps)

	train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
	words = (" ".join(train_qs)).lower().split()
	counts = Counter(words)
	weights = {word: get_weight(count) for word, count in counts.items()}

	stops = set(stopwords.words("english"))
	def word_shares(row):
		q1 = set(str(row['question1']).lower().split())
		q1words = q1.difference(stops)
		if len(q1words) == 0:
			return '0:0:0:0:0'

		q2 = set(str(row['question2']).lower().split())
		q2words = q2.difference(stops)
		if len(q2words) == 0:
			return '0:0:0:0:0'

		q1stops = q1.intersection(stops)
		q2stops = q2.intersection(stops)

		shared_words = q1words.intersection(q2words)
		shared_weights = [weights.get(w, 0) for w in shared_words]
		total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
		
		R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share
		R2 = len(shared_words) / (len(q1words) + len(q2words)) #count share
		R31 = len(q1stops) / len(q1words) #stops in q1
		R32 = len(q2stops) / len(q2words) #stops in q2
		return '{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32)

	df = pd.concat([df_train, df_test])
	print("Data clean")
	process_questions(df[['question1']], df[['question1']])
	process_questions(df[['question2']], df[['question2']])
	print("Data clean done")
	df['word_shares'] = df.apply(word_shares, axis=1, raw=True)
    
	x = pd.DataFrame()

	#x['word_match']       = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
	#x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
	#x['shared_count']     = df['word_shares'].apply(lambda x: float(x.split(':')[2]))
	
	train_orig =  df_train
	test_orig  =  df_test
	
	
	tic0=timeit.default_timer()
	df1 = train_orig[['question1']].copy()
	df2 = train_orig[['question2']].copy()
	df1_test = test_orig[['question1']].copy()
	df2_test = test_orig[['question2']].copy()
	
	df2.rename(columns = {'question2':'question1'},inplace=True)
	df2_test.rename(columns = {'question2':'question1'},inplace=True)
	
	train_questions = df1.append(df2)
	train_questions = train_questions.append(df1_test)
	train_questions = train_questions.append(df2_test)
	train_questions.drop_duplicates(subset = ['question1'],inplace=True)
	
	train_questions.reset_index(inplace=True,drop=True)
	questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()
	train_cp = train_orig.copy()
	test_cp = test_orig.copy()
	train_cp.drop(['qid1','qid2'],axis=1,inplace=True)
	
	test_cp['is_duplicate'] = -1
	test_cp.rename(columns={'test_id':'id'},inplace=True)
	comb = train_cp.append(test_cp)
	comb['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
	comb['word_match']       = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
	comb['shared_count']     = df['word_shares'].apply(lambda x: float(x.split(':')[2]))
	
	comb['how'] = df['question1'].apply(lambda x: ("how" in str(x).lower())*1)*df['question2'].apply(lambda x: ("how" in str(x).lower())*1)
	comb['what'] = df['question1'].apply(lambda x: ("what" in str(x).lower())*1)*df['question2'].apply(lambda x: ("what" in str(x).lower())*1)
	comb['which'] = df['question1'].apply(lambda x: ("which" in str(x).lower())*1)*df['question2'].apply(lambda x: ("which" in str(x).lower())*1)
	comb['why'] = df['question1'].apply(lambda x: ("why" in str(x).lower())*1)*df['question2'].apply(lambda x: ("why" in str(x).lower())*1)
	comb['when'] = df['question1'].apply(lambda x: ("when" in str(x).lower())*1)*df['question2'].apply(lambda x: ("when" in str(x).lower())*1)
	comb['where'] = df['question1'].apply(lambda x: ("where" in str(x).lower())*1)*df['question2'].apply(lambda x: ("where" in str(x).lower())*1)
	comb['learn'] = df['question1'].apply(lambda x: ("learn" in str(x).lower())*1)*df['question2'].apply(lambda x: ("learn" in str(x).lower())*1)
	comb['book'] = df['question1'].apply(lambda x: ("book" in str(x).lower())*1)*df['question2'].apply(lambda x: ("book" in str(x).lower())*1)

	print(comb['how'])
    
	comb['q1_hash'] = comb['question1'].map(questions_dict)
	comb['q2_hash'] = comb['question2'].map(questions_dict)
	
	q1_vc = comb.q1_hash.value_counts().to_dict()
	q2_vc = comb.q2_hash.value_counts().to_dict()
	def try_apply_dict(x,dict_to_apply):
	    try:
	        return dict_to_apply[x]
	    except KeyError:
	        return 0
	comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
	comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
	
	train_comb = comb[:df_train.shape[0]][['q1_hash','q2_hash','q1_freq','q2_freq','tfidf_word_match','how','what','which','why','word_match','shared_count','when','where','learn','book']]
	test_comb = comb[df_train.shape[0]:][['q1_hash','q2_hash','q1_freq','q2_freq','tfidf_word_match','how','what','which','why','word_match','shared_count','when','where','learn','book']]
	#x['stops1_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
	#x['stops2_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
	#x['diff_stops_r']     = x['stops1_ratio'] - x['stops2_ratio']

	#x['len_q1'] = df['question1'].apply(lambda x: len(str(x)))
	#x['len_q2'] = df['question2'].apply(lambda x: len(str(x)))
	#x['diff_len'] = x['len_q1'] - x['len_q2']

	#x['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))
	#x['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))
	#x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']

	#x['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))
	#x['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))
	#x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']

	#x['avg_world_len1'] = x['len_char_q1'] / x['len_word_q1']
	#x['avg_world_len2'] = x['len_char_q2'] / x['len_word_q2']
	#x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']

	#x['exactly_same'] = (df['question1'] == df['question2']).astype(int)
	#x['duplicated'] = df.duplicated(['question1','question2']).astype(int)
	#x['how'] = df['question1'].apply(lambda x: ("how" in str(x).lower())*1)*df['question2'].apply(lambda x: ("how" in str(x).lower())*1)
	#x['q2_how'] = df['question2'].apply(lambda x: ("how" in str(x).lower())*1)
	#x['what'] = df['question1'].apply(lambda x: ("what" in str(x).lower())*1)*df['question2'].apply(lambda x: ("what" in str(x).lower())*1)
	#x['q2_what'] = df['question2'].apply(lambda x: ("what" in str(x).lower())*1)
	
	#x['which'] = df['question1'].apply(lambda x: ("which" in str(x).lower())*1)*df['question2'].apply(lambda x: ("which" in str(x).lower())*1)
	#x['q2_which'] = df['question1'].apply(lambda x: ("what" in str(x).lower())*1)

	#x['where'] = df['question1'].apply(lambda x: ("where" in str(x).lower())*1)*df['question2'].apply(lambda x: ("where" in str(x).lower())*1)
	#x['q2_where'] = df['question1'].apply(lambda x: ("where" in str(x).lower())*1)
	
	#x['when'] = df['question1'].apply(lambda x: ("when" in str(x).lower())*1)*df['question2'].apply(lambda x: ("when" in str(x).lower())*1)
	#x['q2_when'] = df['question1'].apply(lambda x: ("when" in str(x).lower())*1)

    #x['q1_which'] = ("which" in df['question1'].lower()).astype(int)
    #x['q2_which'] = ("which" in df['question2'].lower()).astype(int)

    #x['q1_where'] = ("where" in df['question1'].lower()).astype(int)
    #x['q2_where'] = ("where" in df['question2'].lower()).astype(int)

    #x['q1_when'] = ("when" in df['question1'].lower()).astype(int)
    #x['q2_when'] = ("when" in df['question2'].lower()).astype(int)
	#... YOUR FEATURES HERE ...
	
	feature_names = list(x.columns.values)
	create_feature_map(feature_names)
	print("Features: {}".format(feature_names))

	#x_train = x[:df_train.shape[0]]
	#x_test  = x[df_train.shape[0]:]
	x_train = train_comb
	x_test  = test_comb
	y_train = df_train['is_duplicate'].values
	del x, df_train

	if 1: # Now we oversample the negative class - on your own risk of overfitting!
		pos_train = x_train[y_train == 1]
		neg_train = x_train[y_train == 0]

		print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
		p = 0.165
		scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
		while scale > 1:
			neg_train = pd.concat([neg_train, neg_train])
			scale -=1
		neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
		print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))

		x_train = pd.concat([pos_train, neg_train])
		y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
		del pos_train, neg_train
	
	print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(x_train.shape, len(y_train), x_test.shape))
	clr = train_xgb(x_train, y_train, params)
	preds = predict_xgb(clr, x_test)

	print("Writing output...")
	sub = pd.DataFrame()
	sub['test_id'] = df_test['test_id']
	sub['is_duplicate'] = preds
	sub.to_csv("xgb_seed{}_n{}.csv".format(RS, ROUNDS), index=False)

	print("Features importances...")
	importance = clr.get_fscore(fmap='xgb.fmap')
	importance = sorted(importance.items(), key=operator.itemgetter(1))
	ft = pd.DataFrame(importance, columns=['feature', 'fscore'])

	ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
	plt.gcf().savefig('features_importance.png')

main()
print("Done.")