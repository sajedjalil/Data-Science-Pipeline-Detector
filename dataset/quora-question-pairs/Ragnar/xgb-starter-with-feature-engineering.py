import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import operator
from sklearn.cross_validation import train_test_split

# FE Stage 3
from nltk import ngrams
from simhash import Simhash
from sklearn.model_selection import train_test_split
from multiprocessing import Pool # We use pool to speed up feature creation

# Stage 0
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

# Stage 1
from fuzzywuzzy import fuzz

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import timeit
#stop_words = stopwords.words('english')
# import matplotlib.pyplot as plt
# from pylab import plot, show, subplot, specgram, imshow, savefig

RS = 12357
ROUNDS = 500

print("Started")
np.random.seed(RS)
input_folder = '../input/'


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


def main():
	params = {}
	params['objective'] = 'binary:logistic'
	params['eval_metric'] = 'logloss'
	params['eta'] = 0.11
	params['max_depth'] = 5
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
        
        # Concat train and test datasets
	df = pd.concat([df_train, df_test])
	df['word_shares'] = df.apply(word_shares, axis=1, raw=True)
	#Join q1 and q2 separeted by _split_tag_
	# df['questions'] = df['question1'] + '_split_tag_' + df['question2']

	x = pd.DataFrame()

	x['word_match']       = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
	x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
	x['shared_count']     = df['word_shares'].apply(lambda x: float(x.split(':')[2]))

	x['stops1_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
	x['stops2_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
	x['diff_stops_r']     = x['stops1_ratio'] - x['stops2_ratio']

	x['len_q1'] = df['question1'].apply(lambda x: len(str(x)))
	x['len_q2'] = df['question2'].apply(lambda x: len(str(x)))
	x['diff_len'] = x['len_q1'] - x['len_q2']

	x['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))
	x['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))
	x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']

	x['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))
	x['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))
	x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']

	x['avg_world_len1'] = x['len_char_q1'] / x['len_word_q1']
	x['avg_world_len2'] = x['len_char_q2'] / x['len_word_q2']
	x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']
	x['common_words'] = df.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))),axis=1)

	# Stage 2: Fuzzy Features
	print("Calculating Fuzzy Features")
	#x['fuzz_qratio'] = df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])),axis=1)
	#x['fuzz_WRatio'] = df.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])),axis=1)
	#x['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])),axis=1)
	#x['fuzz_partial_token_set_ratio'] = df.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])),axis=1)
	#x['fuzz_partial_token_sort_ratio'] = df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])),axis=1)
	#x['fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])),axis=1)
	#x['fuzz_token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])),axis=1)
	print("Done With Fuzzy Features")
	
	x['exactly_same'] = (df['question1'] == df['question2']).astype(int)
	x['duplicated'] = df.duplicated(['question1','question2']).astype(int)
	
	# FE Stage 4: SimHash Features
	def tokenize(sequence):
		words = word_tokenize(sequence)
		filtered_words = [word for word in words if word not in stopwords.words('english')]
		return filtered_words

	def clean_sequence(sequence):
		tokens = tokenize(sequence)
		return ' '.join(tokens)
	
	def get_word_ngrams(sequence, n=3):
		tokens = tokenize(sequence)
		return [' '.join(ngram) for ngram in ngrams(tokens, n)]
	
	def get_character_ngrams(sequence, n=3):
		sequence = clean_sequence(sequence)
		return [sequence[i:i+n] for i in range(len(sequence)-n+1)]
	
	
	def caluclate_simhash_distance(sequence1, sequence2):
		return Simhash(sequence1).distance(Simhash(sequence2))

	def get_word_distance(questions):
		q1, q2 = questions.split('_split_tag_')
		q1, q2 = tokenize(q1), tokenize(q2)
		return caluclate_simhash_distance(q1, q2)
	
	def get_word_2gram_distance(questions):
		q1, q2 = questions.split('_split_tag_')
		q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
		return caluclate_simhash_distance(q1, q2)
	
	def get_char_2gram_distance(questions):
		q1, q2 = questions.split('_split_tag_')
		q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
		return caluclate_simhash_distance(q1, q2)
	
	def get_word_3gram_distance(questions):
		q1, q2 = questions.split('_split_tag_')
		q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
		return caluclate_simhash_distance(q1, q2)
	
	def get_char_3gram_distance(questions):
		q1, q2 = questions.split('_split_tag_')
		q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
		return caluclate_simhash_distance(q1, q2)
	
	# Instead of 8, swap the number with the number of cpu cores/threads you have
	pool = Pool(processes=4)
	
	x['tokenize_distance'] = pool.map(get_word_distance, df['questions'])
	
	x['word_2gram_distance'] = pool.map(get_word_2gram_distance, df['questions'])
	x['char_2gram_distance'] = pool.map(get_char_2gram_distance, df['questions'])
	
	x['word_3gram_distance'] = pool.map(get_word_3gram_distance, df['questions'])
	x['char_3gram_distance'] = pool.map(get_char_3gram_distance, df['questions'])
	print('Done with SimHash Features')
	
	# FE Stage 5: 
       	
	# FE Stage 4: This features are based on question fequency
	print('Started With Magic Features')
	print('Calculating the 4 Features based on question Frequency')
	train_orig =  pd.read_csv('../input/train.csv', header=0)
	test_orig =  pd.read_csv('../input/test.csv', header=0)
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
	#train_questions.drop_duplicates(subset = ['qid1'],inplace=True)
	train_questions.drop_duplicates(subset = ['question1'],inplace=True)
	train_questions.reset_index(inplace=True,drop=True)
	questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()
	train_cp = train_orig.copy()
	test_cp = test_orig.copy()
	train_cp.drop(['qid1','qid2'],axis=1,inplace=True)
	test_cp['is_duplicate'] = -1
	test_cp.rename(columns={'test_id':'id'},inplace=True)
	comb = pd.concat([train_cp,test_cp])
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
	train_comb = comb[comb['is_duplicate'] >= 0][['id','q1_hash','q2_hash','q1_freq','q2_freq','is_duplicate']]
	test_comb = comb[comb['is_duplicate'] < 0][['id','q1_hash','q2_hash','q1_freq','q2_freq']]
	
	x_train = x[:df_train.shape[0]]
	x_train['q1_hash'] = train_comb['q1_hash']
	x_train['q2_hash'] = train_comb['q2_hash']
	x_train['q1_freq'] = train_comb['q1_freq']
	x_train['q2_freq'] = train_comb['q2_freq']
	x_test  = x[df_train.shape[0]:]
	x_test['q1_hash'] = test_comb['q1_hash']
	x_test['q2_hash'] = test_comb['q2_hash']
	x_test['q1_freq'] = test_comb['q1_freq']
	x_test['q2_freq'] = test_comb['q2_freq']
   	
	feature_names = list(x.columns.values)
	feature_names.append('q1_hash')
	feature_names.append('q2_hash')
	feature_names.append('q1_freq')
	feature_names.append('q2_freq')
	print('Done With Feature Engineering')
	print(len(feature_names))
	#create_feature_map(feature_names)
	#print("Features: {}".format(feature_names))

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
	sub.to_csv("xgb_seed{}_n{}_v4.csv".format(RS, ROUNDS), index=False)

main()
print("Done.")
