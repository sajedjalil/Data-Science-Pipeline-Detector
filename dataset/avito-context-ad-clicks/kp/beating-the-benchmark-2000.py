# ad click prediction : a view from the trenches
# __author__ : Abhishek Thakur
# __credits__ : tinrtgu

from math import sqrt, exp, log
from csv import DictReader
import pandas as pd
import numpy as np

class ftrl(object):
	def __init__(self, alpha, beta, l1, l2, bits):
		self.z = [0.] * bits
		self.n = [0.] * bits
		self.alpha = alpha
		self.beta = beta
		self.l1 = l1
		self.l2 = l2
		self.w = {}
		self.X = []
		self.y = 0.
		self.bits = bits
		self.Prediction = 0.
	
	def sgn(self, x):
		if x < 0:
			return -1  
		else:
			return 1

	def fit(self,line):
		try:
			self.ID = line['ID']
			del line['ID']
		except:
			pass

		try:
			self.y = float(line['IsClick'])
			del line['IsClick']
		except:
			pass

		del line['HistCTR']
		self.X = [0.] * len(line)
		for i, key in enumerate(line):
			val = line[key]
			self.X[i] = (abs(hash(key + '_' + val)) % self.bits)
		self.X = [0] + self.X

	def logloss(self):
		act = self.y
		pred = self.Prediction
		predicted = max(min(pred, 1. - 10e-15), 10e-15)
		return -log(predicted) if act == 1. else -log(1. - predicted)

	def predict(self):
		W_dot_x = 0.
		w = {}
		for i in self.X:
			if abs(self.z[i]) <= self.l1:
				w[i] = 0.
			else:
				w[i] = (self.sgn(self.z[i]) * self.l1 - self.z[i]) / (((self.beta + sqrt(self.n[i]))/self.alpha) + self.l2)
			W_dot_x += w[i]
		self.w = w
		self.Prediction = 1. / (1. + exp(-max(min(W_dot_x, 35.), -35.)))
		return self.Prediction

	def update(self, prediction): 
		for i in self.X:
			g = (prediction - self.y) #* i
			sigma = (1./self.alpha) * (sqrt(self.n[i] + g*g) - sqrt(self.n[i]))
			self.z[i] += g - sigma*self.w[i]
			self.n[i] += g*g

if __name__ == '__main__':

	"""
	SearchID	AdID	Position	ObjectType	HistCTR	IsClick
	"""
	train = '../input/trainSearchStream.tsv'
	clf = ftrl(alpha = 0.1, 
			   beta = 1., 
			   l1 = 0.1,
			   l2 = 1.0, 
			   bits = 20)

	loss = 0.
	count = 0
	for t, line in enumerate(DictReader(open(train), delimiter='\t')):
		clf.fit(line)
		pred = clf.predict()
		loss += clf.logloss()
		clf.update(pred)
		count += 1
		if count%1000 == 0: 
			print ("(seen, loss) : ", (count, loss * 1./count))
		if count == 10: 
			break

	test = '../input/testSearchStream.tsv'
	with open('temp.csv', 'w') as output:
		for t, line in enumerate(DictReader(open(test), delimiter='\t')):
			clf.fit(line)
			output.write('%s\n' % str(clf.predict()))

	sample = pd.read_csv('../input/sampleSubmission.csv')
	preds = np.array(pd.read_csv('temp.csv', header = None))
	index = sample.ID.values - 1

	sample['IsClick'] = preds[index]
	sample.to_csv('submission.csv', index=False)
	