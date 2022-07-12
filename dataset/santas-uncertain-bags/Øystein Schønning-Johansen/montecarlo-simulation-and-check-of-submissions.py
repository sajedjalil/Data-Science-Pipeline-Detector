# This scriot is a checking tool for submissions. And it makes a plain MC simulation
# of the submission that can be an indication of what score to expect.

# It checks the following:
#
#  * There is exactly 1000 bags
#    (More bags is not allowed and less bags is just waste)
#  * There is at least 3 gifts in each bag.
#  * Same gift is not reused
#
# The MC-simulation will just do a plain naive simulation N times. Nothing fancy. 
# I have coded this quick'n'dirty, and not tried to optimize or bautify anything.

from __future__ import division
import numpy as np

dispatch = {
		"horse"  : lambda:max(0, np.random.normal(5,2,1)[0]),
		"ball"   : lambda:max(0, 1 + np.random.normal(1,0.3,1)[0]),
		"bike"   : lambda:max(0, np.random.normal(20,10,1)[0]),
		"train"  : lambda:max(0, np.random.normal(10,5,1)[0]),
		"coal"   : lambda:47 * np.random.beta(0.5,0.5,1)[0],
		"book"   : lambda:np.random.chisquare(2,1)[0],
		"doll"   : lambda:np.random.gamma(5,1,1)[0],
		"blocks" : lambda:np.random.triangular(5,10,20,1)[0],
		"gloves" : lambda:3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]
}

class Bag(object):
	def __init__(self, items):
		self.items = [ tuple(i.split("_")) for i in items ]
	def simulate_weight(self):		
		tot = 0.0
		for item in self.items:
			tot += dispatch[item[0]]()
		return tot
	def __len__(self):
		return len(self.items)
	def __iter__(self):
		return self.items.__iter__()
	

def check_submission( submitted_bags ):
	# should be exactly 1000 bags
	assert( len(submitted_bags) == 1000)
	
	# check minimum 3 in each
	for bag in submitted_bags:
		assert( len(bag) >= 3)
	
	# No used more than once
	fl = [item for sublist in submitted_bags for item in sublist]
	assert( len(fl) == len(set([ "%s_%s" % i for i in fl])))
	
def score_submission( submitted_bags, n=1000):
	arr = np.zeros((n))
	rej = np.zeros((n))
	for i in range(n):
		tot = 0.0
		rejected = 0
		for bag in submitted_bags:
			w = bag.simulate_weight()
			if w <= 50.0:
				tot += w
			else:
				rejected += 1
		arr[i] = tot
		rej[i] = rejected / len(submitted_bags)
	#print("mean of {} simulations: {}  std: {} ".format(n,arr.mean(), np.std(arr)))
	return arr, rej

def bags_from_file(filename):
	bags = []
	with open(filename) as f:
		head = f.readline()
		for l in f.readlines():
			bags.append( Bag( l.split() ))
	return bags

if __name__ == "__main__":
	import sys
	if len(sys.argv)==3:
		n = int(sys.argv[2])
	elif len(sys.argv)==2:
		n = 10
	else:
		print("usage: {} <filename> <n>".format(sys.argv[0]))
		exit(0)

	bags = bags_from_file(sys.argv[1])
	check_submission( bags )
	arr, rej = score_submission( bags, n=n )
	print("mean of {} simulations: {}  std: {} ".format(n,arr.mean(), np.std(arr)))
	print("Reject rate: {}".format(rej.mean()))
