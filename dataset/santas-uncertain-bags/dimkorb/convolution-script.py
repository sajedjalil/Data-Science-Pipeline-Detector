import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, beta, chi2, gamma, triang
from collections import Counter

# Make probability distributions for gifts
n = 1000
x = np.linspace(0, 100, n)
dx = 100/(n-1)
n_lt50 = sum(x<=50)
print(n_lt50)
gift_dist = dict()       

def addToDist(gift, prob): 
    gift_dist[gift] = prob/sum(prob)
    
# truncated normal
def tnorm(mu, sig):
    p = norm.pdf(x, mu, sig)*dx
    p[0] = norm.cdf(0, mu, sig)
    return p
addToDist('horse', tnorm(5, 2) )
addToDist('ball', tnorm(2, 0.3) )
addToDist('bike', tnorm(20, 10) )
addToDist('train', tnorm(10, 5) )

# custom glove distribution
p = pd.Series([0]*len(x))
p[(x>=0) & (x<=1)] =  0.7
p[(x>=3) & (x<=4)] =  0.3
addToDist('gloves', np.array(p) )

# scipy distributions
p = beta.pdf(x, 0.5, 0.5, 0, 47)
p[0] = p[1]
addToDist('coal', p )
addToDist('book', chi2.pdf(x, 2))
addToDist('doll', gamma.pdf(x, 5) )
addToDist('blocks', triang.pdf(x, 1/3, 5, 15))

# functions of a random variable
def expectedUtility(p):
    return np.dot(x[:n_lt50], p[:n_lt50])
    
def overloadProb(p):
    return 1-sum(p[:n_lt50])   
    
    # order gifts in order of increasing weight
gift_weights = {k: np.dot(v, x) for k, v in gift_dist.items()}
gift_types =  sorted(gift_weights, key=gift_weights.get)

# get children gifts that are heavier than gift
gift_order = {gift_types[i]:i for i in range(9)}
def getHeavier(gift):
    return gift_types[gift_order[gift]:]

# a node in the candidate bag tree 
class Node(object):
    def __init__(self, bag, prob, utility):
        self.bag = bag.copy()
        self.prob = prob
        self.utility = utility
        
        # get all candidate with size < bag_capacity
candidate_list = []
bag_capacity= 8
for g, p in gift_dist.items():
    print('processing : ' + g)
    open_nodes = [Node([g], p, expectedUtility(p))]
    while open_nodes:
        node = open_nodes.pop()
        # consider adding heavier gifts to the bag 
        for gift in getHeavier(node.bag[-1]):
            b = node.bag + [gift]
            p = np.convolve(node.prob, gift_dist[gift])[:n]        
            u = expectedUtility(p)
            # check if utility increased
            if u>node.utility:
                # add node to candidate list
                candidate_list.append({**Counter(b), 'utility': u, 'p_overload': overloadProb(p)})    
                
                # if bag is not a leaf add it to open_nodes
                if len(b)<bag_capacity:
                    open_nodes.append(Node(b, p, u))
                    
                    #postprocess and save data
df = pd.DataFrame(candidate_list)
df.fillna(0, inplace=True)
print(df)
# add column for bag_size
df['bag_size'] = df[gift_types].sum(axis=1)
df = df[df['bag_size']>=3]
df = df.reset_index(drop=True)
print(bag_capacity)
# reorder columns and save
df = df.reindex_axis(gift_types + ['bag_size', 'p_overload', 'utility'], axis=1)
df.to_csv('bag_utility' + str(bag_capacity) + '.csv', index =False)
df.to_csv('sumission1.csv', index =False)

bestbags=df.loc[(df['utility']>=10)].sort(['utility', 'p_overload'])
print (bestbags.shape)

numhorse = 1000
numball = 1100
numbike = 500
numtrain = 1000
numbook = 1200
numdoll = 1000
numblocks = 1000
numgloves = 200
numcoal = 166

outfile = open( 'submission_conv1.csv', 'w' )
outfile.write( 'Gifts\n' )
bags = 0
for bag in range(0,500):
    if ((numgloves >0) & (numbook >0) &  (numball >0) &  (numhorse >0) &  (numbike>0) &   (numblocks >0) &  (numdoll >0) &  (numtrain >0) & (numcoal >0) ):
        print(bag)
        s = ''
        if (int(bestbags.iloc[bag][0])>0):
            for i in range(int(bestbags.iloc[bag][0])):
                s = s + 'gloves_%d' % (numgloves-1) + ' '
                numgloves -= 1
    
        if (int(bestbags.iloc[bag][1])>0):
            for i in range(int(bestbags.iloc[bag][1])):
                s = s + 'book_%d' % (numbook-1) + ' '
                numbook -= 1
    
        if (int(bestbags.iloc[bag][2])>0):
            for i in range(int(bestbags.iloc[bag][2])):
                s = s + 'ball_%d' % (numball-1) + ' '
                numball -= 1
    
        if (int(bestbags.iloc[bag][0])>0):
            for i in range(int(bestbags.iloc[bag][3])):

                s = s + 'doll_%d' % (numdoll-1) + ' '
                numdoll-= 1
    
        if (int(bestbags.iloc[bag][4])>0):
            for i in range(int(bestbags.iloc[bag][4])):

                s = s + 'horse_%d' % (numhorse-1) + ' '
                numhorse -= 1
    
        if (int(bestbags.iloc[bag][5])>0):
            for i in range(int(bestbags.iloc[bag][5])):

                s = s + 'train_%d' % (numtrain-1) + ' '
                numtrain -= 1
    
        if (int(bestbags.iloc[bag][6])>0):
            for i in range(int(bestbags.iloc[bag][6])):
                s = s + 'blocks_%d' % (numblocks-1) + ' '
                numblocks-= 1

    
        if (int(bestbags.iloc[bag][7])>0):
            s = ''
            for i in range(int(bestbags.iloc[bag][7])):

                s = s + 'bike_%d' % (numbike-1) + ' '
                numbike -= 1
            
        if (int(bestbags.iloc[bag][8])>0):
            s = ''
            for i in range(int(bestbags.iloc[bag][8])):

                s = s + 'coal_%d' % (numcoal-1) + ' '
                numcoal -= 1
        print(s)
    

                
    

        outfile.write( s+'\n' )

        