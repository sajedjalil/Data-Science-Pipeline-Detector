import numpy as np 
import pandas as pd

LIMIT_WEIGHT = 30
LIMIT_SIZE = 3

def sampleweight(gift):
    if gift == "horse":
        return max(0, np.random.normal(5,2,1)[0])
    if gift == "ball":
        return max(0, 1 + np.random.normal(1,0.3,1)[0])
    if gift == "bike":
        return max(0, np.random.normal(20,10,1)[0])
    if gift == "train":
        return max(0, np.random.normal(10,5,1)[0])
    if gift == "coal":
        return 47 * np.random.beta(0.5,0.5,1)[0]
    if gift == "book":
        return np.random.chisquare(2,1)[0]
    if gift == "doll":
        return np.random.gamma(5,1,1)[0]
    if gift == "blocks":
        return np.random.triangular(5,10,20,1)[0]
    if gift == "gloves":
        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]

gifts = pd.read_csv('../input/gifts.csv')['GiftId'].values
giftscount = gifts.shape[0]
print("We have {} gifts".format(giftscount))

# TODO - better first sample weight, then sort descending by weights

bags = [''] * 1000
bag_weights = [LIMIT_WEIGHT] * 1000
bag_sizes = [LIMIT_SIZE] * 1000
used = []
for gift in list(gifts):
    weight = sampleweight(gift.split('_')[0])
    for i in range(1000):
        if bag_weights[i] > weight or sum(bag_sizes) >= (giftscount - i):
            bags[i] += ' %s' % gift
            bag_weights[i] -= weight
            bag_sizes[i] -= 1
            break



estimate = LIMIT_WEIGHT * 1000 - round(sum(bag_weights), 0)
print('We go for total weight to be: %s' % str(estimate))

with open('submission_%s.csv' % str(estimate), 'w') as outfile:
    outfile.write('Gifts\n')
    for bag in bags:
        outfile.write('%s\n' % bag.strip())

print('Done.')