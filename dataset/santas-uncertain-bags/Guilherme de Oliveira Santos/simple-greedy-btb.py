import pandas as pd
import numpy as np 
import sys

gifts = pd.read_csv("../input/gifts.csv")

weigths = {}
weigths['horse'] = 5
weigths['ball'] = 2
weigths['bike'] = 20
weigths['train'] = 10
weigths['coal'] = 47
weigths['book'] = 2
weigths['doll'] = 5
weigths['blocks'] = 12
weigths['gloves'] = 4

itens = []

for ix, row in gifts.iterrows():
	key = row['GiftId'].split('_')[0]
	itens.append((row['GiftId'], weigths[key]))

itens = [(x, y) for (x,y) in sorted(itens, key=lambda x: x[1], reverse=True)]

bags = []
total = 0

while len(bags) < 1000 and len(itens) > 0:
	bag_weigth = 0
	bag = []
	for ix, item in enumerate(itens):		
		if (bag_weigth+item[1] <= 50):
			if (len(bag) == 0 and item[1] > 20):
				continue
			bag.append(item[0])
			bag_weigth += item[1]
			del itens[ix]

	total += bag_weigth
	bags.append(bag)

print("Estimated total: ", total)

with open("submission.csv", 'w') as outfile:
    outfile.write('Gifts\n')
    for bag in bags:
    	if (len(bag) >= 3):
	    	presents = " ".join(bag)
	    	outfile.write('%s\n' % presents)