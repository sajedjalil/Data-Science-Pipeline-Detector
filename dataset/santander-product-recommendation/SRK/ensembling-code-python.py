"""
Python version of the R code by BreakfastPirate for ensembling two (or three) submission files from the forum post
https://www.kaggle.com/c/santander-product-recommendation/forums/t/25815/creating-ensemble-from-2-submissions
__author__ == SRK
"""

import csv

## Weights of the individual subs ##
sub1_weight = 1.2
sub2_weight = 0.8
#sub3_weight = 0.8 

place_weights = {
0 : 7.,
1 : 6.,
2 : 5.,
3 : 4.,
4 : 3.,
5 : 2.,
6 : 1.
}

## input files ##
sub1 = csv.DictReader(open("../Submissions/sub1.csv"))
sub2 = csv.DictReader(open("../Submissions/sub2.csv"))
#sub3 = csv.DictReader(open("../Submissions/sub3.csv"))

## output file ##
out = open("./sub_ens.csv", "w")
writer = csv.writer(out)
writer.writerow(['ncodpers','added_products'])

for row1 in sub1:
	row2 = next(sub2)
	#row3 = next(sub3)
	assert row1['ncodpers'] == row2['ncodpers'] #== row3['ncodpers']
	product_weight = {}
	for ind, prod in enumerate(row1['added_products'].split()):
		product_weight[prod] = product_weight.get(prod,0) + (place_weights[ind]*sub1_weight)
	for ind, prod in enumerate(row2['added_products'].split()):
		product_weight[prod] = product_weight.get(prod,0) + (place_weights[ind]*sub2_weight)
	#for ind,prod in enumerate(row3['added_products'].split()):
	#	product_weight[prod] = product_weight.get(prod,0) + (place_weights[ind]*sub3_weight)
	top7_prod = sorted(product_weight, key=product_weight.get, reverse=True)[:7]
	writer.writerow([row1['ncodpers'], " ".join(top7_prod)])

out.close()