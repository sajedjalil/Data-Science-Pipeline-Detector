# -*- coding: utf-8 -*-

"""
 Frequent Itemsets.

 Based on chapter 6 of Mining Massive Datasets.
 http://www.mmds.org/#book
 http://infolab.stanford.edu/~ullman/mmds/book.pdf

 Check the book for definitions of Support, Confidence, Interest and
 Association Rules. If a basket contains a certain set of items I, then it is 
 likely to contain another particular item j as well. The probability that j is 
 also in a basket containing I is called the confidence of the rule. 
 The interest of the rule is the amount by which the confidence deviates from 
 the fraction of all baskets that contain j.

 Created on Thu Dec  1 20:10:42 2016

 @author: amaia

"""

import pandas as pd
import numpy as np
from collections import defaultdict
from operator import itemgetter
from itertools import combinations


TOTAL_ROWS = 13647309
PRODUCTS = [ 'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1' ]
PRODUCT_ID = {k:k for k,v in enumerate(PRODUCTS)}
PRODUCT_NAME = {k:v for k,v in enumerate(PRODUCTS)}
LIMIT = int(.2 * TOTAL_ROWS) #1 * 1000 * 1000 #while debugging.
#LIMIT = TOTAL_ROWS


def pair_counts(train):
    """ calculate counts for each product and each pair of products    
    """
    
    # holds counts for products
    items_1 = defaultdict(int)
    
    # holds counts for pairs of products
    # pairs of products i and j are represented by the tuple (i,j), where i<j.
    items_2 = defaultdict(int)
    rows = 0        
    for basket in train.values:
        basket_items = sorted([PRODUCT_ID[i] for i, x in enumerate(basket) if x == 1])
        for p1_idx in range(0, len(basket_items)):
            # for each product in the basket increase the counter
            p1 = basket_items[p1_idx]
            items_1[p1] += 1
            # this product with every other following constitutes a pair
            for p2_idx in range(p1_idx+1, len(basket_items)):
                # increase the count of the pair
                p2 = basket_items[p2_idx]
                items_2[(p1, p2)] += 1
        rows += 1
        #if rows % 100000 == 0:
        #    print(rows)
    return items_1, items_2
    
def count_set(train, sets):
    """ count ocurrences of given list of item sets
    """
    items = defaultdict(int)    
    rows = 0
    for basket in train.values:
        basket_items = sorted([i for i, x in enumerate(basket) if x == 1])
        for s in sets:  
            if len(np.setdiff1d(s, basket_items)) == 0:            
                items[s] += 1
        rows += 1
        #if rows % 10000 == 0:
        #    print(rows)    
    return items
        

def build_sets(itemsets, n, support=1):    
    """ build itemsets of length n+1 given itemsets of length n and a support threshold
    """
    sets_to_check = []
    for s, _ in itemsets.items():        
        for testset in sorted([tuple(list(s)+[x]) for x in range(len(PRODUCTS)) if x not in s]):
            valid = True
            for subset in combinations(testset, n):
                if (subset not in itemsets) or (itemsets[subset] < support):
                    valid = False
                    break
            if valid:
                sets_to_check.append(testset)
    return sets_to_check
                

 
def confidence_x(freqs, sets):
    """ confidence(J) = suppor(J) / support(J-j)   
    """
    confidence = {}
    for s, support in sets:
        s_confidence = [np.nan] * len(s)
        for index, j in enumerate(s):
            set_without_j = np.setdiff1d(s, j)
            set_size = len(set_without_j)
            if set_size > 1:
                set_without_j = tuple(set_without_j)
            else:
                set_without_j = set_without_j[0] 
        s_confidence[index] = support / freqs[set_size][set_without_j]
        confidence[s] = s_confidence
    return confidence
          
def interest_x(freqs, sets):
    """ Interest(j) = confidence(J) - %j.
    """
    interest = {}
    for s, support in sets:
        s_confidence = [np.nan] * len(s)
        s_interest = [np.nan] * len(s)
        for index, j in enumerate(s):
             set_without_j = np.setdiff1d(s, j)
             set_size = len(set_without_j)
             if set_size > 1:
                  set_without_j = tuple(set_without_j)
             else:
                  set_without_j = set_without_j[0]                              
             s_confidence[index] = support / freqs[set_size][set_without_j]
             s_interest[index] = s_confidence[index] - freqs[1][j]/LIMIT
        interest[s] = s_interest
    return interest


if __name__ == '__main__':
    print('1.', LIMIT)
    train = pd.read_csv('../input/train_ver2.csv', usecols=PRODUCTS, nrows=LIMIT)
    
    print('2.')
    counts_single, counts_pairs = pair_counts(train)

    # note that frequent triples can only be frequent
    # if all it's subsets are also frequent. with this
    # in mind to reduce costs we restrict to the
    #
    # n most frequent pairs
    # (10 exceeded the run time limit of 20 minutes)
    n_pairs = 10
    n_most_pairs = {k:v for k,v in sorted(counts_pairs.items(), key=itemgetter(1), reverse=True)[:n_pairs]}

    print('3.')
    triples = build_sets(n_most_pairs, 2)
    count_triples = count_set(train, triples)

    print('4.')
    # formatting and sorting for display    
    #most_frequent_single_product = [(PRODUCT_NAME[x[0]],x[1]) for x in sorted(counts_single.items(), key=itemgetter(1), reverse=True)]    
    #most_frequent_pairs = [(PRODUCT_NAME[p[0]], PRODUCT_NAME[p[1]], val) for p,val in sorted(counts_pairs.items(), key=itemgetter(1), reverse=True)]    
    #most_frequent_triples = [(PRODUCT_NAME[p[0]], PRODUCT_NAME[p[1]], PRODUCT_NAME[p[2]], val) for p,val in sorted(count_triples.items(), key=itemgetter(1), reverse=True)]

    most_frequent_pairs = [(p, val) for p, val in sorted(counts_pairs.items(), key=itemgetter(1), reverse=True)]    
    most_frequent_triples = [(p, val) for p, val in sorted(count_triples.items(), key=itemgetter(1), reverse=True)]
    
    print(most_frequent_triples[:10])
    print(most_frequent_pairs)
    
    support = {
        1: counts_single,
        2: counts_pairs,
        3: count_triples
    }
    
    
    for k in [2,3]:
        threshold = 0.001 * LIMIT
        interest_threshold = 0.5
        rules = interest_x(support, {k:v for k,v in support[k].items() if v>threshold}.items())
        for s, probs in rules.items():
            for i, p in enumerate(probs):
                if p > interest_threshold:
                    rule_from = list(s)
                    rule_to = rule_from[i]
                    del rule_from[i]
                    rule_from_names = [PRODUCT_NAME[x] for x in rule_from]
                    rule_to_name = PRODUCT_NAME[rule_to]
                    
                    print("{} -> {}, i={:.2}, s={}".format(rule_from, rule_to, p, threshold))
    


"""
output

pairs
['ind_nom_pens_ult1'] -> ind_cno_fin_ult1, i=0.86, s=2729.4610000000002
['ind_cno_fin_ult1'] -> ind_nom_pens_ult1, i=0.63, s=2729.4610000000002
['ind_hip_fin_ult1'] -> ind_recibo_ult1, i=0.71, s=2729.4610000000002
['ind_nomina_ult1'] -> ind_cno_fin_ult1, i=0.86, s=2729.4610000000002
['ind_cno_fin_ult1'] -> ind_nomina_ult1, i=0.58, s=2729.4610000000002
['ind_nom_pens_ult1'] -> ind_recibo_ult1, i=0.64, s=2729.4610000000002
['ind_tjcr_fin_ult1'] -> ind_recibo_ult1, i=0.56, s=2729.4610000000002
['ind_cno_fin_ult1'] -> ind_recibo_ult1, i=0.59, s=2729.4610000000002
['ind_nomina_ult1'] -> ind_recibo_ult1, i=0.64, s=2729.4610000000002
['ind_hip_fin_ult1'] -> ind_cno_fin_ult1, i=0.54, s=2729.4610000000002
['ind_nom_pens_ult1'] -> ind_nomina_ult1, i=0.86, s=2729.4610000000002
['ind_nomina_ult1'] -> ind_nom_pens_ult1, i=0.93, s=2729.4610000000002

triples
['ind_nomina_ult1', 'ind_nom_pens_ult1'] -> ind_cno_fin_ult1, i=0.86, s=2729.4610000000002
['ind_cno_fin_ult1', 'ind_nom_pens_ult1'] -> ind_nomina_ult1, i=0.86, s=2729.4610000000002
['ind_cno_fin_ult1', 'ind_nomina_ult1'] -> ind_nom_pens_ult1, i=0.93, s=2729.4610000000002
['ind_nom_pens_ult1', 'ind_recibo_ult1'] -> ind_nomina_ult1, i=0.87, s=2729.4610000000002
['ind_nomina_ult1', 'ind_recibo_ult1'] -> ind_nom_pens_ult1, i=0.93, s=2729.4610000000002
['ind_nomina_ult1', 'ind_nom_pens_ult1'] -> ind_recibo_ult1, i=0.64, s=2729.4610000000002
['ind_nom_pens_ult1', 'ind_recibo_ult1'] -> ind_cno_fin_ult1, i=0.87, s=2729.4610000000002
['ind_cno_fin_ult1', 'ind_recibo_ult1'] -> ind_nom_pens_ult1, i=0.68, s=2729.4610000000002
['ind_cno_fin_ult1', 'ind_nom_pens_ult1'] -> ind_recibo_ult1, i=0.65, s=2729.4610000000002
['ind_nomina_ult1', 'ind_recibo_ult1'] -> ind_cno_fin_ult1, i=0.87, s=2729.4610000000002
['ind_cno_fin_ult1', 'ind_recibo_ult1'] -> ind_nomina_ult1, i=0.63, s=2729.4610000000002
['ind_cno_fin_ult1', 'ind_nomina_ult1'] -> ind_recibo_ult1, i=0.65, s=2729.4610000000002

"""


"""
output running on full dataset

{(2, 12, 23): [-0.22162147846816227, 0.1425956412539449, 0.24370512342394995],
 (4, 21, 22): [0.8593240187949802, 0.8680010711825157, 0.9406414114313671],
 (4, 21, 23): [0.8681490841658668, 0.6318421304094728, 0.661798142488087],
 (4, 22, 23): [0.8667162581907716, 0.6833749604342055, 0.6604053260049099],
 (21, 22, 23): [0.8682357444119557, 0.9406414114313671, 0.6544544533508765]}
 
{(2, 4): [-0.5534820614034004, -0.06828325567933832],
 (2, 6): [-0.11821587353931562, -0.0017541844633752225],
 (2, 7): [0.0046892605966469825, 0.000922911633603779],
 (2, 8): [-0.11341146762308218, -0.007492753372610343],
 (2, 11): [0.02298989602331525, 0.0015069817903582902],
 (2, 12): [-0.12164413584619149, -0.015355485026717908],
 (2, 13): [0.03532115112546663, 0.0009961291857547225],
 (2, 15): [-0.15177237690242662, -0.0021234690129626832],
 (2, 17): [-0.1848122287792791, -0.014812516716639267],
 (2, 18): [-0.21155956414912913, -0.01432659565001701],
 (2, 19): [0.041232847171201126, 0.001610833101765418],
 (2, 21): [-0.5079906907052807, -0.04236068591904768],
 (2, 22): [-0.5110901665831351, -0.04628275722026311],
 (2, 23): [-0.08599739496207637, -0.016782204183116475],
 (4, 7): [0.015082715801227123, 0.02406160704145749],
 (4, 8): [0.23902523047750932, 0.12800223212161055],
 (4, 11): [0.12303774518882415, 0.06537304150121227],
 (4, 12): [0.27805440955577054, 0.28450603173142397],
 (4, 13): [0.17074207301146113, 0.039031091451426636],
 (4, 14): [0.5511283939557499, 0.040118383734282746],
 (4, 17): [0.3702708826132103, 0.2405509300447795],
 (4, 18): [0.4890270946690327, 0.2684309522944262],
 (4, 19): [0.20573982899034007, 0.06515007587430374],
 (4, 21): [0.8593240187949802, 0.5808359801238852],
 (4, 22): [0.8574713449624243, 0.6294056599951844],
 (4, 23): [0.38128804515077563, 0.6031234626739737],
 (7, 8): [0.05966766680325877, 0.020029392225255446],
 (7, 11): [0.12286890226805186, 0.040922035117732944],
 (7, 12): [-0.01138357878814994, -0.007301219870889622],
 (7, 13): [0.2130466954453082, 0.03052810636375231],
 (7, 17): [0.08693636654893777, 0.035403311163966125],
 (7, 18): [0.09906895159247536, 0.03408726486036664],
 (7, 19): [0.19861743409495342, 0.039424739852604904],
 (7, 21): [0.03788113162726148, 0.016049977297606974],
 (7, 22): [0.03766310847460197, 0.017329343382457014],
 (7, 23): [0.03767097876870951, 0.03735208568380691],
 (8, 11): [0.07174394458264849, 0.07118226143862351],
 (8, 12): [0.045545219080737664, 0.08702232644377766],
 (8, 17): [0.14544839227389111, 0.17645038356931625],
 (8, 18): [0.15778003519235606, 0.16172528358167168],
 (8, 19): [0.11587866421052968, 0.06852140102160184],
 (8, 21): [0.14609683633018888, 0.18440112845461323],
 (8, 22): [0.1457434704842448, 0.1997681914411132],
 (8, 23): [0.1072310315016808, 0.3167376726962447],
 (11, 12): [0.14996696464251286, 0.28879985126116103],
 (11, 13): [0.26199904654421685, 0.11272221892994692],
 (11, 17): [0.05510289554946142, 0.06737543346570378],
 (11, 18): [0.10021579041200397, 0.10353221337411381],
 (11, 19): [0.10215947520660808, 0.060885640219975334],
 (11, 21): [0.06594105273224324, 0.08388650661804681],
 (11, 22): [0.07309634356460928, 0.1009825548176638],
 (11, 23): [0.049728343938531414, 0.14804600891505953],
 (12, 13): [0.28600752499130505, 0.06389780127016657],
 (12, 17): [0.23414393529860783, 0.1486649882887992],
 (12, 18): [0.32683176353130333, 0.17533242359824386],
 (12, 19): [0.2562735320241058, 0.0793119475571558],
 (12, 21): [0.30144005849056393, 0.19912960589118006],
 (12, 22): [0.3092992452722967, 0.2218851294791999],
 (12, 23): [0.21303603662534265, 0.3293399516388646],
 (13, 19): [0.14928280734824537, 0.20679317395486058],
 (13, 22): [0.04418644690563463, 0.14188286947071965],
 (13, 23): [0.03288617348643606, 0.22756016112537164],
 (14, 23): [0.03387140812019699, 0.7360302182374692],
 (15, 23): [0.02604675655316506, 0.3632989675208246],
 (17, 18): [0.2786300870241756, 0.23541829046560184],
 (17, 19): [0.1557417669879309, 0.07591265309786488],
 (17, 21): [0.2555695552514669, 0.2658999149278076],
 (17, 22): [0.2594025061173354, 0.29308778371511285],
 (17, 23): [0.200308320080684, 0.48771323895323576],
 (18, 19): [0.17808789435019484, 0.10273804095511342],
 (18, 21): [0.3055183550226263, 0.37621334535798673],
 (18, 22): [0.30374762903320296, 0.40618534938965606],
 (18, 23): [0.20147948387884412, 0.580609576254824],
 (19, 21): [0.07448207051225315, 0.15898338880146187],
 (19, 22): [0.07724949512702622, 0.179064760741046],
 (21, 22): [0.866182941188584, 0.9406414114313671],
 (21, 23): [0.2796552343548496, 0.6544544533508765],
 (22, 23): [0.3028873030790158, 0.6527142208937022]} 
"""     
