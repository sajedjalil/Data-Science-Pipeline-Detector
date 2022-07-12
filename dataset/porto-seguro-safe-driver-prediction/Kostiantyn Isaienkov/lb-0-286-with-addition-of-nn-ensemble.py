#Modification of Beletecheneke and Bojan' kernels using ensemble of NN .

import pandas as pd 
import numpy as np
 
#Download csv files from the following links. Make sure kernal's ID is the same as the one in the link.

#Author: Andy Harless
#File: xgb_submit.csv
#Link: https://www.kaggle.com/aharless/xgboost-cv-lb-284?scriptVersionId=1673404

#Author: Vladimir Demidov
#File: stacked_1.csv
#Link: https://www.kaggle.com/yekenot/simple-stacker-lb-0-284?scriptVersionId=1665392

#Author: Keui Shen Nong
#File: Froza_and_Pascal.csv
#Link: https://www.kaggle.com/kueipo/base-on-froza-pascal-single-xgb-lb-0-284?scriptVersionId=1679911

#Author: areeves87
#File: median_rank_submission.csv
#Link: https://www.kaggle.com/areeves87/aggregate-20-kernel-csvs-by-median-rank-lb-285

#Author: Kostiantyn Isaienkov & DanyaKosmin
#File: NN.csv
#Link: https://www.kaggle.com/pluchme/0-285-lb-average-of-3xgboost-lightgbm-nn

#Read csv files

stacked_1 = pd.read_csv('../input/stacked1/stacked_1.csv')
xgb_submit = pd.read_csv('../input/xgbsubmit/xgb_submit_1.csv')
Froza_and_Pascal = pd.read_csv('../input/forza-and-pascal/Froza_and_Pascal.csv')
median_rank_submission = pd.read_csv('../input/median-rank-submission/median_rank_submission.csv')
akvelon_NN_ensemble = pd.read_csv('../input/nn-ensemble/NN.csv')

# Ensemble and create submission 

sub = pd.DataFrame()
sub['id'] = stacked_1['id']
sub['target'] = np.exp(np.mean(
	[	
	stacked_1['target'].apply(lambda x: np.log(x)),\
	xgb_submit['target'].apply(lambda x: np.log(x)),\
	Froza_and_Pascal['target'].apply(lambda x: np.log(x)),\
	median_rank_submission['target'].apply(lambda x: np.log(x)),\
    akvelon_NN_ensemble['target'].apply(lambda x: np.log(x))\
	], axis =0))
	
sub.to_csv('sub.csv', index = False, float_format='%.6f') 
