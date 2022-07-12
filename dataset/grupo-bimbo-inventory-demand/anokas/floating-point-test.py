import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
# https://www.kaggle.com/marknagelberg/caterpillar-tube-pricing/rmsle-function
def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
	
targets = pd.read_csv('../input/train.csv', usecols=['Demanda_uni_equil'], nrows=10000000)

best = -1
bestscore = 9
for i in np.arange(3,5,0.1).tolist():
    targets['pred'] = i
    print('Evaluating ' + str(i).ljust(2), end='', flush=True)
    score = rmsle(targets['Demanda_uni_equil'].tolist(), targets['pred'].tolist())
    print(' - Score : ' + str(round(score, 5)))
    if score < bestscore:
        bestscore = score
        best = i
        

print('\nBest value found: ' + str(best))
print('           Score: ' + str(bestscore))

print('\nGenerating submission ...')
sub = pd.read_csv('../input/sample_submission.csv')
sub['Demanda_uni_equil'] = best
sub.to_csv('best_naive.csv', index=False)