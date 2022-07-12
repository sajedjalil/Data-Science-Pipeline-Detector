import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
# https://www.kaggle.com/jpopham91/caterpillar-tube-pricing/rmlse-vectorized
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

targets = pd.read_csv('../input/train.csv', usecols=['Demanda_uni_equil'], nrows=10000000)

best = -1
bestscore = 9
for i in np.arange(3.915,3.917,0.0001):
    targets['pred'] = i
    print('Evaluating ' + str(i).ljust(2), end='', flush=True)
    score = rmsle(targets['Demanda_uni_equil'].values, targets['pred'].values)
    print(' - Score : ' + str(round(score, 7)))
    if score < bestscore:
        bestscore = score
        best = i
        

print('\nBest value found: ' + str(best))
print('           Score: ' + str(bestscore))

print('\nGenerating submission ...')
sub = pd.read_csv('../input/sample_submission.csv')
sub['Demanda_uni_equil'] = best
sub.to_csv('best_naive.csv', index=False)