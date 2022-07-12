"""Makes a table of score contributions for every (kid, toy) combo and
demonstrates using it to score the sample submission.

Rev. A - Reduced from ~1 sec to ~0.2 sec evaluation using numba (thanks to
Luis Bronchal for the suggestion!)

***Important*** - Uncomment the lines 50 and 60 to save table to your
hard drive.

"""
import pandas as pd
import numpy as np
import time
from numba import jit

n_toys = 1000
n_kids = 1000000


# CREATE LOOKUP TABLE - DELTA_SCORE(KID, TOY) -------------------------------

# Load data
santa_pref = pd.read_csv('../input/gift_goodkids.csv', header=None,
                         index_col=0).values
kids_pref = pd.read_csv('../input/child_wishlist.csv', header=None,
                         index_col=0).values

# Make table with score contribution from Santa's preference
print('Creating look-up table...')
score_lookup = np.zeros((1000000, 1000), dtype='float32')
for toy in range(santa_pref.shape[0]):
    for kid_index in range(santa_pref.shape[1]):
        kid = santa_pref[toy, kid_index]
        score_lookup[kid, toy] += (1000 - kid_index) * 2. \
                                  / (2000. * n_toys) / 1000.

# Fill in pairs not in Santa's preference list
score_lookup[score_lookup == 0.] = -1. / (2000 * n_toys) / 1000.

# Make list of (kid, toy) combos in kid pref
kid_mask = np.zeros((n_kids, n_toys), dtype='bool')

# Go through kid prefs and update score table
for kid in range(kids_pref.shape[0]):
    for toy_index in range(kids_pref.shape[1]):
        toy = kids_pref[kid, toy_index]
        score_lookup[kid, toy] += (10 - toy_index) * 2. / (20. * n_kids)
        kid_mask[kid, toy] = True

# Fill in pairs not in kid's preference list
score_lookup[~kid_mask] -= 1. / (20. * n_kids)

# Save lookup table
#np.save('score_lookup.npy', score)
print('Done.')


# SCORE SAMPLE SUBMISSION --------------------------------------------------

# Load sample submission
submission = pd.read_csv('../input/sample_submission_random.csv').values

# Load score_lookup table
#score_lookup = np.load('score_lookup.npy')

# Accumulate score
@jit(nopython=True)
def score_submission(submission, score_lookup):
    score = 0
    for row in range(submission.shape[0]):
        score += score_lookup[submission[row][0], submission[row][1]]
    return score


print('Starting scoring...')
start = time.time()
score = score_submission(submission, score_lookup)
print('Done scoring.')
print('Elapsed time: %4.2f sec, Score: %10.9f' % (time.time() - start,
                                                  score))


#---------------------------------------------------------------------------