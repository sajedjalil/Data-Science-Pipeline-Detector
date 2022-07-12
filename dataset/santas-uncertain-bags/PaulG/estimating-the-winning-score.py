#
# A super-simple script that estimates what the winning score will be.
#
# It assumes that the winning entry will come from shuffling an "optimal"
# packing pattern that has a normal distribution of scores.
#
import numpy as np


num_submissions = 18000  # (400/2 teams shuffling) x (3 submissions per day) x (30 days left)    
score_avg       = 35540  # Average score of the optimal packing solution
score_sdev      = 315    # Standard deviation of the optimal packing solution score


# Simulate running the competition 10,000 times.
np.random.seed(0)
num_trials = 10000
winning_scores = np.zeros(num_trials)
for i in range(num_trials):
    winning_scores[i] = np.amax( np.random.normal(score_avg, score_sdev, num_submissions) )

print( "Predicted winning score: %0.0f +/- %0.0f" %(np.mean(winning_scores), np.std(winning_scores)) )
