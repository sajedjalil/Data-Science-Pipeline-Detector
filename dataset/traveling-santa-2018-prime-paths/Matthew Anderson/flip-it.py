# FLIP IT.
# BY MATTHEW ANDERSON

# Due to most TSP solvers not being built for a prime path problem
# such as Concorde or LKH, order of primes is not accounted for in
# the solution.  As a result, if we flip the tour, the order of
# primes may actually be better and improve results.  I was
# shcoked that there is no kernel for this, especially given how
# little code it requires.
# See below:


# Read in your path data frame as an array:
# We'll use the current best public kernel:
import pandas as pd
tour = pd.read_csv('../input/lkh-solver/submission.csv')['Path'].tolist()
cities = pd.read_csv('../input/traveling-santa-2018-prime-paths/cities.csv')

# Define a function for length of a path:
import numpy as np
from sympy import primerange
primes = list(primerange(0, len(cities)))
def score_tour(tour):
    # length of any given tour with primes calculation
    df = cities.reindex(tour + [0]).reset_index()
    df['prime'] = df.CityId.isin(primes).astype(int)
    df['dist'] = np.hypot(df.X - df.X.shift(-1), df.Y - df.Y.shift(-1))
    df['penalty'] = df['dist'][9::10] * (1 - df['prime'][9::10]) * 0.1
    return df.dist.sum() + df.penalty.sum()

# Let's take a look at our tour
print("Tour path (0-5):",tour[0:5])
# And the flipped tour looks like:
tourflip = tour[::-1]
print("Flipped tour path (0-5):", tourflip[0:5])
# The scores of our tours are:
print("Score of original tour:", score_tour(tour))
print("Score of flipped tour:", score_tour(tourflip))

# If the flipped tour is quicker, change our tour:
if score_tour(tourflip) < score_tour(tour):
    print("The total improvement was:", abs(score_tour(tourflip) - score_tour(tour)))
    tour = tourflip 
    print("The better of the original/flipped tour is:", tour[0:5])

pd.DataFrame({'Path': list(tour)}).to_csv('submission.csv', index=False)
    
    
# Final note:
# This can be performed in batches during the solve or can be
# performed at the end of the entire thing.  Either way, you'll
# improve or stay the same, but never get worse.