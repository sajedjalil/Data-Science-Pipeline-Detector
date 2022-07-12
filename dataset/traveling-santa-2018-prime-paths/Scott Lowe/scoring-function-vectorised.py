#!/usr/bin/env python

import math
import os.path
import pickle
import sys

import numpy as np
import pandas as pd


# Path to problem definition file
CITIES_FILE = '../input/cities.csv'
# Cache file(s)
CITIES_PRIME_CSV_FILE = 'cities_plus_prime.csv'
CITIES_PRIME_PKL_FILE = 'cities_plus_prime.pkl'


def sieve_of_eratosthenes(n):
    """
    Identify prime numbers
    
    Parameters
    ----------
    n : int
        Largest integer of interest.

    Returns
    -------
    is_prime : list
        List of length `(n+1)`. `is_prime[i]` is True if
        `i` is prime, False otherwise.
        Note that `is_prime` is indexed from 0;
        `is_prime[0]` and `is_prime[1]` are both False, as
        neither 0 nor 1 are prime.
    """
    # Edge case handling
    if n < 2:
        return [False for i in range(n+1)]
    # Create a boolean array "is_prime[0..n]" and initialize
    # all entries in it as True.
    is_prime = [True for i in range(n+1)]
    # Manually set 0 and 1 to not be prime
    is_prime[0] = False
    is_prime[1] = False
    # Now starting from 2, we do the sieve
    p = 2
    while (p * p <= n):
        # If this number is prime, we still need to remove its
        # multiples
        # If it is not prime, we already removed its multiples
        # so we can move on straight away.
        if is_prime[p]: 
            # Update all multiples of p
            for i in range(2*p, n+1, p): 
                is_prime[i] = False
        p += 1
    return is_prime


if os.path.isfile(CITIES_PRIME_PKL_FILE):
    # Check if we have our cache already, load from that
    cities_xy, cities_p = pickle.load(open(CITIES_PRIME_PKL_FILE, 'rb'))
elif os.path.isfile(CITIES_FILE):
    # No cache, so identify the primes and save to cache
    cities_df = pd.read_csv(CITIES_FILE)
    # Find the primes
    cities_p = sieve_of_eratosthenes(len(cities_df) - 1)
    cities_df['is_prime'] = cities_p
    # Save to a CSV file, in case that is convenient for you.
    # In the original implementation, we used this for our cache instead
    # of pickling, but it might be a useful artefact so it is still
    # generated.
    cities_df.to_csv(CITIES_PRIME_CSV_FILE, index=None)
    
    # Reformat the data so its how we need it
    cities_xy = np.stack((cities_df['X'].values, cities_df['Y'].values), axis=1)
    cities_p = np.array(cities_p)
    # Save to pickle cache
    pickle.dump((cities_xy, cities_p), open(CITIES_PRIME_PKL_FILE, 'wb'))
else:
    raise EnvironmentError('Missing file: {}'.format(CITIES_FILE))


def score_path(path):
    """
    Scores the Traveling Santa Distance of a path.
    
    Criterion is Euclidean distance, but every 10th step is
    10% longer if it doesn't originate at a prime city number.
    
    Parameters
    ----------
    path : array_like
        Can be implicit or explicit cycle.

    Returns
    -------
    float
        Total distance along path, under Traveling Santa Metric.
    """
    # Remove trailing home city, if present
    if path[-1] == path[0]:
        path = path[:-1]
    # Permute matrix to path ordering
    xy = cities_xy[path, :]
    distances = np.sqrt(
        np.sum(
            np.square(xy - np.roll(xy, -1, axis=0)),
            axis=1
        )
    )
    # Extract out prime status for every 10th starting city
    p = cities_p[path[9::10]]
    p_multiplier = np.where(p, 1.0, 1.1)
    # Scale up relevant routes
    distances[9::10] *= p_multiplier
    return distances.sum()


def score_csv(fname):
    """
    Scores the Traveling Santa Distance of a path from CSV file.
    
    Parameters
    ----------
    fname : str
        Location of CSV file. The CSV file must contain a field
        labelled 'Path', containing the path.

    Returns
    -------
    float
        Total distance along path, under Traveling Santa Metric.
    """
    path_df = pd.read_csv(fname)
    path = path_df['Path'].values
    return score_path(path)


if __name__ == '__main__':
    # --------------------------------
    # This is just for the Kaggle kernel demo.
    # Remove this block after downloading.
    print(score_csv('../input/sample_submission.csv'))
    sys.exit()
    # --------------------------------
    # Run as a script, with path to CSV file as only argument
    print(score_csv(sys.argv[1]))
