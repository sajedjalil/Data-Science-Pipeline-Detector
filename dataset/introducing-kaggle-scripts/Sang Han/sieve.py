"""
Prime Sieves
============
For a more thorough explanation and information about
prime sieve algorithms.

http://sanghan.me/blog/2014/09/sieve

Sieve of Eratosthenes Complexity
================================

Time:  O(n*log(log(n)))
Space: O(n)

Results
=======
Results show a much different complexity than what is expected.
It's likely this is due to the fact that modern computational
architectures utilize L2/L3 caches as a form of high performance memory

This may explain for what looks like a large under utilization
leading into the huge gain towards the end.

Conclusion
==========
* The utilization of cache causes optimal performance at low n
* Numpy arrays are an optimal data structure, 
  they are both space and time efficient since we already
  know the size of the array beforehand.
* Improving your memory profile means improving your runtime efficiency!
    
"""

from __future__ import print_function
    
import functools
import sys
import time
import os
import psutil
import operator
import resource

from contextlib import contextmanager, redirect_stdout
from collections import namedtuple

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import kendalltau

class Lambda:
    """
    Returns a callable λ class instance which impements
    a prime sieve algorithm from 2 to n.
    """
    \
                def __call__(self, n):
                    m = memory_usage_ps()
                    p = (lambda very_math:\
                                 map(lambda\
                                 __lololol_:\
                                      filter(
                               (( ( ((None)) ) )),
                          (map(lambda __suchwoow:\
                    map(lambda  __because___yolo__:\
          __lololol_.__setitem__((      (__because___yolo__))  ,                (0)),
    range(2*(__suchwoow),               ((very_math)),     __suchwoow               ) ),
range(2,very_math)),                    (__lololol_))[1])[1:],[range(very_math)])[0])(n)
                    return p, m


def sieve(n):
    """
    Straight forward implementation of The Sieve of Eratosthenes.
    Uses a just a simple list as an index.
    """
    primes = 2*[False] + (n-1)*[True]
    for i in range(2, int(n**0.5+1)):
        for j in range(i*i, n+1, i):
            primes[j] = False

    p = [prime for prime, checked in enumerate(primes) if checked]
    memory = memory_usage_ps()

    return p, memory
    
def np_sieve(n):
    """
    Sieve using numpy boolean array with zero copy.
    """
    primes = np.ones(n+1, dtype=np.bool)
    for i in np.arange(3, n**0.5+1, 2, dtype=np.uint32):
        if primes[i]:
            primes[i*i::i] = False

    p = primes.nonzero()[0][2:]
    memory = memory_usage_ps()
    
    return p, memory
    
def set_sieve(n):
    """
    Sets are mutable, unordered collections which are useful
    for quick membership testing and math operations.
    """
    factors = set()
    for i in range(2,int(n**0.5+1)):
        if i not in factors:
            factors |= set(range(i*i, n+1, i))

    primes = set(range(2,n+1)) - factors
    memory = memory_usage_ps()

    return primes, memory

lol_sieve = Lambda()

def memory_usage_ps():
    """
    Parses Unix ps and returns memory allocation in MB.
    """
    process = psutil.Process(os.getpid())
    mem     = process.get_memory_info()[0]
    return mem / (1024 * 1024)

def profile(implementation, n, name):
    alloc  = memory_usage_ps()
    start  = time.time()
    memory = implementation(n)
    end    = time.time()
    return name, n, memory[1]-alloc, end-start

def output(func, name, start=3, stop=7):
    """
    Helper function for formatted printint.
    """
    o = "{0},{1},{2:3.02f},{3:3.03f}"

    for i in range(start, stop+1):
        print(o.format(*profile(func, 8**i, name)))
        print(o.format(*profile(func, 9**i, name)))
        print(o.format(*profile(func, 10**i, name)))

def write_data(filename, *header, **algorithms):
    """
    Problem size is increased logarithmically starting at n=10e3
    """
    with open(filename, 'wt+') as fp:

        with redirect_stdout(fp):
            print(*header, sep=',')

            for name, algorithm in algorithms.items():
                output(algorithm.implementation, name, stop=algorithm.top)

    return pd.read_csv(filename)

def draw_plot(implementation, filename):
    """
    A Kendal Tau significance test show coupling of memory and runtime.
    """
    fig = sns.jointplot(x="Runtime (s)",
                  y="Memory (MB)",
                  data=implementation,
                  kind='reg',
                  size=5,
                  stat_func=kendalltau
    )
    
    fig.plot_joint(sns.regplot)
    fig.plot_marginals(sns.kdeplot, shade=True)
    
    fig.savefig(filename + '.png')
    
    return fig

def main(filename):
    """
    Run everything and do the magic
    """
    header     = "Implementation","Size (n)","Memory (MB)","Runtime (s)"
    Algorithm  = namedtuple('Algorithm', ['implementation', 'top'])
    get_subset = lambda df, x: df[df.Implementation == x]

    sieve_dict = {
        'Python List':  Algorithm(sieve,     8),
        'Python Sets':  Algorithm(set_sieve, 7),
        'Numpy Array':  Algorithm(np_sieve,  9),
    }

    df = write_data(filename, *header, **sieve_dict)

    # Make plot for every sieve
    for i in df.Implementation.unique():
        draw_plot(get_subset(df, i), i)

if __name__ == '__main__':
    sys.exit(main('output.csv'))