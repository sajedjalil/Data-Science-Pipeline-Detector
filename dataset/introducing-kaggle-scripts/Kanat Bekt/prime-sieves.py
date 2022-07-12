"""
Prime Sieves
============
For a more thorough explanation and information about
prime sieve algorithms.

http://sanghan.me/blog/2014/09/sieve

Sieve of Eratosthenes Complexity
--------------------------------
Time:  n*log(log(n)) 
Space: n
"""
from __future__ import print_function
    
import functools
import sys
import time
import os
import resource

from contextlib import contextmanager, redirect_stdout

import numpy as np

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
    
def list_sieve(lim):
    """Returns the set of all primes under lim."""
    p = [True] * (lim // 2)
    for i in range(3, int(lim ** 0.5)+1, 2):
        if p[i//2]:
            p[i*i//2::i] = [False] * ((lim-i*i-1) // (2*i)+1)
    primes = set([2] + [2*i+1 for i in range(1, lim//2) if p[i]])    
    memory = memory_usage_ps()
    return primes, memory
    
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

def memory_usage_ps():
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.get_memory_info()[0] / (1024 * 1024)
    return mem


def profile(implementation, n, name):
    alloc  = memory_usage_ps()
    start  = time.time()
    memory = implementation(n)
    end    = time.time()
    return name, n, memory[1]-alloc, end-start

def printer(func, name, start=3, stop=7):
    output = "{0},{1},{2:3.02f},{3:3.03f}"
    for i in range(start, stop+1):
        print(output.format(*profile(func, 10**i, name)))

def main(filename):
    """
    Problem size is increased logarithmically starting at n=10e3
    
    Note:
        Due to the availability of L2/L3 caches in modern CPU's it is
        likely that runtime speeds do not directly reflect theoretical
        limits. Cache misses and eventually the usage of swap space
        towards the upper bounds is also something to consider.
    """
    with open(filename, 'wt+') as fp:
        with redirect_stdout(fp):
            print("Implementation,Size (n),Memory (MB),Runtime (s)")
            printer(sieve,     "Python List", stop=8)
            printer(set_sieve, "Python Sets", stop=7)
            printer(np_sieve,  "Numpy Array", stop=9)
            printer(list_sieve,  "Python List2", stop=8)

if __name__ == '__main__':
    main('output.csv')