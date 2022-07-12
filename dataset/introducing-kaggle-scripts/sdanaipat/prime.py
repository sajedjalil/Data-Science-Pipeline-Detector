import numpy as np


def prime(n):
    flags = np.ones(n+1, dtype=np.bool)
    p = 2
    while p <= n:
        yield p
        flags[p*p::p] = False
        p += 1
        while p <= n and not flags[p]:
            p += 1
            

with open ("primes.csv", 'w') as f:
    for x in prime(20000000):
        f.write("%d\n" % x)