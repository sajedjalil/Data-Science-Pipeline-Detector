import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def primesfrom2to(n):
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = np.ones(n//3 + (n%6==2), dtype=np.bool)
    for i in range(1,int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[       k*k//3     ::2*k] = False
            sieve[k*(k-2*(i&1)+4)//3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0][1:]+1)|1)]

def score(path, coords, primes):
    c = coords[path]
    d = np.linalg.norm(c[1:] - c[:-1], axis=1)
    i = np.where(np.isin(path[9:-1:10], primes, True, True))[0] * 10 + 9
    d[i] *= 1.1
    return np.sum(d)

coords = pd.read_csv('../input/cities.csv', index_col=['CityId']).values
path = pd.read_csv('../input/sample_submission.csv').values[:,0]
primes = primesfrom2to(coords.shape[0])

print("Score")
print(score(path, coords, primes))
