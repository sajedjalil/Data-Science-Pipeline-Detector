import numpy as np
from math import sqrt, ceil
from time import time


#Using Sieve of Sundaram

start = time()

N = 1e7
SQR = int(ceil(sqrt(N)))

prime = np.ones(N, dtype=np.bool)

i_limit = int(ceil((sqrt(2 * N + 1) - 1) / 2))
for i in range(1, i_limit):
    j_limit = int(ceil((N - i) / (2 * i + 1)))
    for j in range(i, j_limit):
        val = i + j + 2 * i * j
        prime[val] = False
        
print (time() - start)
primes = 2 * np.nonzero(prime)[0][1::].astype(np.int32) + 1
with open("prime.csv", 'w') as file:
    file.write('2\n')   #Because '2' is not included in 'primes' list
    for item in primes:
        file.write(str(item))
        file.write('\n')
        
print (time() - start)
print (len(primes))
