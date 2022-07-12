import numpy as np
import time


n = 10 ** 8.499
t = time.time()
sieve = np.ones(n/2, dtype=np.bool)
for i in range(3, int(n**0.5)+1, 2):
    if sieve[i >> 1]:
        sieve[i*i >> 1::i] = False
primes = (sieve.nonzero()[0][1::] << 1) + 1
time_ = time.time()
time_ -= t

print ('Found ' + str(primes.shape[0]) + ' primes in ', str(time_), ' (s)')
print ('Saving....')
t = time.time()
primes.tofile('primes.csv', sep=',\n')
done = time.time()
done -= t
print ('Saved in '+ str(done) + ' (s)')
print ('Total time ' + str(done + time_) + ' (s)')
print (primes)
