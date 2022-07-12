from math import sqrt

prime=[]

def is_prime(n):
    sq=sqrt(n)
    for p in prime:
        if (n%p==0): return 0
        if (p>sq): return 1
    return 1

with open ("primes.csv", 'w') as f:
    
    for i in range(6,1800000,6):

      if (is_prime(i-1)):
        prime.append(i-1)

      if (is_prime(i+1)):
        prime.append(i+1)
        
        
    f.write("%d,\n" % 2)
    f.write("%d,\n" % 3)
        
    for p in prime:
        f.write("%d,\n" % p)
