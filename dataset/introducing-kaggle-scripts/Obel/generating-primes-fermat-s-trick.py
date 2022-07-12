#Improvements
#1 - Looks only at odd numbers
#2 - Square-root trick (don't divide by primes larger than sqrt(n))

import csv
import time
import math

n = 3
primes = [2] #Kind of cheating by starting with 2 already...

start = time.time()

with open('primes.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow([2]) #Yeah yeah, I know
    
    while time.time() < start + 10:
        is_prime = True
        for value in primes:
            if n % value == 0:
                is_prime = False
                break
            if value > (math.sqrt(n)): #the square-root trick
                break
        if is_prime:
            primes.append(n)
            csvwriter.writerow([n])
        n = n+2

print (len(primes))