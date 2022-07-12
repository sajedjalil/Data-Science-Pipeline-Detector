import csv
import time

n = 2
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
        if is_prime:
            primes.append(n)
            csvwriter.writerow([n])
        n = n+1

print (len(primes))