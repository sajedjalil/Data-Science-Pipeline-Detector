
primes = [2,3,5]
for num in range(2, 197768):
    for i in range(2, int(num / 2)):
        if num % i == 0:
            break
        elif i == int(num / 2)-1:
            if num in primes:
                continue
            else:
                primes.append(num)
print(primes)