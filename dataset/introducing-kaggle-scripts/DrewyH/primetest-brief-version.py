from numpy import ones, arange, bool, uint32

# Nominate the upper limit for the prime search
hl = 10**7

# Create an array of odd numbers (ignore 2)
nos = arange(3, hl, 2, dtype=uint32) #from blog for uint32

# Create an array of booleans determining which are prime
ps = ones(len(nos), dtype=bool)

# Iterate over all the numbers up to the high limit
for i,n in enumerate(nos):
    # If the value is not prime then skip it
    if not ps[i]:
        continue
    
    # If the value is prime set all multiples of that number to false (not prime)
    ps[i+n::n] = False

# Print all the numbers which are prime
print(len(nos[ps]))