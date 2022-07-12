import zlib
import bz2
import random
random.seed("The Complete Works of William Shakespeare")

# Tries to find a character to append that causes a better compression ratio
# Ensemble of zlib (DEFLATE) and bz2 (Burrows–Wheeler).

base = "kaggle"

genes = "abcdefghijklmnopqrstuvwxyz"

for population_id in range(1000):
    candidates = []
    for gene in genes:
        candidate = base + gene
        
        candidates.append(( len(zlib.compress(candidate.encode("ascii")))+
                            len(bz2.compress(candidate.encode("ascii"))),
                            candidate))
    base = sorted(candidates)[0][1]

nice_output = ""
for i, char in enumerate(base):
    nice_output += char
    if i > 0 and i % 100 == 0:
        nice_output += "\n"
print(nice_output)
print("Length: %s"%(sorted(candidates)[0][0]))

# Tries to find a character to append that causes a worse compression ratio (random)
# Ensemble of zlib (DEFLATE) and bz2 (Burrows–Wheeler).

base = "kaggle"

genes = list("abcdefghijklmnopqrstuvwxyz")

for population_id in range(1000):
    candidates = []
    for gene in genes:
        candidate = base + gene
        tie_breaker = random.randint(0,1000) # to randomly break ties when sorting
        candidates.append(( len(zlib.compress(candidate.encode("ascii")))+
                            len(bz2.compress(candidate.encode("ascii"))),
                            tie_breaker,
                            candidate))
    base = sorted(candidates, reverse=True)[0][2]
print()    
nice_output = ""
for i, char in enumerate(base):
    nice_output += char
    if i > 0 and i % 100 == 0:
        nice_output += "\n"
print(nice_output)
print("Length: %s"%(sorted(candidates, reverse=True)[0][0]))

# I should really turn this into a function by now
# Anyway, lets see how 2-chargrams fare

base = "kaggle"

genes = list("abcdefghijklmnopqrstuvwxyz")

for population_id in range(500):
    candidates = []
    for gene_a in genes:
        for gene_b in genes:
            candidate = base + gene_a + gene_b
            tie_breaker = random.randint(0,10000) # to randomly break ties when sorting
            candidates.append(( len(zlib.compress(candidate.encode("ascii")))+
                                len(bz2.compress(candidate.encode("ascii"))),
                                tie_breaker,
                                candidate))
    base = sorted(candidates, reverse=True)[0][2]
print()    
nice_output = ""
for i, char in enumerate(base):
    nice_output += char
    if i > 0 and i % 100 == 0:
        nice_output += "\n"
print(nice_output)
print("Length: %s"%(sorted(candidates, reverse=True)[0][0]))