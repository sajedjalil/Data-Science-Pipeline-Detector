import numpy as np

def sumPrimesTo(n):
    # n is ceiling of the primes requested

    # Create a list of numbers up to n
    nums = np.arange(2,n, dtype='uint32')

    currInd = 0
    pSum = 0
    while len(nums):
        # If we are here, nums[currInd] must be prime.
        # So we must filter out all multiples of nums[currInd]
        pSum += nums[currInd]
        nums = nums[nums % nums[currInd] != 0]
    return pSum


print(sumPrimesTo(2000000)) # Gives the answer : 142913828922