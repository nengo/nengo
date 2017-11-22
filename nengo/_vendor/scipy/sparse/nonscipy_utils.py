import math

# simple custom implementation of n choose k
def comb(n, k):
    """Compute n choose k"""
    if k > n or k < 0:
        return 0
    if k == 0:
        return 1
    return math.factorial(n) // math.factorial(n - k) // math.factorial(k)
