# From https://github.com/scipy/scipy/blob/v1.3.0/scipy/special/_comb.pyx
def comb(N, k):
    """Compute N choose k"""
    if k > N or N < 0 or k < 0:
        return 0

    M = N + 1
    nterms = min(k, N - k)

    numerator = 1
    denominator = 1
    for j in range(1, nterms + 1):
        numerator *= M - j
        denominator *= j

    return numerator // denominator
