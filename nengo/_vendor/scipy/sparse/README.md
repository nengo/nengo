linalg_expm
-----------
This code for the matrix exponential is taken from `scipy.sparse`.

One function that could not be ported over (it is in C) is `scipy.special.comb`.
A replacement function `comb` is in `nonscipy_utils.py`.