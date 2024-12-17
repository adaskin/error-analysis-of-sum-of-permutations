'''
copy of https://github.com/adaskin/stochastic-matrix-into-permutations/blob/main/main.py
includes functions to generate random stochastic matrix 
and find disjoint cycles after applying bvn, adaskin 2024
'''
from bvn import bvn
from itertools import permutations as perm
import numpy as np

def disjointcycles(p):
    '''given a permutation p in one row format, 
    find disjoint sets and return them
    '''
    n = len(p);
    icol = 0
    Cycles = []
    #to visit only once, 
    #first we marked all not-visited
    visited = {}
    icycle = 0 #cycle index
    for i in range(n):
        print("cycles", Cycles)
        Cycles.append([])
        current = p[i]
        cyclesize = 0
        while(current not in visited):
            visited[current] = 1
            Cycles[icycle].append(current)
            current = p[current]
            cyclesize += 1
        if(cyclesize == 0):
            Cycles.pop()
        else:
            icycle += 1
        
    return Cycles 



def rand_perm_matrix_sparse(n):
    from scipy.sparse import csr_matrix
    prow = np.random.permutation(n)
    P = csr_matrix((np.ones(n, dtype=int), (np.arange(n), prow)), shape=(n,n))
    return P, prow

def rand_perm_matrix(n):
    '''generate random permutation matrix of size n
    returns matrix and onerow-representation
    P <-> prow--row based representation: i.e. P[i, prow[i]] = 1
    '''
    rng = np.random.default_rng()
    prow = rng.permutation(n)

    #P = np.eye(3)  # 3x3 identity 
    #P = np.random.shuffle(P)  # shuffles rows

    P = np.zeros((n, n), dtype=int)
    P[np.arange(n),prow] = 1
    return P, prow


def rand_bistochastic(n, m = 10):
    ''' sum random m random permutation matrix 
    with m random coefficients 
    to generate a doubly stochastic matrix.
    '''
    S = 0;
    rng = np.random.default_rng()
    coeff = rng.integers(0, 10, m)
    L = []
    for i in range(m):
        P, prow = rand_perm_matrix(n)
        L.append(prow)
        S = S + coeff[i]*P;
    return S, L, coeff


if __name__ == "__main__":
    S, L, coeff =  rand_bistochastic(128, 15)
    Prows, weights = bvn(S, 0.1, 100)