'''
copy of https://github.com/adaskin/qcompiling-by-permutations/blob/main/bvn.py
birkhoff von Neuman algorithm by using networkx for bipartite graph matching

Example run for doubly stochastic matrix S, error 0.1, niteration 1000
    Prows, weights = bvn(S, 0.1, 1000)
Prows is the one-row representation of permutations, 
weights are the list of weights for each permutation.

Note: if wanted, you can adjust bvn to return matrix form.

adaskin, 2024
'''
import networkx as nx
import numpy as np

def get_maximal_matching(A):
    '''A adjacency matrix,
        return maximal bipartite matching
    '''
    n = len(A)
    #convert rows and columns into vertices
    # M = np.zeros((2*n,2*n))
    # M[0:n,n:2*n] = A
    # M[n:2*n, 0:n] = A
    
    #M is a bipartite graph
    G = construct_bipartite_graph(A)
    u = [n for n in G.nodes if G.nodes[n]['bipartite'] == 0]

    #nx.from_numpy_array(M);
    matching = nx.bipartite.maximum_matching(G,top_nodes=u)
    return matching

def get_maximal_matching_old(A):
    '''A adjacency matrix,
        return maximal bipartite matching
        extends matrix..
    '''
    n = len(A)
    #convert rows and columns into vertices
    M = np.zeros((2*n,2*n))
    M[0:n,n:2*n] = A
    M[n:2*n, 0:n] = A
    
    #M is a bipartite graph
    #G = constrcut_bipartite_graph(A)
    G = nx.from_numpy_array(M);
    u = [n for n in G.nodes if G.nodes[n]['bipartite'] == 0]

    #nx.from_numpy_array(M);
    matching = nx.bipartite.maximum_matching(G,top_nodes=u)
    return matching

def construct_bipartite_graph(A):
    n = len(A)#number of nodes
    B = nx.Graph()
    
    # Add nodes with the node attribute "bipartite"
    row_nodes = np.arange(n)
    col_nodes = np.arange(n) + n;
    
    B.add_nodes_from(row_nodes, bipartite=0)
    
    B.add_nodes_from(col_nodes, bipartite=1)
    for i in range(n):
        for j in range(n):
            if A[i,j] != 0:
                
                # Add edges only between nodes of opposite node sets
                B.add_edge(i,j+n)
    return B

def permutation_from_bipartite_matching(matching):
    '''given bipartite matching m a dictionary, generate a permutation matrix'''
    n = int(len(matching)/2)
    
    prow = np.zeros(n, dtype=int); #row representation of the permutation
    for  i in range(n):
        a = i 
        b = matching[i] % n
       # print("a, b", a, b, i, matching[i])
        prow[a] = b

    
    #matrix form for the permutatiuon
    P = np.zeros((n, n), dtype=int)
    P[np.arange(n),prow] = 1
        
    return P, prow;

def find_permutation(S):
    '''given stochastic matrix S, find a permutation'''
    matching = get_maximal_matching(S)
    #print('matching', matching)
    P, prow = permutation_from_bipartite_matching(matching) 
    return P, prow    


def compute_weight(Si, Pi):
    '''given permutation and stochastic matrices, 
    computes the minimum element that corresponds to 1 in the permutation,
    returns it as weight
    '''
    nonzeros = np.nonzero(Pi)
    #print(Si)
    #print(Si-Pi)
    minweight = np.min(Si[nonzeros])

    return minweight

def bvn(S, precision=0.001, max_iter = 100):
    '''Birkhoff von Neuman Decomposition of bistochastic S
        returns dictionary P: permutations
        and list of coefficients p
    '''
    P = {}
    Prows = [] #permutations in one-row format
    weights = [] #weights for each permutaiton
    Si = S.copy();
    epsilon = np.linalg.norm(Si,1);  
    print("error norm: ", epsilon)
    i = 0;
    while epsilon > precision and i < max_iter:
       # print(Si)
        Pi, prow = find_permutation(Si)
        
        weight = compute_weight(Si, Pi)
        print(f'iteration: {i}, weight: {weight}, permutation:', prow)
        Si = Si - weight*Pi
        weights.append(weight)
       # P[i] = Pi #matrix forms
        Prows.append(prow)
        epsilon = np.linalg.norm(Si,1);
        print(f"error norm: {epsilon}\n")
        i += 1
        
    return Prows, weights

#if __name__ == "__main__":


