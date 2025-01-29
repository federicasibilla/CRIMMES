"""
functions to generate matrices in various ways

"""

import numpy as np

#--------------------------------------------------------------------------------------------------------------------------------
# complete_up: function to create a noisy structured uptake matrix

def complete_up(n_s, n_r, n_supplied, n_consumed, p=0., bias=1., PCS_var=0.1):
    """
    n_s:        int, number of species
    n_r:        int, number of resources
    n_supplied: int, supplied CS
    n_consumed: int, number of consumed CS per species
    n_produced: int, number of produced secondary CS per PCS
    p:          float, probability of moving an element to an empty spot in the row

    RETURNS up_mat: matrix, n_sxn_r, containing uptake preferences (binary matrix)
    """

    n_sec = n_r - n_s

    # first block: supplied resources (least possible overlap)
    b1=np.zeros((n_s,n_s))
    for i in range(n_s):
        b1[i, i % n_supplied] = np.random.normal(bias, PCS_var)

    # Second block: consumed resources
    b2 = np.zeros((n_s, n_sec))
    for i in range(n_s):
        for j in range(n_consumed):
            b2[i, (i*n_consumed + j) % n_sec] = np.random.normal(1, 0.1)

    up_mat = np.zeros((n_s, n_r))
    up_mat[:,:n_s]=b1
    up_mat[:, n_s:] = b2

    # Randomly move elements with probability p
    for i in range(up_mat.shape[0]):  # Iterate over rows
        for j in range(n_s,up_mat.shape[1]):  # Iterate over columns
            if up_mat[i, j] > 0:  # Only consider non-zero elements
                # With probability p, move the element to an empty spot in the same row
                if np.random.rand() < p:
                    # Find all empty spots (where the element is 0)
                    empty_spots = np.where(up_mat[i, n_s:] == 0)[0]
                    if len(empty_spots) > 0:  # Check if there are any empty spots
                        # Choose a random empty spot
                        empty_spot = np.random.choice(empty_spots)
                        # Swap the element with the empty spot
                        up_mat[i, n_s+empty_spot] = up_mat[i, j]
                        up_mat[i, j] = 0  # Make the original position empty


    return up_mat

#--------------------------------------------------------------------------------------------------------------------------------
# complete_met: metabolic matrix only looking at sparsity

def complete_met(n_s,n_r,n_supplied,sparsity):

    """
    n_s:        int, number of species
    n_r:        int, number of resources
    n_supplied: int, supplied CS
    n_produced: int, number of produced secondary CS per PCS

    RETURNS met_mat: matrix, n_rxn_r, containing metabolic production

    """

    n_sec = n_r-n_s

    # first block: no production of PCS
    b1 = np.zeros((n_s,n_r))    
    # second block: first produce all from primary, then set the remaining 
    b2 = np.zeros((n_sec,n_supplied))
    for i in range(b2.shape[0]):
        for j in range(b2.shape[1]):
            b2[i,j]=1 if np.random.rand() > sparsity else 0
    b3 = np.zeros((n_sec,n_sec))
    for i in range(b3.shape[0]):
        for j in range(b3.shape[1]):
            b3[i,j]=1 if np.random.rand() > sparsity else 0

    blocks = np.zeros((n_sec,n_r))
    blocks[:,:n_supplied]=b2
    blocks[:,n_s:]=b3


    met_mat=np.zeros((n_r,n_r))
    met_mat[:n_s,:]=b1
    met_mat[n_s:,:]=blocks

    # diagonal elements should be 0
    np.fill_diagonal(met_mat, 0)                                            

    # sample all from D. distribution
    for column in range(n_r):
        # Find indices where matrix[:, col_idx] == 1 (non-zero entries)
        non_zero_indices = np.where(met_mat[:, column] == 1)[0]  
        if len(non_zero_indices) > 0:
            # Sample from Dirichlet distribution for non-zero entries
            dirichlet_values = np.random.dirichlet(np.ones(len(non_zero_indices)))
            met_mat[non_zero_indices, column] = dirichlet_values


    return met_mat

