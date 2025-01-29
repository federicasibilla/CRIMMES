"""
functions to generate matrices in various ways

"""

import numpy as np

#--------------------------------------------------------------------------------------------------------------------------------
# structured_up_relaxed: function to create a noisy structured uptake matrix

def complete_up_F(n_s, n_r, n_supplied, n_consumed, bias=1., PCS_var=0.1):
    """
    n_s:        int, number of species
    n_r:        int, number of resources
    n_supplied: int, supplied CS
    n_consumed: int, number of consumed CS per species
    n_produced: int, number of produced secondary CS per PCS
    p:          float, probability of moving an element to an empty spot in the row

    RETURNS up_mat: matrix, n_sxn_r, containing uptake preferences (binary matrix)
    """

    n_sec = n_r - n_supplied

    # first block: supplied resources (least possible overlap)
    b1=np.ones((n_s,1))*np.random.normal(bias,PCS_var,(n_s,1))
    
    # Second block: consumed resources
    b2 = np.zeros((n_s, n_sec))
    for i in range(n_s):
        # Select random positions in the sectors for the ones
        random_positions = np.random.choice(n_sec, n_consumed, replace=False)
        for pos in random_positions:
            # Assign a value drawn from a normal distribution to the selected position
            b2[i, pos] = np.random.normal(1, 0.1)

    up_mat = np.zeros((n_s, n_r))
    up_mat[:,:1]=b1
    up_mat[:, 1:] = b2

    return up_mat

#--------------------------------------------------------------------------------------------------------------------------------
# structured_up_relaxed: function to create a noisy structured uptake matrix

def complete_up_O(n_s, n_r, n_supplied, n_consumed, bias=1., PCS_var=0.1):
    """
    n_s:        int, number of species
    n_r:        int, number of resources
    n_supplied: int, supplied CS
    n_consumed: int, number of consumed CS per species
    n_produced: int, number of produced secondary CS per PCS
    p:          float, probability of moving an element to an empty spot in the row

    RETURNS up_mat: matrix, n_sxn_r, containing uptake preferences (binary matrix)
    """

    n_sec = n_r - n_supplied

    # first block: supplied resources (least possible overlap)
    b1=np.ones((n_s,1))*np.random.normal(bias,PCS_var,(n_s,1))
    
    # Second block: consumed resources
    b2 = np.zeros((n_s, n_sec))
    for i in range(n_s):
        # Select random positions in the sectors for the ones
        random_positions = np.random.choice(n_sec, n_consumed, replace=False)
        for pos in random_positions:
            # Assign a value drawn from a normal distribution to the selected position
            b2[i, pos] = np.random.normal(1, 0.1)

    up_mat = np.zeros((n_s, n_r))
    up_mat[:,:1]=b1
    up_mat[:, 1:] = b2

    return up_mat


#--------------------------------------------------------------------------------------------------------------------------------
# spec_met_vertical: production complementary to uptake


def spec_met_vertical(up_mat, n_producers):
    """
    up_mat:       La matrice di uptake (n_s x n_r) con zeri e valori diversi da zero.
    n_producers:   Il numero di uno che devono essere inseriti in ciascuna colonna di spec_met.
    
    Returns:
    spec_met:     Una matrice con uno e zero, seguendo le regole specificate.
    """

    # Inizializza spec_met con zeri (la stessa forma di up_mat)
    spec_met = np.zeros_like(up_mat)
    
    # Itera su ogni colonna della matrice up_mat
    for i in range(up_mat.shape[1]):
        # Trova le posizioni dove up_mat è zero
        zero_indices = np.where(up_mat[:, i] == 0)[0]
        
        # Se ci sono abbastanza zeri da soddisfare n_consumed, seleziona casualmente n_consumed zeri
        if len(zero_indices) >= n_producers:
            selected_indices = np.random.choice(zero_indices, n_producers, replace=False)
            spec_met[selected_indices, i] = 1
        else:
            # Se ci sono meno zeri di n_consumed, inserisci 1 solo nelle posizioni disponibili
            spec_met[zero_indices, i] = 1
    
    return spec_met

#--------------------------------------------------------------------------------------------------------------------------------
# fill_dirichelet, columns

def fill_dirichlet_col(mat):

    for column in range(mat.shape[1]):
        # Find indices where matrix[:, col_idx] == 1 (non-zero entries)
        non_zero_indices = np.where(mat[:, column] == 1)[0]  
        if len(non_zero_indices) > 0:
            # Sample from Dirichlet distribution for non-zero entries
            dirichlet_values = np.random.dirichlet(np.ones(len(non_zero_indices))*10)
            mat[non_zero_indices, column] = dirichlet_values
    return mat
