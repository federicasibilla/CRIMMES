"""
definitions.py: functions to generate matrices with different protocols

CONTAINS: 
        FUNCTIONS INTENDED FOR THE SUBSTITUTABLE/OBLIGATE CF COMPARISON (i.e. functions assume one PCS)
          - tradeoff_cooperative_up_F: generating uptake for facultative CF simulations, with C matrix biased against competition in its structure, and relaxed tradeoffs on each species's uptakes
          - uniform_cooperative_up_O: generating uptake for obligate CF simulations, C matrix biased against competition, C is binary a part from the PCS column
          - tradeoff_random_up_F: same as 'tradeoff_cooperative_up_F' but no bias in the uptake vectors structure
          - uniform_random_up_O: same as 'uniform_cooperative_up_O' but no bias in the uptake vectors structure
          - spec_met_complem: generating the spec_met matrix, each species can produce the first n_produced resources after the ones it consumes
          - spec_met_vertical: here focus on how many producers each resource has 
          - fill_dirichlet_col: general function to fill the columns of a matrix from Dirichlet distribution
        FUNCTIONS INTENDED TO GENERATE CR NETWORKS WITH VARIOUS PROTOCOLS
          - tradeoff_up: uptake with relaxed tradeoffs on rows, random uptaken resources
          - simple_met: metabolic matrix with specified number of metaboloite produced per resource
          - structured_up_relaxed: uptake with structure bias against competition, bias can be relaxed with some noisy displacement
          - structured_met_relaxed: metabolism complements uptake in a structured way (function to get high p+-low connecitivity networks)
          - up_binary: function to create a binary uptake matrix
          - up_relaxed_tradeoffs_sparsity: function to create C of a given sparsity with tradeoffs on consumption, but not exact
          - up_relaxed_tradeoffs_biased_sparsity: bias against PCS consumption
          - met_dir_sparsity: function to sample D coefficients from Dirichelet distribution, with fixed sparcity

"""

import numpy as np

#--------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS INTENDED FOR THE SUBSTITUTABLE/OBLIGATE CF COMPARISON
#--------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------
# tradeoff_cooperative_up_F: up matrix biased against competition, assumes n_supplied=1
# consumption of the primary carbon source is drawn from normal distribution around 1, the rest of uptakes sum to 1 (relaxed)

def tradeoff_cooperative_up_F(n_s,n_r,n_consumed):

    """
    n_s:        int, number of species
    n_r:        int, number of resources
    n_consumed: int, number of consumed CS per species (ATT: needs to be less than n_r-1)

    RETURNS up_mat: matrix, n_sxn_r, containing uptake preferences (binary matrix)

    """

    n_sec = n_r-1

    # first block: supplied resources (least possible overlap)
    b1=np.random.normal(1., 0.1, size=(n_s,1))

    # extract non zero
    b2 = np.zeros((n_s,n_sec))
    for i in range(n_s):
        for j in range(n_consumed):
            b2[i,(i*n_consumed+j)%n_sec] = 1

    up_mat = np.zeros((n_s,n_r))
    up_mat[:,1:]=b2

    # sample all non-zero entries from D. distribution
    for row in range(n_s):
        # Find indices where matrix[:, col_idx] == 1 (non-zero entries)
        non_zero_indices = np.where(up_mat[row,:] == 1)[0]  
        if len(non_zero_indices) > 0:
            # Sample from Dirichlet distribution for non-zero entries
            dirichlet_values = np.random.dirichlet(np.ones(len(non_zero_indices)))
            up_mat[row,non_zero_indices] = dirichlet_values
    
    # relax tradeoffs by multiplying each non zero entry for a N(1,0.01)
    for i in range(up_mat.shape[0]):
        for j in range(up_mat.shape[1]):
            up_mat[i,j]=up_mat[i,j]*np.random.normal(1.0,0.01)

    # normal distribution arou nd one for uptake of primary
    up_mat[:,:1]=b1

    return up_mat

#--------------------------------------------------------------------------------------------------------------------------------
# uniform_cooperative_up_O: up matrix biased against competition, assumes n_supplied=1
# PCS column drawn from normal distribution around 1, the rest is binary (1 for consumed, 0 for not consumed)

def uniform_cooperative_up_O(n_s,n_r,n_consumed):

    """
    n_s:        int, number of species
    n_r:        int, number of resources
    n_consumed: int, number of consumed CS per species

    RETURNS up_mat: matrix, n_sxn_r, containing uptake preferences (binary matrix)

    """

    n_sec = n_r-1

    # first block: supplied resource, uniform distribution of rates
    b1=np.random.normal(1., 0.1, size=(n_s,1))

    # complete with auxotrophies (least possible overlap)
    b2 = np.zeros((n_s,n_sec))
    for i in range(n_s):
        for j in range(n_consumed):
            b2[i,(i*n_consumed+j)%n_sec] = 1

    up_mat = np.zeros((n_s,n_r))
    up_mat[:,:1]=b1
    up_mat[:,1:]=b2

    return up_mat

#--------------------------------------------------------------------------------------------------------------------------------
# tradeoff_random_up_F: up matrix biased against competition, assumes n_supplied=1
# no bias against competition in the structure

def tradeoff_random_up_F(n_s,n_r,n_consumed):

    """
    n_s:        int, number of species
    n_r:        int, number of resources
    n_consumed: int, number of consumed CS per species

    RETURNS up_mat: matrix, n_sxn_r, containing uptake preferences (binary matrix)

    """

    n_sec = n_r-1

    # first block: supplied resources (least possible overlap)
    b1=np.random.normal(1., 0.1, size=(n_s,1))

    # extract non zero
    b2 = np.zeros((n_s,n_sec))
    for i in range(n_s):
        random_indices = np.random.choice(range(1,n_r-1), n_consumed, replace=False)  # Randomly select n_ones indices
        b2[i, random_indices] = 1 

    up_mat = np.zeros((n_s,n_r))
    up_mat[:,1:]=b2

    # sample all non-zero entries from D. distribution
    for row in range(n_s):
        # Find indices where matrix[:, col_idx] == 1 (non-zero entries)
        non_zero_indices = np.where(up_mat[row,:] == 1)[0]  
        if len(non_zero_indices) > 0:
            # Sample from Dirichlet distribution for non-zero entries
            dirichlet_values = np.random.dirichlet(np.ones(len(non_zero_indices)))
            up_mat[row,non_zero_indices] = dirichlet_values
    
    # relax tradeoffs by multiplying each non zero entry for a N(1,0.01)
    for i in range(up_mat.shape[0]):
        for j in range(up_mat.shape[1]):
            up_mat[i,j]=up_mat[i,j]*np.random.normal(1.0,0.01)

    up_mat[:,:1]=b1

    return up_mat

#--------------------------------------------------------------------------------------------------------------------------------
# uniform_cooperative_up_O: up matrix biased against competition, assumes n_supplied=1
# structure o fuptake vectors not biased against competition

def uniform_random_up_O(n_s,n_r,n_consumed):

    """
    n_s:        int, number of species
    n_r:        int, number of resources
    n_consumed: int, number of consumed CS per species

    RETURNS up_mat: matrix, n_sxn_r, containing uptake preferences (binary matrix)

    """

    n_sec = n_r-1

    # first block: supplied resource, uniform distribution of rates
    b1=np.random.normal(1., 0.1, size=(n_s,1))

    # complete with auxotrophies (least possible overlap)
    b2 = np.zeros((n_s,n_sec))
    for i in range(n_s):
        random_indices = np.random.choice(range(1,n_r-1), n_consumed, replace=False)  # Randomly select n_ones indices
        b2[i, random_indices] = 1

    up_mat = np.zeros((n_s,n_r))
    up_mat[:,:1]=b1
    up_mat[:,1:]=b2

    return up_mat

#--------------------------------------------------------------------------------------------------------------------------------
# spec_met_complem: production complementary to uptake

def spec_met_complem(n_s,n_r,n_consumed,n_produced):

    """
    n_s,n_r:    int, species and resources
    n_consumed: int, number of consumed CS per species
    n_produced: int, produced by each species

    RETURNS spec_met

    """

    spec_met = np.zeros((n_s,n_r))

    for i in range(n_s):
        for j in range(n_produced):
            if n_consumed*n_s<=n_r-1:
                spec_met[i,1+((i*n_consumed+n_consumed+j)%(n_consumed*n_s))]=1
            else:
                spec_met[i,1+((i*n_consumed+n_consumed+j)%(n_r-1))]=1

    return spec_met

#--------------------------------------------------------------------------------------------------------------------------------
# spec_met_complem: production complementary to uptake
# production goes in order after the consumption 

def spec_met_random(n_s,up_mat,n_produced):

    """
    n_s,n_r:    int, species and resources
    up_mat:     matrix, uptake
    n_produced: int, produced by each species

    RETURNS spec_met

    """

    spec_met = 1-np.where(up_mat!=0,1,0)

    for i in range(n_s):
        # Find indices of ones in the current row
        ones_indices = np.where(spec_met[i] == 1)[0]
        
        # If the row has more than n_ones ones, randomly select n_ones to keep
        if len(ones_indices) > n_produced:
            keep_indices = np.random.choice(ones_indices, n_produced, replace=False)
            # Set all elements to zero first
            spec_met[i] = 0
            # Set the selected indices to one
            spec_met[i, keep_indices] = 1

    return spec_met

#--------------------------------------------------------------------------------------------------------------------------------
# spec_met_vertical: production complementary to uptake
# here the focus is on how many species can produce each metabolite, and the ordering is random

def spec_met_vertical(n_s,up_mat,n_produced):

    """
    n_s,n_r:    int, species and resources
    up_mat:     matrix, uptake
    n_produced: int, how many species can produce each metabolite

    RETURNS spec_met

    """

    spec_met = 1-np.where(up_mat!=0,1,0)
    n_r = up_mat.shape[1]

    for i in range(1,n_r):
        # Find indices of ones in the current row
        ones_indices = np.where(spec_met[:,i] == 1)[0]
        
        # If the row has more than n_ones ones, randomly select n_ones to keep
        if len(ones_indices) > n_produced:
            keep_indices = np.random.choice(ones_indices, n_produced, replace=False)
            # Set all elements to zero first
            spec_met[:,i] = 0
            # Set the selected indices to one
            spec_met[keep_indices,i] = 1

    return spec_met

#--------------------------------------------------------------------------------------------------------------------------------
# fill_dirichelet, columns
# filling a column from Dirichlet distribution wher ethe entries are non zero

def fill_dirichlet_col(mat):

    """
    mat: matrix, any shape, containing ones and zeros

    RETURNS mat: matrix, filled from Dirichlet distribution on the columns where non-zero entries 

    """

    for column in range(mat.shape[1]):
        # Find indices where matrix[:, col_idx] == 1 (non-zero entries)
        non_zero_indices = np.where(mat[:, column] == 1)[0]  
        if len(non_zero_indices) > 0:
            # Sample from Dirichlet distribution for non-zero entries
            dirichlet_values = np.random.dirichlet(np.ones(len(non_zero_indices)))
            mat[non_zero_indices, column] = dirichlet_values
    return mat


#--------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS INTENDED TO GENERATE CR NETWORKS WITH VARIOUS PROTOCOLS
#--------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------
# tradeoff_up: relaxed tradeoffs on rows

def tradeoff_up(n_s,n_r,n_supplied,n_consumed,bias=1.):

    """
    n_s:        int, number of species
    n_r:        int, number of resources
    n_supplied: int, supplied CS
    n_consumed: int, number of consumed CS per species

    RETURNS up_mat: matrix, n_sxn_r, containing uptake preferences (binary matrix)

    """

    # extract non zero
    b2 = np.zeros((n_s,n_r-n_s))
    for i in range(n_s):
        # Randomly select positions for '1's
        ones_positions = np.random.choice(n_r-n_s, size=n_consumed, replace=False)
        # Set the selected positions to '1'
        b2[i, ones_positions] = 1

    up_mat=np.zeros((n_s,n_r))
    up_mat[:,n_s:]=b2 

    # sample all non-zero entries from D. distribution
    for row in range(n_s):
        # Find indices where matrix[:, col_idx] == 1 (non-zero entries)
        non_zero_indices = np.where(up_mat[row,:] == 1)[0]  
        if len(non_zero_indices) > 0:
            # Sample from Dirichlet distribution for non-zero entries
            biased_parameters = np.ones(len(non_zero_indices))
            biased_parameters[0] = bias
            # Sample from Dirichlet distribution for non-zero entries
            dirichlet_values = np.random.dirichlet(biased_parameters)
            up_mat[row,non_zero_indices] = dirichlet_values
    
    # relax tradeoffs by multiplying each non zero entry for a N(1,0.01)
    for i in range(up_mat.shape[0]):
        for j in range(up_mat.shape[1]):
            up_mat[i,j]=up_mat[i,j]*np.random.normal(1.0,0.01)

    # first block: supplied resources (least possible overlap)
    b1=np.zeros((n_s,n_s))
    for i in range(n_s):
        b1[i, i % n_supplied] = np.random.normal(0.2, 0.005)

    up_mat[:,:n_s]=b1

    return up_mat

#--------------------------------------------------------------------------------------------------------------------------------
# simple_met: function to create a D-sampled D matrix with given n_produced

def simple_met(n_s,n_r,n_supplied,n_produced):

    """
    n_s:        int, number of species
    n_r:        int, number of resources
    n_supplied: int, supplied CS
    n_produced: int, number of produced secondary CS per PCS

    RETURNS met_mat: matrix, n_rxn_r, containing metabolic production

    """

    # first block: no production of PCS
    b1 = np.zeros((n_s,n_r))    
    # second block: first produce all from primary, then set the remaining 
    if n_produced<=n_r-n_s:  
        b2 = np.zeros((n_r-n_s,n_s))
        for column in range(n_supplied):
            ones_positions = np.random.choice(n_r-n_s, size=n_produced, replace=False)
            # Set the selected positions to '1'
            b2[ones_positions,column] = 1
        b3 = np.zeros((n_r-n_s,n_r-n_s))
    else:
        b2 = np.ones((n_r-n_s,n_s))
        b3 = np.ones((n_r-n_s,n_r-n_s))

    met_mat=np.zeros((n_r,n_r))
    met_mat[:n_s,:]=b1
    met_mat[n_s:,:n_s]=b2
    met_mat[n_s:,n_s:]=b3

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

#--------------------------------------------------------------------------------------------------------------------------------
# structured_up_relaxed: function to create a noisy structured uptake matrix

def structured_up_relaxed(n_s, n_r, n_supplied, n_consumed, n_produced, p=0.2, bias=1.):
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

    # Second block: consumed resources
    b2 = np.zeros((n_s, n_sec))
    for i in range(n_s):
        for j in range(n_consumed):
            b2[i, (i + j + n_produced) % n_sec] = 1

    up_mat = np.zeros((n_s, n_r))
    up_mat[:, n_s:] = b2

    # Sample all non-zero entries from Dirichlet distribution
    for row in range(n_s):
        non_zero_indices = np.where(up_mat[row, :] == 1)[0]  
        if len(non_zero_indices) > 0:
            biased_parameters = np.ones(len(non_zero_indices))
            biased_parameters[0] = bias  # Bias the first resource in each row
            dirichlet_values = np.random.dirichlet(biased_parameters)
            up_mat[row, non_zero_indices] = dirichlet_values

    # Relax tradeoffs by multiplying each non-zero entry by a N(1, 0.01)
    for i in range(up_mat.shape[0]):
        for j in range(up_mat.shape[1]):
            if up_mat[i, j] > 0:
                up_mat[i, j] = up_mat[i, j] * np.random.normal(1.0, 0.01)

    # Randomly move elements with probability p
    for i in range(up_mat.shape[0]):  # Iterate over rows
        for j in range(up_mat.shape[1]):  # Iterate over columns
            if up_mat[i, j] > 0:  # Only consider non-zero elements
                # With probability p, move the element to an empty spot in the same row
                if np.random.rand() < p:
                    # Find all empty spots (where the element is 0)
                    empty_spots = np.where(up_mat[i, :] == 0)[0]
                    if len(empty_spots) > 0:  # Check if there are any empty spots
                        # Choose a random empty spot
                        empty_spot = np.random.choice(empty_spots)
                        # Swap the element with the empty spot
                        up_mat[i, empty_spot] = up_mat[i, j]
                        up_mat[i, j] = 0  # Make the original position empty

    # first block: supplied resources (least possible overlap)
    b1=np.zeros((n_s,n_s))
    for i in range(n_s):
        b1[i, i % n_supplied] = np.random.normal(0.2, 0.005)

    up_mat[:,:n_s]=b1

    return up_mat

#--------------------------------------------------------------------------------------------------------------------------------
# structured_met_relaxed: function to create a D-sampled D matrix with given n_produced

def structured_met_relaxed(n_s,n_r,n_supplied,n_produced,p=0.2):

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
    if n_produced<=n_sec:  
        b2 = np.zeros((n_sec,n_s))
        for i in range(n_supplied):
            for j in range(n_produced):
                b2[(i*n_produced+j) % n_sec,i] = 1
        b3 = np.zeros((n_sec,n_sec))
    else:
        b2 = np.zeros((n_sec,n_s))
        for i in range(n_supplied):
            for j in range(n_produced):
                b2[(i*n_produced+j) % n_sec,i] = 1
        b3 = b2.copy()

    blocks = np.zeros((n_sec,n_r))
    blocks[:,:n_s]=b2
    blocks[:,n_s:]=b3
    # Randomly move elements with probability p
    for i in range(blocks.shape[0]):  # Iterate over rows
        for j in range(blocks.shape[1]):  # Iterate over columns
            if blocks[i, j] > 0:  # Only consider non-zero elements
                # With probability p, move the element to an empty spot in the same row
                if np.random.rand() < p:
                    # Find all empty spots (where the element is 0)
                    empty_spots = np.where(blocks[:, j] == 0)[0]
                    if len(empty_spots) > 0:  # Check if there are any empty spots
                        # Choose a random empty spot
                        empty_spot = np.random.choice(empty_spots)
                        # Swap the element with the empty spot
                        blocks[empty_spot,j] = blocks[i, j]
                        blocks[i, j] = 0  # Make the original position empty

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

#-------------------------------------------------------------------------------------------------------------
# up_binary: function to create a binary uptake matrix

def up_binary(n_s,n_r,n_pref):

    """
    n_s:    int, number of species
    n_r:    int, number of resources
    n_pref: int, number of resources consumed by each species

    RETURNS up_mat: matrix, n_sxn_r, containing uptake preferences (binary matrix)
    """

    up_mat = np.zeros((n_s,n_r))
    # each species has a given number of preferred resources
    for i in range(n_s):
        ones_indices = np.random.choice(n_r, n_pref, replace=False)
        up_mat[i, ones_indices] = 1
    # check that someone eats primary source
    if (up_mat[:,0] == 0).all():
        up_mat[np.random.randint(0, n_s-1),0] = 1

    return up_mat

#-------------------------------------------------------------------------------------------------------------
# up_relaxed_tradeoffs_sparsity: like goldford in emergent simplicity, plus adding the layer of sparsity (tunes generalists vs specialists)

def up_relaxed_tradeoffs_sparsity(n_s,n_r,sparsity):

    """
    n_s:      int, number of species
    n_r:      int, number of resources
    sparsity: float, sparsity of the C matrix

    RETURNS up_mat: matrix, n_sxn_r, containing uptake preferences 

    """

    up_mat = np.ones((n_s,n_r))*(np.random.rand(n_s, n_r) > sparsity)     # make metabolic matrix sparse
    # add consumption of PCS
    if (up_mat[:,0]==0).all():
        up_mat[np.random.randint(0,n_s),0]=1

    # sample all non-zero entries from D. distribution
    for row in range(n_s):
        # Find indices where matrix[:, col_idx] == 1 (non-zero entries)
        non_zero_indices = np.where(up_mat[row,:] == 1)[0]  
        if len(non_zero_indices) > 0:
            # Sample from Dirichlet distribution for non-zero entries
            dirichlet_values = np.random.dirichlet(np.ones(len(non_zero_indices)))
            up_mat[row,non_zero_indices] = dirichlet_values
    
    # relax tradeoffs by multiplying each non zero entry for a N(1,0.01)
    for i in range(up_mat.shape[0]):
        for j in range(up_mat.shape[1]):
            up_mat[i,j]=up_mat[i,j]*np.random.normal(1.0,0.01)

    return up_mat

#-------------------------------------------------------------------------------------------------------------
# up_relaxed_tradeoffs_biased_sparsity: modify concentration of D. distribution to bias against PCS

def up_relaxed_tradeoffs_biased_sparsity(n_s,n_r,sparsity,bias):

    """
    n_s:      int, number of species
    n_r:      int, number of resources
    sparsity: float, sparsity of the C matrix
    bias: float, parameter for D. distribution on PCS, relative to 1

    RETURNS up_mat: matrix, n_sxn_r, containing uptake preferences 

    """

    # extract non zero, forcing at least one to be PCS
    up_mat = np.ones((n_s,n_r))*(np.random.rand(n_s, n_r) > sparsity)
    while((up_mat[:,0]==0).all()):
        up_mat = np.ones((n_s,n_r))*(np.random.rand(n_s, n_r) > sparsity)     # make metabolic matrix sparse

    # sample all non-zero entries from D. distribution
    for row in range(n_s):
        # Find indices where matrix[:, col_idx] == 1 (non-zero entries)
        non_zero_indices = np.where(up_mat[row,:] == 1)[0]  
        if len(non_zero_indices) > 0:
            biased_parameters = np.ones(len(non_zero_indices))
            biased_parameters[0] = bias
            # Sample from Dirichlet distribution for non-zero entries
            dirichlet_values = np.random.dirichlet(biased_parameters)
            up_mat[row,non_zero_indices] = dirichlet_values
    
    # relax tradeoffs by multiplying each non zero entry for a N(1,0.01)
    for i in range(up_mat.shape[0]):
        for j in range(up_mat.shape[1]):
            up_mat[i,j]=up_mat[i,j]*np.random.normal(1.0,0.01)

    return up_mat

#-------------------------------------------------------------------------------------------------------------
# met_dir_sparsity: function to create a D-sampled D matrix with given sparcity

def met_dir_sparsity(n_r,sparcity):

    """
    n_r:      int, number of reosurces
    sparsity: float, sparsity of the metabolic matrix

    RETURNS met_mat: matrix, n_rxn_r, metabolic matrix 

    """

    met_mat = np.ones((n_r,n_r))*(np.random.rand(n_r, n_r) > sparcity)      # make metabolic matrix sparce
    met_mat[0,:] = 0                                                        # carbon source is not produced
    np.fill_diagonal(met_mat, 0)                                            # diagonal elements should be 0
    # check that at least one thing is produced from primary carbon source
    if (met_mat[:,0] == 0).all():
        met_mat[np.random.randint(0, n_r-1),0] = 1
    # sample all from D. distribution
    for column in range(n_r):
        # Find indices where matrix[:, col_idx] == 1 (non-zero entries)
        non_zero_indices = np.where(met_mat[:, column] == 1)[0]  
        if len(non_zero_indices) > 0:
            # Sample from Dirichlet distribution for non-zero entries
            dirichlet_values = np.random.dirichlet(np.ones(len(non_zero_indices)))
            met_mat[non_zero_indices, column] = dirichlet_values

    return met_mat
