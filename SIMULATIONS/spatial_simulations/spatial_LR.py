"""
saptial_CR.py: file taking in imput a dataframe in pkl form and running the spatial version of the networks specified inside

OUTPUTS a pickle file with         data['last_2_frames_N'] = last_2_frames_N
                                   data['current_R']       = current_R
                                   data['current_N']       = current_N
                                   data['g_rates']         = g_rates
                                   data['s_list']          = s_list
                                   data['abundances']      = abundances[::100]
                                   data['t_list']          = t_list
                                   data['biomass']         = biomass
        inside the dedicated Data folder, in addition to the same columns that the input dataframe already had

"""

import os
import sys
import pickle
import ast

import pandas as pd
import numpy as np

from pathlib import Path

import N_dynamics
import R_dynamics
import update

#----------------------------------------------------------------------------------------------------------------------
# import dataframe of networks already run and mapped
# as an example here we use the ones from the CR_to_LV file 
pickle_file_path = Path('/work/FAC/FBM/DMF/smitri/nccr_microbiomes/federica/MES_networks_generation_analysis/subsampled_df.pkl')

df = pd.read_pickle(pickle_file_path)
#----------------------------------------------------------------------------------------------------------------------

def spatial_CR(i):

    """
    i: int, index in the dataframe of the network to run

    """

    # create paths to save results
    # create paths to save results, these are examples, ad they will end up in the Data folder
    path = f'/work/FAC/FBM/DMF/smitri/nccr_microbiomes/federica/MES_spatial/spatial_LR'
    results_dir = f"{path}_results/{i}"
    os.makedirs(results_dir, exist_ok=True)

    n=100
    n_s=8
    n_supplied = df['n_supplied'].iloc[i]
    n_consumed = df['n_consumed'].iloc[i]

    # defining binary uptake matrix with 5 preferences for each species
    up_mat = df['C'].iloc[i][0]  
    n_r = up_mat.shape[1]

    # defining sign matrix (all positive nutrients here)
    sign_mat = np.ones((n_s,n_r))

    # no essential nutrients (only catabolic cross-feeding)
    mat_ess = np.zeros((n_s,n_r))

    # no auxotrophies (anyone can produce what metabolism allows)
    spec_met = np.ones((n_s,n_r))

    # create metabolic matrix of sparcity 0. with entries sampled from Dirichelet distribution: everyone produces everything
    met_mat = df['D'].iloc[i][0]

    # recapitulate in dictionary
    mat = {
        'uptake'  : up_mat,
        'met'     : met_mat,
        'ess'     : mat_ess,
        'spec_met': spec_met,
        'sign'    : sign_mat
    }

    # definition of the rest of the model parameters

    # set dilution
    param = {}
    param['l']=df['leakage'].iloc[i]*np.ones((n_r))
    param['w']=np.ones((n_r))/n_consumed
    param['g']=np.ones((n_s))
    param['ext']=np.zeros((n_r))
    param['ext'][:n_supplied]=float(df['supply'].iloc[i])/n_supplied
    param['tau']=100

    param['n']=100                                 
    param['sor']=1.55
    param['L']=100
    param['D']=10000
    param['Dz']=1e-4
    param['acc']=1e-5             

    # rescale influx so that wm and space compare in terms of supplied energy
    param['ext']=param['ext']/param['tau']

    # simulate in space
    # initial guesses and conditions
    R_space_ig = np.zeros((n,n,n_r))
    R_space_ig[:,:,param['ext']>0.]=param['ext'][0]/2
    N0_space   = np.zeros((n,n))
    N0_space   = N_dynamics.encode(np.random.randint(0, n_s, size=(n,n)),np.array(np.arange(n_s)))
    biomass = np.random.uniform(0, 2, (n, n)) 

    # define functions
    fR = R_dynamics.f
    gr = N_dynamics.growth_rates

    # spatial
    last_2_frames_N, _, current_R, current_N, g_rates, s_list, abundances, t_list, biomass  = update.simulate_3D_NBC(800000, fR, gr, R_space_ig, N0_space, biomass, param, mat)

    data = {
                                        'n_supplied': df.iloc[i]['n_supplied'],
                                        'n_consumed': df.iloc[i]['n_consumed'],
                                        'sparsity': df.iloc[i]['sparsity'],
                                        'noise':  df.iloc[i]['noise'],
                                        'PCS_var': df.iloc[i]['PCS_var'],
                                        'PCS_bias': df.iloc[i]['PCS_bias'],
                                        'leakage':df.iloc[i]['leakage'],
                                        'parameters':param,
                                        'replica':df.iloc[i]['replica'],
                                        'uptake':up_mat,
                                        'D':met_mat,
                                        'CR_R':df.iloc[i]['R_cr'], 
                                        'CR_N':df.iloc[i]['N_cr'], 
                                        'LV': df.iloc[i]['N_lv'],
                                        'g0':df.iloc[i]['g'],
                                        'A':df.iloc[i]['A']
                                    }

    data['last_2_frames_N'] = last_2_frames_N
    data['current_R']       = current_R
    data['current_N']       = current_N
    data['g_rates']         = g_rates
    data['s_list']          = s_list[0::50]
    data['abundances']      = abundances[0::50]
    data['t_list']          = t_list[0::50]
    data['biomass']         = biomass

    # output file path
    output_file = f'{results_dir}/all_data.pkl'

    # save as pickle
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)

    return 

    
#------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Check if there are command-line arguments
    if len(sys.argv) == 2:
        i = int(sys.argv[1])

        # Run the simulation with the provided parameters
        spatial_CR(i)
        print(f"Simulation completed for row number {i}")

    else:
        print("Usage: python spatial_CR.py")
