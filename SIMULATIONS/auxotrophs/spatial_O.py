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
import definitions

#----------------------------------------------------------------------------------------------------------------------
# import dataframe of networks already run and mapped
# as an example here we use the ones from the CR_to_LV file 
pickle_file_path = Path('/work/FAC/FBM/DMF/smitri/nccr_microbiomes/federica/MES_aux_sensitivity_wm_analysis/aux.pkl')

df = pd.read_pickle(pickle_file_path)
#----------------------------------------------------------------------------------------------------------------------

def spatial_O(i):

    """
    i: int, index in the dataframe of the network to run

    """

    # create paths to save results
    # create paths to save results, these are examples, ad they will end up in the Data folder
    path = f'/work/FAC/FBM/DMF/smitri/nccr_microbiomes/federica/MES_aux/spatial_O_MR'
    results_dir = f"{path}_results/{i}"
    os.makedirs(results_dir, exist_ok=True)

    replica = df['replica'].iloc[i]
    n_consumed = df['n_consumed'].iloc[i]
    n_producers = df['n_produced'].iloc[i]
    PCS_bias = df['PCS_bias'].iloc[i]
    PCS_var = df['PCS_var'].iloc[i]
    leakage = df['leakage'].iloc[i]

    # set seed for random generations
    np.random.seed(replica)

    # number of species and number of nutrients
    n_s = 8            # 8 species
    n_r = 21           # maximum of 16 CS
    n_ext = 1          # one supplied CS everyone can consume
    n = 100

    # generate uptake and metabolic matrices
    up_mat_F  = definitions.complete_up_F(n_s, n_r, n_ext, n_consumed, bias=PCS_bias, PCS_var=PCS_var)
    # for obligate fill first row with normal around one with same variance
    up_mat_O  = np.zeros_like(up_mat_F)
    up_mat_O[:,0]=up_mat_F[:,0].copy()
    mask = up_mat_F[:, 1:] != 0
    up_mat_O[:,1:][mask]=1

    # metabolic matrix: vertical production
    met_mat_F = np.zeros((n_r,n_r))
    met_mat_O = np.zeros((n_r,n_r))
    met_mat_F[1:,:]=1
    np.fill_diagonal(met_mat_F,0)
    met_mat_O[1:,0]=1
    np.fill_diagonal(met_mat_O[1:,1:],1)
    met_mat_F=definitions.fill_dirichlet_col(met_mat_F)
    met_mat_O=definitions.fill_dirichlet_col(met_mat_O)

    # all nutrients apport positive contributions to gr
    sign_mat_F = np.ones((n_s,n_r))
    sign_mat_O = np.zeros((n_s,n_r))
    sign_mat_O[:,0]=1
    
    # production specific (you produce n_produced among the ones you don't consume)
    spec_met = definitions.spec_met_vertical(up_mat_F,n_producers)

    # essential specific
    mat_ess_F  = np.zeros((n_s,n_r))
    mat_ess_O  = np.where(up_mat_O!=0,1,0)
    mat_ess_O[:,0] = 0

    mat_O = {
        'uptake'  : up_mat_O,
        'met'     : met_mat_O,
        'ess'     : mat_ess_O,
        'spec_met': spec_met,
        'sign'    : sign_mat_O
    }

    # set dilution
    tau = 100

    # growth and maintainence
    g_F = np.ones((n_s))*1
    g_O = np.ones((n_s))*1
    m = np.zeros((n_s))+1/tau

    # reinsertion of chemicals
    tau = np.zeros((n_r))+tau 
    ext_F = np.zeros((n_r))
    ext_O = np.zeros((n_r))
    # primary carbon sources replenished to saturation
    ext_F[:n_ext] = 10000
    ext_O[:n_ext] = 10000

    # initial guess for resources
    guess = np.ones((n_r))*10000

    # leakage
    l_F = np.ones((n_r))*leakage
    l_O = np.zeros((n_r))
    l_O[0] = leakage

    # define parameters
    param = {
        'w'  : np.ones((n_r))/(n_consumed+1),              # energy conversion     [energy/mass]
        'l'  : l_O,                                        # leakage               [adim]
        'g'  : g_O,                                        # growth conv. factors  [1/energy]
        'm'  : m,                                          # maintainance requ.    [energy/time]
        'ext': ext_O,                                      # external replenishment  
        'tau' : tau,                                       # chemicals dilution                                            
        'guess_wm': guess                                  # initial resources guess
    }

    # definition of the rest of the model parameters
    
    param['n']=n                                 
    param['sor']=1.55
    param['L']=100
    param['D']=100
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
    fR = R_dynamics.f_maslov
    gr = N_dynamics.growth_rates_maslov

    # spatial
    last_2_frames_N, _, current_R, current_N, g_rates, s_list, abundances, t_list, biomass  = update.simulate_3D_NBC(500000, fR, gr, R_space_ig, N0_space, biomass, param, mat_O)

    data = {
                                        'n_consumed': df['n_consumed'].iloc[i],
                                        'n_produced': df['n_produced'].iloc[i],
                                        'parameters':param,
                                        'replica':df['replica'].iloc[i],
                                        'C_O':up_mat_O,
                                        'D_O':met_mat_O,
                                        'spec':spec_met,
                                        'mat': mat_O,
                                        'CR_R':df['CR_R_O'].iloc[i], 
                                        'CR_N':df['CR_N_O'].iloc[i], 
                                        'g0':df['g0_O'].iloc[i],
                                        'A':df['A_O'].iloc[i]
                                    }

    data['last_2_frames_N'] = last_2_frames_N
    data['current_R']       = current_R
    data['current_N']       = current_N
    data['g_rates']         = g_rates
    data['s_list']          = s_list
    data['abundances']      = abundances[::200]
    data['t_list']          = t_list
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
        spatial_O(i)
        print(f"Simulation completed for row number {i}")

    else:
        print("Usage: python spatial_random_O.py")
