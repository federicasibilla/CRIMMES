"""
protocol_1.py: script to simulate CR network that map onto LV covering a span of connectivity and 
               fraction of positive interactions uniformly

networks are generated in the definitions.py file, where functions for generating matrices are defined

parameters are:
 - n_supplied: number of externally supplied resources (can go from 1 to the number of species)
 - n_consumed: number of secondary resources consumed by each species
 - n_produced: number of secondary resources produced by each species
 - structure:  are the matrix structured in favour of cooperation or are they random
 - leakage:    leakage regime can be high or low
 - replica:    we want multiple replicas for each parameter combination

OUTPUTS a pickle file with     data = {
                                        'n_supplied': n_supplied,
                                        'n_consumed':n_consumed,
                                        'n_produced':n_produced,
                                        'structure': structure,
                                        'leakage': leakage,
                                        'parameters':param,
                                        'replica':replica,
                                        'uptake':up_mat,
                                        'D':met_mat,
                                        'CR_R':R_fin[::200], # only save one every 10 time steps to make it lighter
                                        'CR_N':N_fin[::200]  # only save one every 10 time steps to make it lighter
                                    } 
        inside the dedicated Data folder

"""

import numpy as np
import pandas as pd

import os
import sys
import pickle

path = os.path.splitext(os.path.abspath(__file__))[0]
base_path = path.split('/CRIMMES/MODEL')[0]
module_path = f'{base_path}/CRIMMES/MODEL/shared_scripts'

# Add the directory to sys.path
if module_path not in sys.path:
    sys.path.append(module_path)

import R_dynamics 
import N_dynamics 
import definitions 
import update 
import SOR 
import well_mixed

# -------------------------------------------------------------------------------------------------------
# main function 

def protocol_1(n_supplied,n_consumed,n_produced,structure,leakage,replica):

    """
    n_supplied: int, supplied CS
    n_consumed: int, secondary CS consumed by each
    n_produced: int,secondary CS produced by each
    structure:  bool, structured or unstructured matrices
    leakage:    int, can be high (1) or low (2)
    replica:    int, replica number

    RETURNS     pkl file with simulation dynamics and mapping parameters

    """

    # create paths to save results, these are examples, ad they will end up in the Data folder
    path = f'{base_path}/CRIMMES/MODEL/Data/protocol_1'
    results_dir = f"{path}_results/{n_supplied}_{n_consumed}_{n_produced}_{structure}_{leakage}_{replica}"
    os.makedirs(results_dir, exist_ok=True)

    # set seed for random generations
    np.random.seed(replica)

    # number of species and number of nutrients
    n_s = 8            # 8 species
    n_r = 20           # maximum of 16 CS
    n_ext = n_supplied

    # --------------------------------------------------------------------------------------------------------

    # generate uptake and metabolic matrices
    if structure==1:
        up_mat  = definitions.structured_up_relaxed(n_s,n_r,n_supplied,n_consumed,n_produced,p=0.)
        met_mat = definitions.structured_met_relaxed(n_s,n_r,n_supplied,n_produced,p=0.)
    elif structure==2:
        up_mat  = definitions.structured_up_relaxed(n_s,n_r,n_supplied,n_consumed,n_produced,p=0.3)
        met_mat = definitions.structured_met_relaxed(n_s,n_r,n_supplied,n_produced,p=0.3)
    else:
        up_mat  = definitions.tradeoff_up(n_s,n_r,n_supplied,n_consumed)
        met_mat = definitions.simple_met(n_s,n_r,n_supplied,n_produced)

    # all nutrients apport positive contributions to gr
    sign_mat = np.ones((n_s,n_r))

    # no auxotrophies (anyone can produce what metabolism allows)
    spec_met = np.ones((n_s,n_r))
    mat_ess  = np.zeros((n_s,n_r))

    # recapitulate in dictionary
    mat = {
        'uptake'  : up_mat,
        'met'     : met_mat,
        'ess'     : mat_ess,
        'spec_met': spec_met,
        'sign'    : sign_mat
    }

    # ----------------------------------------------------------------------------------------------------------

    # set dilution
    tau = 100

    # growth and maintainence
    g = np.ones((n_s))
    m = np.zeros((n_s))+1/tau

    # reinsertion of chemicals
    tau = np.zeros((n_r))+tau 
    ext = np.zeros((n_r))
    # primary carbon sources replenished to saturation
    ext[:n_ext] = 100

    # initial guess for resources
    guess = np.ones((n_r))*100

    # leakage
    if leakage == 1:
        l = np.zeros((n_r))+0.8
    else:
        l = np.zeros((n_r))+0.2

    # energy conversion
    w = np.ones((n_r))

    # define parameters
    param = {
        'w'  : w,                                          # energy conversion          [energy/mass]
        'l'  : l,                                          # leakage                    [adim]
        'g'  : g,                                          # growth conv. factors       [1/energy]
        'm'  : m,                                          # maintainance requ.         [energy/time]
        'ext': ext,                                        # external replenishment     [units of K] 
        'tau' : tau,                                       # chemicals dilution         [1/time]                           
        'guess_wm': guess                                  # initial resources guess    [units of K]
    }

    # ----------------------------------------------------------------------------------------------------------

    # run CR model for 200000 steps 
    N_fin,R_fin=well_mixed.run_wellmixed(np.ones((n_s)),param,mat,well_mixed.dR_dt_nomod,well_mixed.dN_dt,100000)

    # save results
    data = {
        'n_supplied': n_supplied,
        'n_consumed':n_consumed,
        'n_produced':n_produced,
        'structure': structure,
        'leakage': leakage,
        'parameters':param,
        'replica':replica,
        'uptake':up_mat,
        'D':met_mat,
        'CR_R':R_fin[::200], # only save one every 10 time steps to make it lighter
        'CR_N':N_fin[::200]  # only save one every 10 time steps to make it lighter
    }

    # output file path
    output_file = f'{results_dir}/all_data.pkl'

    # save as pickle
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)

    return 

# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Check if there are command-line arguments
    if len(sys.argv) == 7:
        n_supplied = int(sys.argv[1])
        n_consumed = int(sys.argv[2])
        n_produced = int(sys.argv[3])
        structure  = int(sys.argv[4])
        leakage    = int(sys.argv[5])
        replica    = int(sys.argv[6])

        # Run the simulation with the provided parameters
        protocol_1(n_supplied,n_consumed,n_produced,structure,leakage,replica)
        print(f"Simulation completed for {n_supplied}_{n_consumed}_{n_produced}_{structure}_{leakage}_{replica}")

    else:
        print("Usage: protocol_1.py")



