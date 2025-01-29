"""
networks_generation.py: script to simulate CR network that map onto LV covering a span of connectivity and 
        fraction of positive interactions uniformly

networks are generated as described in the definitions.py file, where functions for generating matrices are defined

across all networks, the total energy supplied from the extern and the total energy that a single cell is able to
transform to biomass are held constant

parameters are:
 - n_supplied: number of externally supplied resources (can go from 1 to the number of species)
 - n_consumed: number of secondary resources consumed by each species
 - sparsity:   sparsity of the metabolic matrix
 - noise:      noise on a non-overlapping resources consumption structure
 - PCS_var:    variance on the PCS uptake rates
 - PCS_bias:   bias on PCS consumption with respect to SCS (=1 means no bias)
 - leakage:    leakage regime can be high or low
 - replica:    we want multiple replicas for each parameter combination

"""

from unittest import result
import mapping
import definitions
import well_mixed

import os
import sys
import pickle
import numpy as np

from scipy.integrate import solve_ivp

# -------------------------------------------------------------------------------------------------------
# main function 

def generate(n_supplied,n_consumed,sparsity,noise,PCS_var,PCS_bias,leakage,replica):

    """
    n_supplied: int, supplied CS
    n_consumed: int, secondary CS consumed by each
    sparsity:   float, sparsity of the metabolic matrix
    noise:      float, structured or unstructured matrices
    PCS_var:    float, variance on the PCS uptake rates
    PCS_bias:   float, bias on PCS consumption with respect to SCS
    leakage:    float, leakage

    RETURNS     saves pkl file with simulation dynamics and mapped parameters

    """

    # Create paths to save results
    path = os.path.splitext(os.path.abspath(__file__))[0]
    results_dir = f"{path}_results/{n_supplied}_{n_consumed}_{sparsity}_{noise}_{PCS_var}_{PCS_bias}_{leakage}_{replica}"

    # Check if the directory already exists and is not empty
    if os.path.exists(results_dir) and os.listdir(results_dir):
        print(f"Directory {results_dir} exists and is not empty. Skipping...")
        return

    os.makedirs(results_dir, exist_ok=True)
    

    # number of species and number of nutrients
    n_s = 8            # 8 species
    n_ext = n_supplied # input n_supplied
    n_r = 20 + n_s     # maximum of 20 secondary CS

    # --------------------------------------------------------------------------------------------------------

    # generate uptake and metabolic matrices
    up_mat  = definitions.complete_up(n_s, n_r, n_supplied, n_consumed, p=noise, bias=PCS_bias, PCS_var=PCS_var)
    met_mat = definitions.complete_met(n_s,n_r,n_supplied,sparsity)

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
    ext[:n_ext] = 100/n_supplied

    # initial guess for resources
    guess = np.ones((n_r))*100/n_supplied

    # leakage
    l = np.ones((n_r))*leakage

    # energy conversion
    w = np.ones((n_r))/n_consumed

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
    N_fin,R_fin=well_mixed.run_wellmixed(np.ones((n_s)),param,mat,well_mixed.dR_dt_nomod,well_mixed.dN_dt,200000)

    # calculate matrices for mapping
    grad_mat=mapping.grad_i_alpha(R_fin[-1,:],param,mat)
    sig=mapping.sigma(R_fin[-1,:],param)
    f_mat=mapping.f_i_alpha(R_fin[-1,:],param,mat)

    # intrinsic growth rates
    g_LV = np.zeros((n_s))
    for i in range(n_s):
        g_LV[i]=np.dot(grad_mat[i],sig)

    # interaction matrix
    A_int=np.zeros((n_s,n_s))
    for i in range(n_s):
        for j in range(n_s):
            A_int[i,j]=np.dot(grad_mat[i],f_mat[j])

    # Solve Lotka-Volterra dynamics with the calculated parameters
    lv_args = (g_LV,A_int,n_s)
    t_span_lv = (0,200000)
    t_eval_lv = np.arange(t_span_lv[0],t_span_lv[1],1)
    solLV = solve_ivp(fun=mapping.LV_model, t_span=t_span_lv, y0=np.ones((n_s)), t_eval=t_eval_lv, args=lv_args)

    # save results
    data = {
        'n_supplied': n_supplied,
        'n_consumed':n_consumed,
        'sparsity':sparsity,
        'noise': noise,
        'PCS_var':PCS_var,
        'PCS_bias':PCS_bias,
        'leakage': leakage,
        'parameters':param,
        'uptake':up_mat,
        'D':met_mat,
        'CR_R':R_fin[::500], # only save one every 10 time steps to make it lighter
        'CR_N':N_fin[::500], # only save one every 10 time steps to make it lighter
        'LV': solLV.y[:,::500],
        'g0':g_LV,
        'A':A_int
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
    if len(sys.argv) == 9:
        n_supplied = int(sys.argv[1])
        n_consumed = int(sys.argv[2])
        sparsity   = float(sys.argv[3])
        noise      = float(sys.argv[4])
        PCS_var    = float(sys.argv[5])
        PCS_bias   = float(sys.argv[6]) 
        leakage    = float(sys.argv[7]) 
        replica    = int(sys.argv[8])

        # Run the simulation with the provided parameters
        generate(n_supplied,n_consumed,sparsity,noise,PCS_var,PCS_bias,leakage,replica)
        print(f"Simulation completed for {n_supplied}_{n_consumed}_{sparsity}_{noise}_{PCS_var}_{PCS_bias}_{leakage}_{replica}")

    else:
        print("Usage: python networks_generation.py")



