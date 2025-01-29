"""
script for the sensitivity analysis preliminar to the obligate/facultative cross-feeding comparison
the goal is to obtain curves in the diversity-n_producers graph, at changing number of n_consumed, and 
across a range of conditions (i.e. parameter collections)

n_supplied is fixed to 1 and the supply regime is high
uptake is binary, energy content is ridcaled for changing n_consumed (in the F, for the O CF doesn't bring energy)

we vary: - leakage
         - PCS_var
         - PCS_bias

"""

import os
import sys
import pickle

import numpy as np

import definitions
import well_mixed
import mapping_F
import mapping_O

from scipy.integrate import solve_ivp

def parameters_sensitivity (n_consumed, n_producers, leakage, PCS_bias, PCS_var, replica):

    """
    n_consumed: int, secondary CS consumed by each
    n_produced: int,secondary CS produced by each
    leakage:    float, leakage parameter
    PCS_bias:   float, relative consumption of PCS compared to SCS
    PCS_var:    float, variance of the uptake rates of PCS
    replica:    int, replica number

    RETURNS     pkl file with simulation dynamics and mapping parameters

    """

    # create paths to save results
    path = os.path.splitext(os.path.abspath(__file__))[0]
    results_dir = f"{path}_results/{n_consumed}_{n_producers}_{leakage}_{PCS_bias}_{PCS_var}_{replica}"
    
    # Check if the directory already exists and is not empty
    if os.path.exists(results_dir) and os.listdir(results_dir):
        print(f"Directory {results_dir} exists and is not empty. Skipping...")
        return

    os.makedirs(results_dir, exist_ok=True)

    # set seed for random generations
    np.random.seed(replica)

    # number of species and number of nutrients
    n_s = 8            # 8 species
    n_r = 21           # maximum of 16 CS
    n_ext = 1          # one supplied CS everyone can consume

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

    # recapitulate in dictionary
    mat_F = {
        'uptake'  : up_mat_F,
        'met'     : met_mat_F,
        'ess'     : mat_ess_F,
        'spec_met': spec_met,
        'sign'    : sign_mat_F
    }

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
    param_F = {
        'w'  : np.ones((n_r))/(n_consumed+1),              # energy conversion     [energy/mass]
        'l'  : l_F,                                        # leakage               [adim]
        'g'  : g_F,                                        # growth conv. factors  [1/energy]
        'm'  : m,                                          # maintainance requ.    [energy/time]
        'ext': ext_F,                                      # external replenishment  
        'tau' : tau,                                       # chemicals dilution                                            
        'guess_wm': guess                                  # initial resources guess
    }
    # define parameters
    param_O = {
        'w'  : np.ones((n_r)),                             # energy conversion     [energy/mass]
        'l'  : l_O,                                        # leakage               [adim]
        'g'  : g_O,                                        # growth conv. factors  [1/energy]
        'm'  : m,                                          # maintainance requ.    [energy/time]
        'ext': ext_O,                                      # external replenishment  
        'tau' : tau,                                                                       
        'guess_wm': guess                                  # initial resources guess
    }

    initial_condition = np.random.normal(loc=1, scale=0.1, size=n_s)

    # run CR model for 200000 steps 
    N_fin_F,R_fin_F=well_mixed.run_wellmixed(initial_condition,param_F,mat_F,well_mixed.dR_dt_nomod,well_mixed.dN_dt,1000000)

    # calculate matrices for mapping
    grad_mat_F=mapping_F.grad_i_alpha(R_fin_F[-1,:],param_F,mat_F)
    sig_F=mapping_F.sigma(R_fin_F[-1,:],param_F)
    f_mat_F=mapping_F.f_i_alpha(R_fin_F[-1,:],param_F,mat_F)

    # intrinsic growth rates
    g_LV_F = np.zeros((n_s))
    for i in range(n_s):
        g_LV_F[i]=np.dot(grad_mat_F[i],sig_F)

    # interaction matrix
    A_int_F=np.zeros((n_s,n_s))
    for i in range(n_s):
        for j in range(n_s):
            A_int_F[i,j]=np.dot(grad_mat_F[i],f_mat_F[j])

    # Solve Lotka-Volterra dynamics with the calculated parameters
    lv_args = (g_LV_F,A_int_F,n_s)
    t_span_lv = (0,1000000)
    t_eval_lv = np.arange(t_span_lv[0],t_span_lv[1],10)
    solLV_F = solve_ivp(fun=mapping_F.LV_model, t_span=t_span_lv, y0=np.ones((n_s)), t_eval=t_eval_lv, args=lv_args)

    # ---------------------------------------------------------------------------------------------------------------------

    # run CR model for 200000 steps 
    N_fin_O,R_fin_O=well_mixed.run_wellmixed(initial_condition,param_O,mat_O,well_mixed.dR_dt_maslov,well_mixed.dN_dt_maslov,1000000)

    # calculate matrices for mapping
    grad_mat_O=mapping_O.grad_i_alpha(R_fin_O[-1,:],param_O,mat_O)
    sig_O=mapping_O.sigma(R_fin_O[-1,:],param_O)
    f_mat_O=mapping_O.f_i_alpha(R_fin_O[-1,:],param_O,mat_O)

    # intrinsic growth rates
    g_LV_O = np.zeros((n_s))
    for i in range(n_s):
        g_LV_O[i]=np.dot(grad_mat_O[i],sig_O)

    # interaction matrix
    A_int_O=np.zeros((n_s,n_s))
    for i in range(n_s):
        for j in range(n_s):
            A_int_O[i,j]=np.dot(grad_mat_O[i],f_mat_O[j])

    # Solve Lotka-Volterra dynamics with the calculated parameters
    lv_args = (g_LV_O,A_int_O,n_s)
    t_span_lv = (0,1000000)
    t_eval_lv = np.arange(t_span_lv[0],t_span_lv[1],10)
    solLV_O = solve_ivp(fun=mapping_O.LV_model, t_span=t_span_lv, y0=initial_condition, t_eval=t_eval_lv, args=lv_args)

    # save results
    data = {
        'n_producers':n_producers,
        'n_consumed':n_consumed,
        'replica':replica,
        'C_F':up_mat_F,
        'C_O':up_mat_O,
        'D_F':met_mat_F,
        'D_O':met_mat_O,
        'CR_R_F':R_fin_F[::200], # only save one every 10 time steps to make it lighter
        'CR_R_O':R_fin_O[::200], # only save one every 10 time steps to make it lighter
        'CR_N_F':N_fin_F[::200], # only save one every 10 time steps to make it lighter
        'CR_N_O':N_fin_O[::200], # only save one every 10 time steps to make it lighter
        'LV_F': solLV_F.y[:,::500],
        'LV_O': solLV_O.y[:,::500],
        'g0_F':g_LV_F,
        'A_F':A_int_F,
        'g0_O':g_LV_O,
        'A_O':A_int_O,
        'initial_condition':initial_condition
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
        n_consumed  = int(sys.argv[1])
        n_producers = int(sys.argv[2])
        leakage     = float(sys.argv[3])
        PCS_bias    = float(sys.argv[4])
        PCS_var     = float(sys.argv[5])
        replica     = int(sys.argv[6])

        # Run the simulation with the provided parameters
        parameters_sensitivity(n_consumed, n_producers, leakage, PCS_bias, PCS_var, replica)
        print(f"Simulation completed for {n_consumed}_{n_producers}_{leakage}_{PCS_bias}_{PCS_var}_{replica}")

    else:
        print("Usage: python parameters_sensitivity.py")