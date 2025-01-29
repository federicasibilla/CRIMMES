"""

functions for mapping CR onto LV in the leakage regulated case, intended for mapping in cases where there are auxotrophs
          [Meacock, 2023]

CONTAINS: - f_i_alpha: function for the impact of species on resources, in a matrix
          - sigma: function for the external impact on resources, in a vector
          - grad_i_alpha: function for the gradient of the sensitivity function, in a matrix
          - LV_model: definition of the LV integration scheme

"""

import numpy as np

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# FUNCTIONS FOR THE MAPPING

# -------------------------------------------------------------------------------------------------------
# impact function of species i on resource alpha

def f_i_alpha(R,param,mat):

    """
    R: vector, n_r, contains concentrations
    param: dictionary, parameters
    mat: dictionary, matrices

    returns matrix, n_sxn_r, the impact function of species i on resource alpha in a matrix

    """

    n_s = param['g'].shape[0]
    n_r = R.shape[0]

    prod = np.zeros((n_s,n_r))
    
    # species specific metabolism and renormalization
    D_species = np.tile(mat['met'],(n_s,1,1))*np.transpose((np.tile(mat['spec_met'],(1,1,n_r)).reshape(n_s,n_r,n_r)),axes=(0,2,1))
    D_s_norma = np.zeros((n_s,n_r,n_r))
    for i in range(n_s):
        sums = np.sum(D_species[i], axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            D_s_norma[i] = np.where(sums != 0, D_species[i] / sums, D_species[i])

    for i in range(n_s):
        # calculate essential nutrients modulation for each species (context-dependent uptake)
        if (np.sum(mat['ess'][i]!=0)):
            mu  = np.min(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))
            lim = np.where(mat['ess'][i] == 1)[0][np.argmin(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))]
            l_eff=param['l'].copy()*mu + (1-mu)     # modulate uptakes 
            l_eff[lim]=param['l'].copy()[lim]
            l_eff[lim]=param['l'][lim].copy()     # restore uptake of the limiting one to max
            prod[i] = np.dot(mat['uptake'][i]*R/(1+R)*param['w']*l_eff,(D_s_norma[i].T))*1/param['w']
        else:
            prod[i] = np.dot(mat['uptake'][i]*R/(1+R)*param['w']*param['l'],(D_s_norma[i].T))*1/param['w']

    # resource loss due to uptake (not modulated by essentials)
    out = np.zeros((n_s,n_r))
    for i in range(n_s):
        out[i] = mat['uptake'][i]*R/(1+R)
        out[i][np.abs(out[i])<1e-14]=0
        
    fialpha = np.zeros((n_s,n_r))

    for i in range(n_s):
        for alpha in range(n_r):
            f_ia = -out[i,alpha]+prod[i,alpha]
            fialpha[i,alpha]+=f_ia

    return fialpha

# -------------------------------------------------------------------------------------------------------
# external replenishment function

def sigma(R,param):

    """
    R: vector, n_r, contains concentrations
    param: dictionary, parameters

    returns vector, n_r, replenishment function for each resource

    """
    sigma = (param['ext']-R)/param['tau']

    return sigma

# -------------------------------------------------------------------------------------------------------
# gradient of sensitivity function of species i on resource alpha

def grad_i_alpha(R,param,mat):

    """
    R: vector, n_r, contains concentrations
    param: dictionary, parameters
    mat: dictionary, matrices

    returns matrix, n_sxn_r, containing the alpha component of the gradient of the sensitivity function of i

    """

    n_s = len(param['g'])
    n_r = len(R)

    gialpha = np.zeros((n_s,n_r))

    for i in range(n_s):
        # find limiting nutrient, if any:
        if (np.sum(mat['ess'][i]!=0)):
            mu  = np.min(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))
            lim = np.where(mat['ess'][i] == 1)[0][np.argmin(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))]
            for alpha in range(n_r):
                g_ia = param['g'][i]*param['w'][alpha]*mat['uptake'][i,alpha]*mu*(1-param['l'][alpha])*1/(1+R[alpha])**2
                gialpha[i,alpha]=g_ia
            gialpha[i,lim]=param['g'][i]*(np.sum(param['w']*(1-param['l'])*mat['uptake'][i]*mat['sign'][i]*R/(1+R)))*1/(1+R[lim])**2
        else:    
            for alpha in range(n_r):
                g_ia = param['g'][i]*param['w'][alpha]*mat['uptake'][i,alpha]*(1-param['l'][alpha])*1/(1+R[alpha])**2
                gialpha[i,alpha]=g_ia

    return gialpha

# -------------------------------------------------------------------------------------------------------
# LV model definition

def LV_model(t,y,r0,A,n_s):

    """
    t: tima
    y: vector, n_s, initial composition
    r0: vector, n_s, intrinsic growth rates
    A: matrix, n_sxn_s, interaction coefficients
    n_s: int, number of species

    RETURNS matrix, n_s*time_steps, time seires of the species

    """

    sp_abund = y

    # Set abundances to zero if they are less than a threshold
    #sp_abund[sp_abund < 1e-10] = 0

    dsdt = [sp_abund[alpha]*(r0[alpha] + np.dot(A[alpha,:],sp_abund)) for alpha in range(n_s)]
    
    return dsdt