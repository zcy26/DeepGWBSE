#!/usr/bin/env python


# This script helps construct the slices/subspaces used in pseudobands.

# Because this implementation of the algorithm uses the N_S (total number
# of subspaces/slices) paramater instead of the constant energy fraction F,
# we need to compute the correct partition of the true mean-field bands
# into N_S subspaces while respecting the constant energy ratio of adjacent
# subspaces. To do so, we enforce an exponential ansatz on the energy spanned
# by the bands in each subspace, i.e. (energy spanned subspace S_j) = α * exp(j * β)
# for some fitting parameters α,β. By doing this, we can deduce F as being 
# roughly proportional to β. This script determines α and β.

# For explanation of the parameters N_S and F, see the paper about this method,
# Altman A. R., Kundu S., da Jornada F. H., "Mixed Stochastic-Deterministic
# Approach For Many-Body Perturbation Theory Calculations" (2023).


import numpy as np
from scipy.optimize import minimize_scalar, fsolve, Bounds
import warnings

# Constants based on 3D free electron gas density of states
C = 1 / (12 * np.pi**8) # atomic units
B = np.sqrt(2) / np.pi**2


def alpha(beta, E0, Emax, nspbps, nslice): 
    dE = Emax - E0
    return (dE * (np.exp(beta)-1)) / (np.exp(beta) * (np.exp(nslice * beta) - 1))


def w(beta, j, E0, Emax, nspbps, nslice): 
    return alpha(beta, E0, Emax, nspbps, nslice) * np.exp(j * beta)


def Ebar(beta, j, E0, Emax, nspbps, nslice): 
    a = alpha(beta, E0, Emax, nspbps, nslice)
    width = w(beta, j, E0, Emax, nspbps, nslice)
    return E0 + a * np.exp(beta) * (np.exp(j * beta) - 1) / (np.exp(beta) - 1) - width / 2


def Loss(beta, E0=1, Emax=10, nspbps=1, nslice=10):
    out = 0
    for j in range(1, nslice+1):
        Ei = Ebar(beta, j, E0, Emax, nspbps, nslice)
        wi = w(beta, j, E0, Emax, nspbps, nslice)

        assert Ei > 0, Ei
        assert wi > 0, wi
        
        # FEG approximation to dimension of the subspace breaks down, let dim = 1 in this case
        if B * wi * np.sqrt(Ei) < 1:
            out += C * wi**2 / Ei**2
            continue
            
        l = C * wi**2 / Ei**2 + 1 / (Ei**2 * nspbps) * (1 - 1 / (B * wi * np.sqrt(Ei)))
        out += l
        
    return out


def optimize(E0=1, Emax=10, nspbps=1, nslice=10):
    
    dE = Emax - E0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        min_dim = fsolve(lambda x: B * x * np.sqrt(E0 + x/2) - 1, 4)
        low = max(fsolve(lambda x: alpha(x, E0, Emax, nspbps, nslice)*np.exp(x) 
                     - max(min_dim), .2*np.ones(3), maxfev=1000))
        high = max(fsolve(lambda x: B * w(x, nslice, E0, Emax, nspbps, nslice) 
                      * np.sqrt(Ebar(x, nslice, E0, Emax, nspbps, nslice)) - 1,
                      .2*np.ones(3), maxfev=1000))
   
        tol=1e-9
        bnds = (tol, 1-tol)
        
        result = minimize_scalar(Loss, args=(E0, Emax, nspbps, nslice), bounds=bnds, method='bounded')
    
    print(f'minimization results: \n{result}')
    print(f'alpha: {alpha(result.x, E0, Emax, nspbps, nslice)}')
    
    return result
    
