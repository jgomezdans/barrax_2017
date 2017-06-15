#!/usr/bin/env python
"""Some functions to invert simple RT models
"""
import os
import sys

import numpy as np
import prosail


def prospect_lklhood(x, rho, rho_unc, tau=None, tau_unc=None):
    """Calculates the log-likelihood of leaf reflectance measurements
    assuming Gaussian additive noise. Can either use reflectance or
    transmittance or both."""
    wv, rho_pred, tau_pred = prosail.run_prospect(x[0], x[1], x[2], 
            x[3], x[4], x[5], ant=x[6])
    rho_unc_ndim = rho_unc.ndim
    
    if rho_unc_ndim == 1:
        cov_obs_rho_inv = np.diag(1.0/(rho_unc*rho_unc))
    elif rho_unc_ndim == 2:
        cov_obs_rho_inv = np.linalg.inv(rho_unc)
    refl_lklhood = 0.5*(rho_pred - rho).dot(
                cov_obs_rho_inv.dot(rho_pred - rho))
    if tau is not None and tau_unc is not None:
        tau_unc_ndim = tau_unc.ndim
        if tau_unc_ndim == 1:
            cov_obs_tau_inv = np.diag(1.0/(tau_unc*tau_unc))
        elif tau_unc_ndim == 2:
            cov_obs_tau_inv = np.linalg.inv(tau_unc)
        else:
            raise ValueError("tau_unc has to be a vector or matrix")
            
        tau_lklhood = 0.5*(tau_pred - tau).dot(
            cov_obs_tau_inv.dot(tau_pred-tau))
    else:
        tau_lklhood = 0.0
    return refl_lklhood + tau_lklhood

