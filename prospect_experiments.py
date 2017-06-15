#!/usr/bin/env python
"""Some functions to invert simple RT models
"""
import os
import sys

import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import prosail


def plot_spectra(rho_meas, rho_unc, tau_meas, tau_unc, 
                 rho_pred=None, tau_pred=None):
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=True)
        axs = axs.flatten()
        wv = np.arange(400, 2501)
        axs[0].plot(wv, rho_meas, '-', label=r'$\rho$')
        axs[0].fill_between(wv, rho_meas - 1.96*rho_unc,
                            rho_meas + 1.96*rho_unc,
                            color="0.8")
        if rho_pred is not None:
            axs[0].plot(wv, rho_pred, '-', label=r'$\rho$ pred')
        axs[1].plot(wv, tau_meas, '-', label=r'$\tau$')
        axs[1].fill_between(wv, tau_meas - 1.96*tau_unc,
                            tau_meas + 1.96*tau_unc,
                            color="0.8")
        if rho_pred is not None:
            axs[1].plot(wv, tau_pred, '-', label=r'$\tau$ pred')

        axs[0].legend(loc="best")
        axs[1].legend(loc="best")
        make_pretty(axs[0])
        make_pretty(axs[1])
        axs[1].set_xlabel("Wavelength [nm]")

def make_pretty(axs):

#    axs.xaxis.set_ticks_position("left")
#    axs.yaxis.set_ticks_position("bottom")
    
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)    
 
 
def read_lopex_sampe(sample_id, do_plot=True):
    
    if sample_id < 1 or sample_id > 116:
        raise ValueError("Only sample numbers between 1 and 116")
    
    rho = np.loadtxt("data/LOPEX93/refl.%03d.dat" % 
                     sample_id).reshape((5, 2101))
    tau = np.loadtxt("data/LOPEX93/trans.%03d.dat" % 
                     sample_id).reshape((5, 2101))
    if do_plot:
        plot_spectra(rho.mean(axis=0), rho.std(axis=0), 
                     tau.mean(axis=0), tau.std(axis=0))
    return rho.mean(axis=0), rho.std(axis=0), tau.mean(axis=0), tau.std(axis=0)

def prospect_prior ( x, mu, inv_cov):
    dif = x-mu
    prior = (x-mu).dot((inv_cov.dot(dif)))
    return 0.5*prior

def prospect_lklhood(x, rho, rho_unc, tau=None, tau_unc=None):
    """Calculates the log-likelihood of leaf reflectance measurements
    assuming Gaussian additive noise. Can either use reflectance or
    transmittance or both."""
    wv, rho_pred, tau_pred = prosail.run_prospect(x[0], x[1], x[2], 
            x[3], x[4], x[5], ant=x[6])
    rho_unc_ndim = rho_unc.ndim
    
    if rho_unc_ndim == 1:
        cov_obs_rho_inv = 1.0/(rho_unc*rho_unc)
    elif rho_unc_ndim == 2:
        cov_obs_rho_inv = 1./rho_unc.diagonal()
    refl_lklhood = np.sum(0.5*cov_obs_rho_inv*(rho_pred - rho)**2)
    if tau is not None and tau_unc is not None:
        tau_unc_ndim = tau_unc.ndim
        if tau_unc_ndim == 1:
            cov_obs_tau_inv = 1.0/(tau_unc*tau_unc)
        elif tau_unc_ndim == 2:
            cov_obs_tau_inv = 1.0/tau_unc.diagonal()
        else:
            raise ValueError("tau_unc has to be a vector or matrix")
            
        tau_lklhood = np.sum(0.5*cov_obs_tau_inv*(tau_pred - tau)**2)
    else:
        tau_lklhood = 0.0
    return refl_lklhood + tau_lklhood

def max_lklhood(x0, rho, rho_unc, tau, tau_unc, do_trans=True, do_plot=True):
    
    bounds = [ [1.1, 3],
               [1., 100],
               [1., 30],
               [0., 0.6],
               [0.00005, 0.1],
               [ 0.001, 0.03],
               [0., 20]]
    

    opts = {'maxiter': 500,
            'disp': True}
    if do_trans:
        def cost(x):
            return prospect_lklhood(x, rho, rho_unc, tau=tau, tau_unc=tau_unc)
            
    else:
        def cost(x):
            return prospect_lklhood(x, rho, rho_unc)
    retval = scipy.optimize.minimize(cost, x0, method="L-BFGS-B", jac=False, 
                                     bounds=bounds, options=opts)
    if do_plot:
        x = retval.x
        wv, rho_pred, tau_pred = prosail.run_prospect(x[0], x[1], x[2], 
            x[3], x[4], x[5], ant=x[6])
        plot_spectra(rho, rho_unc, tau, tau_unc, rho_pred=rho_pred,
                     tau_pred=tau_pred)
    return retval

if __name__ == "__main__":
    x = np.array([2.5, 20., 0., 0.1, 0.015, 0.01, 1.])
    x += x*0.1
    wv, rho_meas, tau_meas = prosail.run_prospect(x[0], x[1], x[2], 
            x[3], x[4], x[5], ant=x[6])
    rho_unc = np.eye(2101)
    tau_unc = np.eye(2101)
    
    retval = max_lklhood(x, rho_meas, rho_unc, tau=tau_meas, tau_unc=tau_unc)
    print retval.x
    #assert np.allclose( retval.x, np.array([  2.76297742e+00,   2.19876369e+01,   1.00000000e+00,
    #     1.11741590e-01,   1.65242763e-02,   1.11853187e-02,
    #     6.44081421e-01]))    