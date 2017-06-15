import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
import numpy as np
import prosail
from pytest import fixture, raises

from prospect_experiments import prospect_lklhood
from prospect_experiments import max_lklhood
from prospect_experiments import calculate_prior


def test_prospect_refl_vector_lklhood():
    x = np.array([2.1, 40., 10., 0.1, 0.015, 0.009, 1.])
    wv, rho_meas, tau_means =  prosail.run_prospect(x[0], x[1], x[2], 
            x[3], x[4], x[5], ant=x[6])
    rho_unc = np.ones(2101)*1.
    test_lklhood = prospect_lklhood(x, rho_meas, rho_unc)
    assert test_lklhood == 0.
    

def test_prospect_refl_matrix_lklhood():
    x = np.array([2.1, 40., 10., 0.1, 0.015, 0.009, 1.])
    wv, rho_meas, tau_means = prosail.run_prospect(x[0], x[1], x[2], 
            x[3], x[4], x[5], ant=x[6])
    rho_unc = np.eye(2101)
    test_lklhood = prospect_lklhood(x, rho_meas, rho_unc)
    assert test_lklhood == 0.
    
def test_prospect_refl_scalar_lklhood():
    x = np.array([2.1, 40., 10., 0.1, 0.015, 0.009, 1.])
    wv, rho_meas, tau_means = prosail.run_prospect(x[0], x[1], x[2], 
            x[3], x[4], x[5], ant=x[6])
    rho_unc = 1
    with raises(AttributeError):
        test_lklhood = prospect_lklhood(x, rho_meas, rho_unc)
        assert "Scalar uncertainty"
        
def test_prospect_trans_vector_lklhood():
    x = np.array([2.1, 40., 10., 0.1, 0.015, 0.009, 1.])
    wv, rho_meas, tau_meas =  prosail.run_prospect(x[0], x[1], x[2], 
            x[3], x[4], x[5], ant=x[6])
    rho_unc = np.ones(2101)*1.
    tau_unc = np.ones(2101)*1.
    test_lklhood = prospect_lklhood(x, rho_meas, rho_unc, tau=tau_meas, 
                                    tau_unc=tau_unc)
    assert test_lklhood == 0.
    

def test_prospect_trans_matrix_lklhood():
    x = np.array([2.1, 40., 10., 0.1, 0.015, 0.009, 1.])
    wv, rho_meas, tau_meas = prosail.run_prospect(x[0], x[1], x[2], 
            x[3], x[4], x[5], ant=x[6])
    rho_unc = np.eye(2101)
    tau_unc = np.eye(2101)

    test_lklhood = prospect_lklhood(x, rho_meas, rho_unc, tau=tau_meas, 
                                    tau_unc=tau_unc)
    assert test_lklhood == 0.
    
def test_prospect_trans_scalar_lklhood():
    x = np.array([2.1, 40., 10., 0.1, 0.015, 0.009, 1.])
    wv, rho_meas, tau_meas = prosail.run_prospect(x[0], x[1], x[2], 
            x[3], x[4], x[5], ant=x[6])
    rho_unc = np.eye(2101)
    tau_unc = 1.
    with raises(AttributeError):
        test_lklhood = prospect_lklhood(x, rho_meas, rho_unc, tau=tau_meas, 
                                    tau_unc=tau_unc)
        assert "Scalar uncertainty"

def test_max_lklhood_refl():
    x = np.array([2.5, 20., 0., 0.1, 0.015, 0.01, 1.])
    x += x*0.1
    wv, rho_meas, tau_meas = prosail.run_prospect(x[0], x[1], x[2], 
            x[3], x[4], x[5], ant=x[6])
    rho_unc = np.eye(2101)
    tau_unc = np.eye(2101)
    
    retval = max_lklhood(x, rho_meas, rho_unc, tau=None, tau_unc=None,
                         do_plot=False)
    assert np.allclose( retval.x, np.array([  2.76297742e+00,   
                                            2.19876369e+01,   1.00000000e+00,
                                            1.11741590e-01,   1.65242763e-02,
                                            1.11853187e-02,   6.44081421e-01]))
    
def test_max_lklhood_tau():
    x = np.array([2.5, 20., 0., 0.1, 0.015, 0.01, 1.])
    x += x*0.1
    wv, rho_meas, tau_meas = prosail.run_prospect(x[0], x[1], x[2], 
            x[3], x[4], x[5], ant=x[6])
    rho_unc = np.eye(2101)
    tau_unc = np.eye(2101)
    
    retval = max_lklhood(x, rho_meas, rho_unc, tau=tau_meas, tau_unc=tau_unc,
                         do_plot=False)
    assert np.allclose( retval.x, np.array([  2.75101612e+00,   2.19855356e+01,
                                            1.00000000e+00,   1.07894421e-01,
                                            1.64960121e-02,   1.10114465e-02,
                                            6.55246141e-01]))
    
    
def test_prior():
    x = np.array([2.5, 20., 0., 0.1, 0.015, 0.01, 1.])
    mu = np.array([2.0, 40., 10., 0.1, 0.015, 0.01, 1.])
    cov = np.eye(7)
    
    lprior = calculate_prior(x, mu, cov)
    assert np.allclose( lprior, 0.5*np.sum((x-mu)**2))
    
