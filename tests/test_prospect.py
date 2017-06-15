import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
import numpy as np
import prosail
from pytest import fixture, raises

from prospect_experiments import prospect_lklhood


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
