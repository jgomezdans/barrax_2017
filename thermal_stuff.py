#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import prosail
def angular_effect(emv, ems, tveg_sun, tsoil_sun, tveg_shade, tsoil_shade,
                   t_atm, lai, lidf, hspot = 0.05, lam=9.5, sza=0., raa=0.):

    canopy_temp = []
    canopy_emissivity = []
    vaa = 0
    saa = 0
    for vza in np.linspace(0,90):
        
        r, tcan, emis = prosail.run_thermal_sail(lam,  
                     tveg_shade, tsoil_shade, tveg_sun, tsoil_sun, t_atm, 
                     lai, lidfa, hspot,  
                     sza, vza, raa, emv=emv, ems=ems)
        canopy_temp.append(tcan-273.15)
        canopy_emissivity.append(emis)
    l1 = plt.plot(np.linspace(-0,90), canopy_temp, label="Temperature")
    plt.ylabel("TOC temperature [degC]")
    plt.xlabel("View zenith angle [deg]")
    
    plt.twinx()
    l2 = plt.plot(np.linspace(-0,90), canopy_emiss, color="#66C2A5", label="Emissivity")
    plt.ylabel("Canopy emissivity")
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc="center left") 
