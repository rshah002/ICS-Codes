#!/usr/bin/python
#from __future__ import print_function
import numpy as np
#import sys

c = 2.998e8                # speed of light [m/s]
q = 1.602e-19              # electron charge [C]
#eV2J = 1.602176565e-19     # eV --> Joules 
J2eV = 6.242e18            # Joules --> eV
#hbar = 6.582e-16           # Planck's constant [eV*s]
hbar = 1.0545718e-34       # Planck's constant [m^2 kg/s]
me = 9.10938356e-31        # electron mass [kg]
alpha = 1.0/137.0          # fine structure constant
pi = np.pi

Q = 1e-11                  # [C]
l = 800e-9                 # [m]
E_e = 30.66e6
mc2 = 511e3
a0 = 0.19
s = 16.0

N = Q/q
gamma = E_e/mc2           
E_l = 2*pi*hbar*c/l         # [J]
E_g = 4*(gamma**2)*E_l      # [J]
E_g = E_g*J2eV              # [eV]
dN_dE = N*alpha*(pi**1.5)*(a0**2)*s/E_g

print('Laser energy = ', E_l*J2eV, ' eV')
print('E_gamma peak = ', E_g, ' eV')
print('dN/dE        = ', dN_dE, 'photons')
print('dN/dE 1e     = ', dN_dE/N, 'photons')
