#!/usr/bin/python
from __future__ import print_function
import numpy as np
from math import atan, cos, sin
import sys

q = 1.602e-19              # electron charge [C]
Q = float(sys.argv[1])
filename = sys.argv[2]
N = Q/q

d  = np.loadtxt(filename)
d_omega = d[1,0]-d[0,0]
Npts = len(d[:,0])
dNdE = sum(d[:,2])
#print(d_omega, dNdE, dNdE*d_omega)
#print('Expected photon count = ', N*dNdE*d_omega)
print(N*dNdE*d_omega)
