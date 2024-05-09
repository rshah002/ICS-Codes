#==============================================================================
# XSENSE (eXtended Simulation of Emitted Nonlinear Scattering Events) code
#
# Computes inverse Compton scattering spectrum in the non-linear regime.
# Described in detail in
# - Terzic et al. 2019, EPL 126, 12003
# - Terzic et al. 2021
# Based on work reported in:
# - Krafft 2004, PRL 92, 204802
# - Terzic, Deitrick, Hofler & Krafft 2016, PRL 112, 074801
#   
# INPUT:  config.in (parameters for e-beam, laser, aperture and simulation)
# OUTPUT: output_SENSE.txt
#   Column 1: E' [eV]
#   Column 2: omega/omega'
#   Column 3: dN/dE' [1/eV] per electron in the e-beam
#             (to get the total number of scattered photons, multiply this 
#              column by the total number of electrons in the beam: Q/q, 
#              where Q is total charge and q the charge of an electron)
#   Column 4: dN/dE' normalized to 1
#   
#==============================================================================
from __future__ import print_function
import multiprocessing
import math
import sys
import os
import pyximport
from scipy.optimize import curve_fit

pyximport.install(setup_args={"script_args" : ["--verbose"]})
import random
import numpy as np
import XSENSE_module
import multiprocessing as mp
from math import pi,e,cos,sin,sqrt,atan,asin,exp,log
from scipy.interpolate import interp1d  #Beth 6/23/2020
import scipy.integrate as integrate
from scipy.stats import poisson
from scipy.special import lambertw

#---------------------
# Physical constants
#---------------------
mc2 = 511e3                # electron mass [eV]
c = 2.998e8                # speed of light [m/s]
r = 2.818e-15              # electron radius [m]
q = 1.602e-19              # electron charge [C]
eV2J = 1.602176565e-19     # eV --> Joules 
J2eV = 6.242e18            # Joules --> eV
hbar = 6.582e-16           # Planck's constant [eV*s]
hbarJ = 1.0545718e-34      # Planck's constant [J*s]
me = 9.10938356e-31        # electron mass [kg]
alpha = 1.0/137.0          # fine structure constant
eps0 = 8.85e-12            # epsilon 0 (C^2/Nm^2)

#-----------------------------------------------------------------
# Read in initial conditions from an external file 'data_ICs.in'. 
# Format:
# px [kg m/s], py [kg m/s], pz [kg m/s], gamma, x [m], y [m]
#-----------------------------------------------------------------
def readICs(fileICs):
    pxv = []
    pyv = []
    pzv = []
    gav = []
    xv = []
    yv = []
    d = np.loadtxt(fileICs)
    Np = len(d[:,0])
    for n in range(0,Np):
        pxv.append(d[n,0])
        pyv.append(d[n,1])
        pzv.append(d[n,2])
        gav.append(d[n,3])
        xv.append(d[n,4])
        yv.append(d[n,5])
    return pxv, pyv, pzv, gav, xv, yv, Np

#----------------------------------------------------------------------
# Create initial conditions by random sampling from e-beam parameters.
# Gaussian distribution is used with rms values specified in config.in.
#----------------------------------------------------------------------
def sample_df(Npart,sigma_e_x,sigma_e_y,sigma_e_px,sigma_e_py,gamma,sig_g,pmag0,sigmaPmag,sigma_p_x,sigma_p_y,a0,lm,sign,lambda0,iTypeEnv):
    random.seed(1)      # for reproducability of the results, fix the same seed
    Npart_all = 0
    pxv = []
    pyv = []
    pzv = []
    gav = []
    for i in range(0,Npart):
        px = pmag0*sigma_e_px*random.gauss(0,1)
        py = pmag0*sigma_e_py*random.gauss(0,1)
        pz = sqrt((pmag0+sigmaPmag*random.gauss(0,1))**2-(px**2)-(py**2))
        ga = sqrt((c**2)*(px**2 + py**2 + pz**2) + (mc2*eV2J)**2)/(mc2*eV2J)
        pxv.append(px)
        pyv.append(py)
        pzv.append(pz)
        gav.append(ga)

    xv = []
    yv = []
    i = 0
    while (i < Npart):
        x = sigma_e_x*random.gauss(0,1)
        y = sigma_e_y*random.gauss(0,1)
        i += 1
        xv.append(x)
        yv.append(y)

    #number of emissions added from SENSE+RR
    sigma_p = sqrt(sigma_p_x**2 + sigma_p_y**2)
    n_g = []
    n_s = []
    for i in range(0, Npart):
        sig_z = sign * lambda0
        pxvt = pxv[i] * lm
        pyvt = pyv[i] * lm
        pzvt = pzv[i] * lm
        delt = sqrt((pxv[i] ** 2 + pyv[i] ** 2)) / pzv[i]
        prat = (pzv[i] / ga) / sqrt(pxv[i] ** 2 + pyv[i] ** 2 + (pzv[i] / gav[i]) ** 2)
        prat_t = prat * sqrt(((pxvt * sig_z) / (prat * sigma_p)) ** 2
                             + ((pyvt * sig_z) / (prat * sigma_p)) ** 2 + 1.0)
        x0 = px / pz
        y0 = py / pz
        sig_z_v = sqrt(1.0 / ((pxvt ** 2) / ((sigma_p ** 2))
                              + (pyvt ** 2) / ((sigma_p ** 2)) + 1.0 / ((sign ** 2))))
        eta_v = (sig_z_v ** 2) * ((pxvt * xv[i]) / (sigma_p ** 2) + (pyvt * yv[i]) / (sigma_p ** 2))
        a0_v = a0 * exp(-((xv[i] ** 2) / (2 * (sigma_p ** 2))) - ((yv[i] ** 2) / (2 * (sigma_p ** 2)))) * exp(
            (eta_v ** 2) / (2 * (sig_z_v ** 2)))

        #n_gamma = 2 * alpha * sign * (a0_v ** 2) * sqrt(pi) / 3.0
        if (iTypeEnv == 2):  # rectangular envelope
            n_gamma = 2 * alpha * sign * (a0 ** 2) * pi / 3.0
        else:  # gaussian envelope
            n_gamma = 2 * alpha * sign * (a0 ** 2) * sqrt(pi) / 3.0

        print("n_gamma: \t",n_gamma)
        n_scatter = poisson.rvs(n_gamma, size=1)[0] + 1
        n_g.append(n_gamma)
        n_s.append(n_scatter)
        #n_s.append(int(n_gamma) + 1)
        print("n_s: \t",n_s)
        Npart_all = Npart_all + n_scatter

    return pxv, pyv, pzv, gav, xv, yv, n_g, n_s, Npart_all

#----------------------------------------------------------
# Sample the circular integration region with N_MC points.
#----------------------------------------------------------
def sample_circle(Npts,Rmax):
    fang = open('angles_SENSE.txt', 'w')
    radius = []
    phi = []
    for i in range(Npts):
        t = 2*np.pi*random.random()
        u = random.random()+random.random()
        if u > 1:
           r = 2-u
        else:
           r = u
        radius.append(Rmax*r)
        phi.append(t)
        xv = radius[i]*cos(phi[i])
        yv = radius[i]*sin(phi[i])
        line = r'%15.8e  %15.8e  %15.8e  %15.8e' % (radius[i],phi[i],xv,yv)
        fang.write(line)
        fang.write('\n')
    fang.close()
    return radius, phi

#-------------------------------------------------------------
# Sample the rectangular integration region with N_MC points.
#-------------------------------------------------------------
def sample_rectangle(Npts,L_aper,x_aper,y_aper):
    fang = open('angles_SENSE.txt', 'w')
    radius = []
    phi = []
    for i in range(Npts):
        x = -0.5*x_aper + x_aper*random.random()
        y = -0.5*y_aper + y_aper*random.random()
        radius.append(atan(sqrt(x**2+y**2)/L_aper))
        if (y > 0.0):
            if (x > 0.0):                       # quadrant 1
                phi.append(atan(y/x))
            else:                               # quadrant 2
                phi.append(pi - atan(y/x))
        else:
            if (x < 0.0):                       # quadrant 3
                phi.append(pi + atan(abs(y/x)))
            else:                               # quadrant 4
                phi.append(2*pi - atan(abs(y/x)))
        line = r'%15.8e  %15.8e  %15.8e  %15.8e' % (radius[i],phi[i],x,y)
        fang.write(line)
        fang.write('\n')
    fang.close()
    return radius, phi

#------------------------------------------------------------------------------
# Callback function to log the results during apply_async parallel computation.
#------------------------------------------------------------------------------
def logResult(result):
    results.append(result)


def totalCrit(As, e_crits, min, max):
    tot1 = 0
    tot2 = 0
    for i in range(len(As)):
        tot1 += (As[i]) * e_crits[i] * math.exp(-min / e_crits[i])
        tot2 += ((As[i]) / e_crits[i]) * math.exp(- max / e_crits[i])
    lwArg = ((min-max) * math.sqrt(tot2))/(2 * math.sqrt(tot1))
    pwrlog = lambertw(lwArg).real
    A = (2 * tot1 * math.exp(((2 * min)/(min-max)) * pwrlog) * pwrlog) / (min - max)
    e_tot = (min - max) / (2 * pwrlog)
    return A, e_tot


def logfunc(xplot, spectrum):
    xplotChunk = []
    logval=[]
    for i in range(len(spectrum)):
        if (spectrum[i] != 0) :
            logval.append(log(spectrum[i]))
            xplotChunk.append(xplot[i])
    return xplotChunk, logval


# ------------------------------------------------------------------------------
# Callback function to log the results during apply_async parallel computation.
# ------------------------------------------------------------------------------
def process_one(xplot, oneSpectrum, gamma, a0, lambda0, ns, Nout, mode):
    manySpectrum = [0 for x in range(Nout)]
    for j in range(Nout):
        manySpectrum[j] = oneSpectrum[j]  # sum of all the spectra

    file = open('xplot_oneSpectrum.txt', 'w')
    for i in range(len(xplot)):
        line = r'%15.8e  %15.8e' % (xplot[i], oneSpectrum[i])
        file.write(line)
        file.write('\n')
    file.close()

    ga = gamma
    f = interp1d(xplot, oneSpectrum, kind='quadratic', bounds_error=False, fill_value=0.0)

    # Recoil energy model 1: energy of the spectral density peak
    index = np.argmax(oneSpectrum)
    print("Radiated energy at the spectral peak: ", xplot[index])
    # Recoil energy model 2: mean radiated energy
    xbar = 0.0
    norm = 0.0
    for i in range(Nout):
        xbar = xbar + xplot[i] * oneSpectrum[i]
        norm = norm + oneSpectrum[i]
    print("Mean radiated energy                : ", (xbar / norm))
    e_crit = hbar * 3 * (gamma ** 2) * 2 * pi * (c / lambda0) * a0

    e_crits = []
    As = []
    xplot_i, logSpec = logfunc(xplot, oneSpectrum)

    def critfunc(x, a):
        # f = (-x_i/e_crit) - (2/3)*log(x_i) + a
        return [(-x_i / e_crit) + (1 / 3) * log(x_i) + a for x_i in x]

    cutoff = int(len(xplot) / 3)
    xplot_i = xplot_i[cutoff:]
    logSpec_i = logSpec[cutoff:]
    fitting = curve_fit(critfunc, xplot_i, logSpec_i)
    logA = fitting[0][0]
    As.append(exp(logA))
    e_crits.append(e_crit)

    if mode == 1:
        hit = False
        while not hit:
            En = (random.random() * (xplot[-1] - xplot[0])) + xplot[0]
            prob = random.random() * max(oneSpectrum)
            if prob <= f(En): hit = True
        E_rad = En
    else:
        E_rad = e_crit / 3

    E_rad0 = E_rad

    radfile = open('E_rads.txt', 'w')
    for i in range(ns - 1):
        line = r'%15.8e  %15.8e' % (i, E_rad)
        radfile.write(line)
        radfile.write('\n')
        ga = ga - E_rad / mc2
        shift = (ga / gamma) ** 2
        nextSpectrum = [0 for x in range(Nout)]
        for j in range(Nout):
            xint = xplot[j] / shift
            if (xint < xplot[0]) or (xint > xplot[Nout - 1]):
                nextSpectrum[j] = 0.0
            else:
                nextSpectrum[j] = (f(xint) * shift)
            manySpectrum[j] = manySpectrum[j] + nextSpectrum[j]
        if mode == 1:
            hit = False
            while not hit:
                En = (random.random() * (xplot[-1] - xplot[0])) + xplot[0]
                prob = random.random() * max(nextSpectrum)
                if prob <= (f(En / shift) * shift): hit = True
            E_rad = En
        else:
            E_rad = E_rad0 * shift
        e_crits.append(E_rad * 3)
        As.append(As[0] * (shift ** (2 / 3)))
    Eloss = (gamma - ga) * mc2
    A_tot, e_crit_tot = totalCrit(As, e_crits, xplot[cutoff], xplot[2*cutoff])

    return manySpectrum, Eloss, A_tot, e_crit_tot


#*****************
# MAIN PROGRAM
#*****************
if __name__ == '__main__':
    #---------------------------------------------------------------------
    # Read in parameters of the simulation from the input file config.in.
    #---------------------------------------------------------------------
    arg_file = open("config.in", "r")
    args = []
    for line in arg_file:
        i = 0
        while (line[i:i + 1] != " "):
            i += 1
        num = float(line[0:i])
        args.append(num)

    #-------------------
    # e-beam parameters
    #-------------------
    En0 = args[0]              # e-beam: mean energy [eV]
    sig_g = args[1]            # e-beam: relative energy spread
    sigma_e_x = args[2]        # e-beam: rms horizontal size [m]
    sigma_e_y = args[3]        # e-beam: rms vertical size [m]
    eps_x_n = args[4]          # e-beam: normalized horizontal emittance [m rad]
    eps_y_n = args[5]          # e-beam: normalized vertical emittance [m rad]

    #------------------
    # Laser parameters
    #------------------
    lambda0 = args[6]          # laser beam: wavelength [m]
    sign = args[7]             # laser beam: normalized sigma [ ]
    sigma_p_x = args[8]        # laser beam: horizontal laser waist size [m]
    sigma_p_y = args[9]        # laser beam: vertical laser waitt size [m]
    a0 = args[10]              # laser beam: field strength a0 []
    iTypeEnv = int(args[11])   # laser beam: laser envelope type
    if (iTypeEnv == 3):        # laser beam: load experimental data -Beth
        data_file = open("Laser_Envelope_Data.txt", "r")
        exp_data = []
        for line in data_file:
           exp_data.append(line.strip().split())
        exp_xi = []
        exp_a = []
        for line in exp_data:
            exp_xi.append(float(line[0]))
            exp_a.append(float(line[1]))
        exp_f=interp1d(exp_xi,exp_a,kind='cubic')       # laser beam: generate beam envelope function
    else:
        exp_xi = []
        exp_f = 0
    modType = int(args[12])    # laser beam: frequency modulation type
    fmd_xi=[]
    fmd_f=[]
    fmdfunc=0
    if (modType == 1):         # exact 1D chirp: TDHK 2014 (f(0)=1)
        a0chirp = a0                # laser beam: a0 chirping value
        fm_param = 0.0
    elif (modType == 2):       # exact 1D chirp: Seipt et al. 2015 (f(+/-inf)=1)
        a0chirp = a0                # laser beam: a0 chirping value
        fm_param = 0.0
    elif (modType == 3):       # RF quadratic chirping
        a0chirp = 0.0
        fm_param = float(args[13])   # laser beam: lambda_RF chirping value
    elif (modType == 4):       # RF sinusoidal chirping
        a0chirp = 0.0
        fm_param = float(args[13])   # laser beam: lambda_RF chirping value
    elif (modType == 5):       # exact 3D chirp: Maroli et al. 2018
        a0chirp = a0                 # laser beam: a0 chirping value
        fm_param = float(args[13])   # p parameter
    elif (modType == 6):       # chirp with ang. dep. (f(0)=1)
        a0chirp = a0                 # laser beam: a0 chirping value
        fm_param = float(args[13])   # theta_FM (optimization angle)
    elif (modType == 7):       # chirp with ang. dep. (f(+/-inf) = 1)
        a0chirp = a0                 # laser beam: a0 chirping value
        fm_param = float(args[13])   # theta_FM (optimization angle)
    elif (modType == 8):       # saw-tooth chirp
        a0chirp = a0                   # laser beam: a0 chirping value
        fm_param = float(args[13])     # chirping slope
    elif (modType == 9):       # read chirping data from a file and generate function -Beth
        data_file = open("Fmod_Data.txt", "r")
        fmod_data = []
        for line in data_file:
           fmod_data.append(line.strip().split())
        fmd_xi = []
        fmd_f = []
        for line in fmod_data:
            fmd_xi.append(float(line[0]))
            fmd_f.append(float(line[1]))
        fmdfunc=interp1d(fmd_xi,fmd_f,kind='cubic')
        a0chirp =  float(args[13])
        lambda_RF = 0.0
    elif (modType == 10):       # chirping from GSU 2013
        a0chirp = a0           
        fm_param = float(args[13])
    else:                      # no chirping
        a0chirp = 0.0
        fm_param = 0.0
    l_angle = args[14]         # laser beam: angle between laser & z-axis [rad]

    #---------------------
    # Aperture parameters
    #---------------------
    TypeAp = args[15]          # aperture: type: =0 circular; =1 rectangular
    L_aper = args[16]          # aperture: distance from IP to aperture [m]
    if (TypeAp == 0):
        R_aper = args[17]      # aperture: physical radius of the aperture [m]
        tmp = args[18]
        theta_max = atan(R_aper/L_aper)
    else:
        x_aper = args[17]
        y_aper = args[18]
 
    #-----------------------
    # Simulation parameters
    #-----------------------
    wtilde_min = args[19]      # simulation: start spectrum [norm. w/w0 units]
    wtilde_max = args[20]      # simulation: end spectrum [norm. w/w0 units]
    Nout = int(args[21])       # simulation: number of points in the spectrum
    Ntot = int(args[22])       # simulation: resolution of the inner intergral
    Npart = int(args[23])      # simulation: number of electron simulated
    N_MC = int(args[24])       # simulation: number of MC samples
    iFile = int(args[25])      # simulation: =1: read ICs from file; <> 1: MC
    iCompton = int(args[26])   # simulation: =1: Compton; <>1: Thomson
    RRmode = int(args[27])     # Radiation reaction model

    #--------------------------------------------
    # Compute basic parameters for the two beams
    #--------------------------------------------
    gamma = En0/mc2
    beta = sqrt(1.0-1.0/(gamma**2))
    c1 = gamma*(1.0 + beta)
    omega0 = 2.0*pi*c1*c1*(c/lambda0)
    ntothm1 = Ntot/2.0 - 1.0
    d_omega = (wtilde_max - wtilde_min)/(Nout-1)

    eps_x = eps_x_n/gamma
    eps_y = eps_y_n/gamma
    sigma_e_px = eps_x/sigma_e_x
    sigma_e_py = eps_y/sigma_e_y
    pmag0 = (eV2J/c)*sqrt(En0**2-mc2**2)
    sigmaPmag = (eV2J/c)*sqrt(((En0*(1.0+sig_g))**2)-mc2**2)-pmag0
    E_laser = 2*pi*hbar*c/lambda0

    #------------------------------------------------------------
    # Adjust for the fact that I ~ exp(-x^2/(2 sig_x^2)), but
    # I ~ a^2  -->  a ~ exp(-x^2/(4 sig_x^2))
    #------------------------------------------------------------
    if (iTypeEnv == 1):
        sign = sqrt(2) * sign
        sigma_p_x = sqrt(2) * sigma_p_x
        sigma_p_y = sqrt(2) * sigma_p_y

    #---------------------------------------------------------------------------
    # Compute the number of expected recoils n_gamma and the recoil parameter X
    # SENSE is valid for n_gamma < 1.
    # In the Compton regime, X is substantial (X ~ 1), and SENSE is not valid .
    #---------------------------------------------------------------------------
    if (iTypeEnv == 2):  # rectangular envelope
        n_gamma = 2*alpha*sign*(a0**2)*pi/3.0
    else:                # gaussian envelope
        n_gamma = 2*alpha*sign*(a0**2)*sqrt(pi)/3.0
    Xs = 4.0*E_laser*gamma/(me*(c**2)*J2eV)

    #--------------------------------
    # Normalization of the spectrum
    #--------------------------------
    om0 = (((1+beta)*gamma)**2)*(2*pi*c/lambda0)   # [1/s]
    yfac = (q**2)/(64.0*(pi**3)*eps0*(c**3))       # [J s^3/m^2]
    yfac = yfac/hbarJ
    yfac = yfac/float(N_MC)                        # [(J s)*(eV/J)/(eV s)] = []
    if (TypeAp == 0):                     # circular aperture
        yfac = pi*(theta_max**2)*yfac              # d Omega <dE/domega dOmega> 
    else:                                 # rectangular aperture
        yfac = (x_aper*y_aper)*yfac                # d Omega <dE/domega dOmega> 

    #-------------------------------------------------------
    # Create or read in initial distribution for the e-beam
    #-------------------------------------------------------
    if (iFile != 1):   # randomly generated from input parameters
        lm = lambda0 / (me * c)
        pxv, pyv, pzv, gav, xv, yv, n_g, n_s, Npart_all = sample_df(Npart,sigma_e_x,sigma_e_y,sigma_e_px,sigma_e_py,gamma,sig_g,pmag0,sigmaPmag,sigma_p_x,sigma_p_y,a0,lm,sign,lambda0,iTypeEnv)
    else:              # from an input file
        pxv, pyv, pzv, gav, xv, yv, n_g, n_s, Npart = readICs('data_ICs.in')

    #-------------------------------------------------------
    # Dump initial distribution to an external file ICs.txt
    #-------------------------------------------------------
    fic = open('ICs_SENSE.txt', 'w')
    for k in range(0, Npart):
        line = r'%15.8e  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e' % (pxv[k],pyv[k],pzv[k],gav[k],xv[k],yv[k])
        fic.write(line)
        fic.write('\n')
    fic.close()

    #-------------------------------------------------------------
    # Compute the mean energy of the sampled e-beam distribution
    # (For verification purposes)
    #-------------------------------------------------------------
    En0_c = 0.0
    for n in range(0, Npart):
        En0_c += sqrt((c**2)*(pxv[n]**2 + pyv[n]**2 + pzv[n]**2 + (me*c)**2))/eV2J
    En0_c = En0_c/Npart
    En0 = En0_c

    #-------------------------------------
    # Print out the simulation parameters
    #-------------------------------------
    print("******************************************************")
    print("***          SENSE Simulation Parameters           ***")
    print("******************************************************")
    if (iFile == 1):
        print(Npart, " ICs read from file data_ICs.in")
    else:
        print(Npart, " ICs randomly generated from input parameters")
    print("------------------------------------------------------")
    print("e-beam:  En0        =  ", En0, ' eV')
    print("e-beam:  gamma_0    =  ", gamma)
    print("e-beam:  sig_g      =  ", sig_g)
    print("e-beam:  eps_x_n    =  ", eps_x_n, ' m rad')
    print("e-beam:  eps_y_n    =  ", eps_y_n, ' m rad')
    print("e-beam:  eps_x      =  ", eps_x, ' m rad')
    print("e-beam:  eps_y      =  ", eps_y, ' m rad')
    print("e-beam:  sigma_e_x  =  ", sigma_e_x, ' m')
    print("e-beam:  sigma_e_y  =  ", sigma_e_y, ' m')
    print("e-beam:  sigma_e_px =  ", sigma_e_px, ' rad')
    print("e-beam:  sigma_e_py =  ", sigma_e_py, ' rad')
    print("------------------------------------------------------")
    print("p-beam:  lambda0    =  ", lambda0, ' m')
    print("p-beam:  p_0        =  ", pmag0, ' kg m/s')
    print("p-beam:  sig        =  ", sign)
    print("p-beam:  sigma_p_x  =  ", sigma_p_x, ' m')
    print("p-beam:  sigma_p_y  =  ", sigma_p_y, ' m')
    print("p-beam:  a0         =  ", a0)
    print("p-beam:  iTypeEnv   =  ", iTypeEnv)
    print("p-beam:  modType    =  ", modType)
    print("p-beam:  n_gamma    =  ", n_gamma)
    print("p-beam:  X          =  ", Xs)
    print("p-beam:  angle      =  ", l_angle, ' rad')
    print("------------------------------------------------------")
    if (modType == 0):
        print("p-beam:  no chirp")
    elif (modType == 1):
        print("p-beam:  exact TDHK 2014 chirp f(0)=1")
    elif (modType == 2):
        print("p-beam:  exact TDHK 2014 chirp f(+/-inf)=1")
    elif (modType == 3):
        print("p-beam:  quadratic RF chirp with lambda_RF = ", fm_param)
    elif (modType == 4):
        print("p-beam:  sinusoidal RF chirp with lambda_RF = ", fm_param)
    elif (modType == 5):
        print("p-beam:  exact 3D chirp with p = ", fm_param)
    elif (modType == 6):
        print("p-beam:  chirp with ang. dep. f(0)=1 theta_FM = ", fm_param)
    elif (modType == 7):
        print("p-beam:  chirp with ang. dep. f(+/-inf)=1 theta_FM = ", fm_param)
    elif (modType == 8):
        print("p-beam:  saw-tooth chirp slope = ", fm_param)
    elif (modType == 9):
        print("p-beam:  data-generated chirp")
    elif (modType == 10):
        print("p-beam:  empirical GSU 2013 prescription")
    else:
        print("p-beam:  no chirp")
    print("------------------------------------------------------")
    if (TypeAp == 0):
        print("aperture:  circular ")
        print("aperture:  L_aper   =  ", L_aper, ' m')
        print("aperture:  R_aper   =  ", R_aper, ' m')
        print("aperture:  theta_a  =  ", theta_max, ' rad')
    else:
        print("aperture:  rectangular ")
        print("aperture:  x_aper   =  ", x_aper, ' m')
        print("aperture:  y_aper   =  ", y_aper, ' m')
    print("------------------------------------------------------")
    if (eps_x != 0.0):
        print( "IP:  beta_x   =  ", (sigma_e_x**2)/eps_x, ' m')
    if (eps_y != 0.0):
        print("IP:  beta_y   =  ", (sigma_e_y**2)/eps_y, ' m')
    print("------------------------------------------------------")
    print("simulation:  w_min  =  ", wtilde_min)
    print("simulation:  w_max  =  ", wtilde_max)
    print("simulation:  Nout   =  ", Nout)
    print("simulation:  Ntot   =  ", Ntot)
    ("simulation:  Npart  =  ", Npart)
    print("simulation:  N_MC   =  ", N_MC)
    print("------------------------------------------------------")
    print("e-beam mean E_0     =  ", En0_c*1e-6, ' MeV (computed)')
    print("scattered mean E'   =  ", om0*hbar*1e-3, ' keV')
    print("******************************************************")

    #----------------------------------------
    # Monte Carlo sample the apertures
    #----------------------------------------
    if (TypeAp == 0):         # circular aperture
        rad_v, phi_v = sample_circle(N_MC,theta_max)
    else:                      # rectangular aperture
        rad_v, phi_v = sample_rectangle(N_MC,L_aper,x_aper,y_aper)
    print("Finished sampling electron distribution...")

    #----------------------------------------------------------------------
    # Compute all parameters used in integration.
    # Terzic et al. 2019, Eq.(4)-(8) and unnumbered equations in between.
    #----------------------------------------------------------------------
    distFile = open('dist_SENSE.out', 'w')
    pxvt    = np.zeros(Npart)
    pyvt    = np.zeros(Npart)
    pzvt    = np.zeros(Npart)
    prat_t  = np.zeros(Npart)
    prat    = np.zeros(Npart)
    x0      = np.zeros(Npart)
    y0      = np.zeros(Npart)
    sig_z_v = np.zeros(Npart)
    eta_v   = np.zeros(Npart)
    a0_v    = np.zeros(Npart)
    shift_v = np.zeros(Npart)
    sig_z   = sign*lambda0
    lm = lambda0/(me*c)
    Phi = pi - l_angle
    for k in range(Npart):
        # Terzic et al. 2019, equation before Eq. (6): normalized momenta
        pxvt[k] = pxv[k]*lm
        pyvt[k] = pyv[k]*lm
        pzvt[k] = pzv[k]*lm
        # Terzic et al. 2019, Eq. (4) not entirely correct 
#        prat[k] = (pzv[k]/gav[k])/sqrt(pxv[k]**2 + pyv[k]**2 + (pzv[k]/gav[k])**2)
        pv = sqrt(pxv[k]**2 + pyv[k]**2 + pzv[k]**2)
        be_x = pxv[k]/pv
        be_y = pyv[k]/pv
        be_z = pzv[k]/pv
        be = sqrt(be_x**2 + be_y**2 + be_z**2)
        prat[k] = (1.0 - be_x*sin(Phi) - be_z*cos(Phi))/(1+be)
        # Terzic et al. 2019, Eq. (8)
        prat_t[k] = prat[k]*sqrt(((pxvt[k]*sig_z)/(prat[k]*sigma_p_x))**2 
                                +((pyvt[k]*sig_z)/(prat[k]*sigma_p_y))**2 +1.0)
        x0[k] = pxv[k]/pzv[k]
        y0[k] = pyv[k]/pzv[k]
        # Terzic et al. 2019, unnumbered equation between Eqs. (7) and (8) with
        # z_0=0: scattering is modeled to have happened at the center of the pulse
        sig_z_v[k] = sqrt(1.0/((pxvt[k]**2)/((sigma_p_x**2))
                   +(pyvt[k]**2)/((sigma_p_y**2)) + (prat[k]**2)/((sign**2))))
        eta_v[k] = (sig_z_v[k]**2)*((pxvt[k]*xv[k])/(sigma_p_x**2)+(pyvt[k]*yv[k])/(sigma_p_y**2))
        a0_v[k]  = a0*exp(-((xv[k]**2)/(2*(sigma_p_x**2)))-((yv[k]**2)/(2*(sigma_p_y**2))))*exp((eta_v[k]**2)/(2*(sig_z_v[k]**2)))
        # Terzic et al. 2019, shift factor 1/k (Eqs. (4) and (5))
        shift_v[k] = (((En0/mc2)/gav[k])**2)/prat_t[k]

        #--------------------------------------------------------
        # Write out parameters a0, eta, sig_z, p, p~, shift
        #--------------------------------------------------------
        line = r'%15.8e  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e' % (a0_v[k],eta_v[k],sig_z_v[k],sign,prat[k],prat_t[k],shift_v[k],pxvt[k],pyvt[k],En0)
        distFile.write(line)
        distFile.write('\n')
    distFile.close()
    print("Finished computing parameters for each electron...")

    #---------------------------------------------------------------
    # Tabulate inner A(xi) integrals (Terzic et al. 2019: Eq. (3)).
    # Computed once and used repeatedly in d^2E/dOmega domega.
    #---------------------------------------------------------------
    xi = np.mgrid[-ntothm1:(Ntot-1)-ntothm1:Ntot*1j]
    ax,axi,axi2,xi_2 = XSENSE_module.tabulate1_ax(iTypeEnv, exp_f, exp_xi, modType,fmdfunc,fmd_xi,fmd_f,Nout,Ntot,sign,1.0,a0,a0chirp,fm_param,beta)
    #---------------------------
    # Dump the chirping profile 
    #---------------------------
    if (modType != 1): #changed to txt from out
       chirpFile = open('fmod_SENSE.txt', 'w')
       for i in range (0, Ntot):
            fxi = XSENSE_module.fmod(xi[i],modType,sign,a0chirp,fm_param,beta,fmdfunc,fmd_xi,fmd_f)
            line = r'%15.8e  %15.8e  %15.8e' % (xi[i],fxi,a0chirp)
            chirpFile.write(line)
            chirpFile.write('\n')
       chirpFile.close()

    #---------------------------
    #  Print Vec Potential
    #---------------------------

    #vec_pot= []
    #x_axis = []

    chirpFile = open('vec_pot_SENSE.txt', 'w')
    for i in range (0, Ntot-1):
         line = r'%15.8e %15.8e %15.8e  %15.8e' % (ax[i],axi[i],axi2[i],xi_2[i])
         chirpFile.write(line)
         chirpFile.write('\n')
    chirpFile.close()

    #--------------------------------------------
    # Set up x-coordinate (frequencies/energies)
    #--------------------------------------------
    xplot = []
    xplotN = []
    dom = (wtilde_max - wtilde_min)/(Nout-1)
    for i in range(0, Nout):
        xvalN = wtilde_min + dom*i   # [ ], normalized frequency
        xplotN.append(xvalN)         # [ ]
        xval = om0*hbar*xvalN        # [eV]
        xplot.append(xval)           # [eV]

    #----------------------
    # Parallel computation
    #----------------------
    results = []

    if (sys.platform == 'linux'): #Ryan 12/21/22
        cpu = len(os.sched_getaffinity(0)) #CPU Allocation for Linux Servers
    else:
        cpu = mp.cpu_count() #CPU Allocation for Local Runs on Mac/Windows
    
    pool = mp.Pool(processes=cpu)    # parallel execute on cpu CPUs
    print("Now executing spectrum computation on ", cpu, " CPUs")
    for x in range(0, Npart):        # parallelize over Npart
        pool.apply_async(XSENSE_module.loopParticles,args=(a0_v[x]
                         ,shift_v[x],x0[x],y0[x],Nout,Ntot
                         ,wtilde_min,wtilde_max,c1,omega0,beta
                         ,d_omega,N_MC,rad_v,phi_v,ax,axi,axi2,xi
                         ,gav[k],lambda0,x,iCompton)
                         ,callback=logResult)
    pool.close()
    pool.join()                      # ends processes when completed

    #------------------------------
    # Organize spectrum for output
    #------------------------------
    faloss = open('Eloss_SENSE.txt', 'w')  # find average E loss
    multipleSpectrum = [0 for x in range(Nout)]
    Yall = [0 for x in range(Nout)]
    Yall_r = [0 for x in range(Nout)]
    Eloss = 0.0
    A_tot = []
    e_tot = []
    for n in range(0, Npart):
        oneSpectrum = [0 for x in range(Nout)]
        oneSpectrum[0:Nout-1] = results[n][0:Nout-1]
        ind = results[n][Nout]
        multipleSpectrum, Eloss1, A_i, e_tot_i = process_one(xplot,oneSpectrum,gav[ind],a0,lambda0,n_s[ind],Nout,RRmode)
        A_tot.append(A_i)
        e_tot.append(e_tot_i)
        Eloss = Eloss + Eloss1
        line = r'%5d  %15.8e  %15.8e  %15.8e' % (n,Eloss1,gav[n]*mc2,gav[n]*mc2-Eloss1)
        faloss.write(line)
        faloss.write('\n')
        for i in range(0, Nout):      # sum up all individual spectra
            Yall[i] += oneSpectrum[i]
            Yall_r[i] += multipleSpectrum[i]
    faloss.close()

    A_tot, e_tot = totalCrit(A_tot, e_tot, int(Nout/3), int(2 * Nout/3))

    #A_tot, e_tot = totalCrit(A_tot, e_tot, xplot[int(Ntot/3)], xplot[int(2 * Ntot/3)])

    Eloss = Eloss/Npart
    fel = open('avg_Eloss_SENSE.txt','w')
    line = r'%15.8e' % (Eloss)
    fel.write(line)
    fel.close()

    for i in range(0, Nout):
        if (xplot[i] == 0.0):
            Yall[i] = 0.0
            Yall_r[i] = 0.0
        else:
            Yall[i] = Yall[i]/xplot[i]       # dN/dE' [1/eV]
                                    # for dE/dE' [eV/eV], comment this line out
            Yall_r[i] = Yall_r[i]/xplot[i]   # dN/dE' [1/eV]
                                    # for dE/dE' [eV/eV], comment this line out
    Ynorm = max(Yall)
    Ynorm_r = max(Yall_r)

    yplot  = []
    yplotN = []
    yplot_r = []
    yplot_rN = []
    for i in range(0, Nout):
        yplot.append(yfac*Yall[i]/Npart)         # dN/dE' [1/eV]
        yplotN.append(Yall[i]/Ynorm)             # normalized to 1
        yplot_r.append(yfac*Yall_r[i]/Npart_all) # dN/dE' [1/eV]
        yplot_rN.append(Yall_r[i]/Ynorm_r)       # normalized to 1

    A_tot = yfac * A_tot / Npart_all
    print("A = ", A_tot)

    file = open('Cole_A_crit.txt', 'w')
    line = r'%15.8e  %15.8e' % (A_tot, e_tot)
    file.write(line)
    file.close()


#    for i in range(0, Nout):         # sum up all individual spectra
#        for n in range(0, Npart):
#            Yall[i] += results[n][i]

#    for i in range(0, Nout):
#        if (xplot[i] == 0.0):
#            Yall[i] = 0.0
#        else:                       # dN/(dO dE') [Js/(hbarJ*eV) = 1/eV]
#                                    # for dE/dE' [eV/eV], comment this line out
#            Yall[i] = Yall[i]/xplot[i]
#    Ynorm = max(Yall)

#    yplotN=[]
#    yplotNn=[]
#    for i in range(0, Nout):
#        yplotN.append(yfac*Yall[i]/Npart)   # dN/dE' [1/eV]
#        yplotNn.append(Yall[i]/Ynorm)       # dN/dE' [1/eV]

    #-----------------------------------
    # Dump the total spectrum to a file
    #-----------------------------------
    f = open('output_SENSE.txt','w')
    for i in range(0, Nout):
        #line = r'%15.8e  %15.8e  %15.8e  %15.8e' % (xplot[i],xplotN[i],yplotN[i],yplotNn[i])
        line = r'%15.8e  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e' % (xplot[i], xplotN[i], yplot[i], yplotN[i], yplot_r[i], yplot_rN[i])
        f.write(line)
        f.write('\n')
    f.close()
