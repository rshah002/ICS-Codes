#==============================================================================
# SENSE (Simulation of Emitted Nonlinear Scattering Events) code
#
# Computes inverse Thomson scattering spectrum in the non-linear regime.
# Described in detail in
# - Terzic et al. 2019, EPL 126, 12003
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
import random
import numpy as np
import SENSE_module
import multiprocessing as mp
from math import pi,e,cos,sin,sqrt,atan,asin,exp
from scipy.interpolate import interp1d  #Beth 6/23/2020
import scipy.integrate as integrate

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
me = 9.10938356e-31        # electron mass [kg]
alpha = 1.0/137.0          # fine structure constant

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
def sample_df(Npart,sigma_e_x,sigma_e_y,sigma_e_px,sigma_e_py,gamma,sig_g,pmag0,sigmaPmag,sigma_p,a0):
    random.seed(1)      # for reproducability of the results, fix the same seed
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
        rat = exp(-((x**2)/(2*(sigma_p**2)))-((y**2)/(2*(sigma_p**2))))
        if (rat > 0.05):
            i += 1
            xv.append(x)
            yv.append(y)
    return pxv, pyv, pzv, gav, xv, yv


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
    sigma_p = args[8]          # laser beam: transverse beam size [m]
    a0 = args[9]               # laser beam: field strength a0 []
    iTypeEnv = int(args[10])   # laser beam: laser envelope type
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
    if (iTypeEnv == 4 or 5): # Ryan
        lam_u = args[12]
    modType = int(args[11])    # laser beam: frequency modulation type
    fmd_xi=[]
    fmd_f=[]
    fmdfunc=0
    if (modType == 2):         # exact 1D chirping from TDHK 2014 (f(0) = 1)
        a0chirp = float(args[12])     # laser beam: a0 chirping value
        lambda_RF = 0.0
    elif (modType == 3):       # RF quadratic chirping
        a0chirp = 0.0
        lambda_RF = float(args[12])   # laser beam: lambda_RF chirping value
    elif (modType == 4):       # RF sinusoidal chirping
        a0chirp = 0.0
        lambda_RF = float(args[12])   # laser beam: lambda_RF chirping value
    elif (modType == 5):       # exact 3D chirping
        a0chirp = a0           # laser beam: a0 chirping value
        lambda_RF = float(args[12])
    elif (modType == 6):       # chirping from GSU 2013
        a0chirp = a0           
        lambda_RF = float(args[12])
    elif (modType == 7):       # exact 1D chirping from TDHK 2014 (f(+/-inf) = 1)
        a0chirp = float(args[12])     # laser beam: a0 chirping value
        lambda_RF = 0.0
    elif (modType == 8):       # saw-tooth chirp
        a0chirp = float(args[12])     # laser beam: a0 chirping value
        lambda_RF = 0.0
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
        a0chirp =  float(args[12])
        lambda_RF = 0.0
    else:                      # no chirping
        a0chirp = 0.0
        lambda_RF = 0.0
    l_angle = args[13]         # laser beam: angle between laser & z-axis [rad]

    #---------------------
    # Aperture parameters
    #---------------------
    TypeAp = args[14]          # aperture: type: =0 circular; =1 rectangular
    L_aper = args[15]          # aperture: distance from IP to aperture [m]
    if (TypeAp == 0):
        R_aper = args[16]      # aperture: physical radius of the aperture [m]
        tmp = args[17]
        theta_max = atan(R_aper/L_aper)
    else:
        x_aper = args[16]
        y_aper = args[17]
 
    #-----------------------
    # Simulation parameters
    #-----------------------
    wtilde_min = args[18]      # simulation: start spectrum [normalized w/w0 units]
    wtilde_max = args[19]      # simulation: spectrum range [normalized w/w0 units]
    Nout = int(args[20])       # simulation: number of points in the spectrum
    Ntot = int(args[21])       # simulation: resolution of the inner intergral
    Npart = int(args[22])      # simulation: number of electron simulated
    N_MC = int(args[23])       # simulation: number of MC samples
    iFile = int(args[24])      # simulation: =1: read ICs from file; <> 1: MC

    #--------------------------------------------
    # Compute basic parameters for the two beams
    #--------------------------------------------
    gamma = En0/mc2
    beta = sqrt(1.0-1.0/(gamma**2))
    c1 = gamma*(1.0 + beta)
    omega0 = 2.0*pi*c1*c1
    ntothm1 = Ntot/2.0 - 1.0
    d_omega = (wtilde_max - wtilde_min)/(Nout-1)

    eps_x = eps_x_n/gamma
    eps_y = eps_y_n/gamma
    sigma_e_px = eps_x/sigma_e_x
    sigma_e_py = eps_y/sigma_e_y
    pmag0 = (eV2J/c)*sqrt(En0**2-mc2**2)
    sigmaPmag = (eV2J/c)*sqrt(((En0*(1.0+sig_g))**2)-mc2**2)-pmag0
    E_laser = 2*pi*hbar*c/lambda0

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
    yfac = q/(8*(pi**2)*(c**2))                    # [J s]  dE/domega
    yfac = yfac*(J2eV/hbar)                        # [(J s)*(eV/J)/(eV s)] = []
    yfac = yfac/float(N_MC)                        # [(J s)*(eV/J)/(eV s)] = []
    if (TypeAp == 0):                     # circular aperture
        yfac = pi*(theta_max**2)*yfac              # d Omega <dE/domega dOmega> 
    else:                                 # rectangular aperture
        yfac = (x_aper*y_aper)*yfac                # d Omega <dE/domega dOmega> 

    #-------------------------------------------------------
    # Create or read in initial distribution for the e-beam
    #-------------------------------------------------------
    if (iFile != 1):   # randomly generated from input parameters
        pxv, pyv, pzv, gav, xv, yv = sample_df(Npart,sigma_e_x,sigma_e_y,sigma_e_px,sigma_e_py,gamma,sig_g,pmag0,sigmaPmag,sigma_p,a0)
    else:              # from an input file
        pxv, pyv, pzv, gav, xv, yv, Npart = readICs('data_ICs.in')

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
    print("p-beam:  sigma_p    =  ", sigma_p, ' m')
    print("p-beam:  a0         =  ", a0)
    print("p-beam:  iTypeEnv   =  ", iTypeEnv)
    print("p-beam:  modType    =  ", modType)
    print("p-beam:  n_gamma    =  ", n_gamma)
    print("p-beam:  X          =  ", Xs)
    print("p-beam:  angle      =  ", l_angle, ' rad')
    print("------------------------------------------------------")
    if (modType == 1):
        print("p-beam:  no chirp")
    elif (modType == 2):
        print("p-beam:  exact chirp with a0_chirp = ", a0chirp)
    elif (modType == 3):
        print("p-beam:  quadratic RF chirp with lambda_RF = ", lambda_RF)
    elif (modType == 4):
        print("p-beam:  sinusoidal RF chirp with lambda_RF = ", lambda_RF)
    elif (modType == 5):
        print("p-beam:  exact 3D chirp with a0_chirp = ", a0chirp)
    elif (modType == 6):
        print("p-beam:  empirical GSU 2013 prescription")
    elif (modType == 7):
        print("p-beam:  exact chirp with f(+/-inf)=1 a0_chirp = ", a0chirp)
    elif (modType == 8):
        print("p-beam:  saw-tooth chirp slope = ", a0chirp)
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
    print("simulation:  Npart  =  ", Npart)
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
        prat_t[k] = prat[k]*sqrt(((pxvt[k]*sig_z)/(prat[k]*sigma_p))**2 
                                +((pyvt[k]*sig_z)/(prat[k]*sigma_p))**2 +1.0)
        x0[k] = pxv[k]/pzv[k]
        y0[k] = pyv[k]/pzv[k]
        # Terzic et al. 2019, unnumbered equation between Eqs. (7) and (8) with
        # z_0=0: scattering is modeled to have happened at the center of the pulse
        sig_z_v[k] = sqrt(1.0/((pxvt[k]**2)/((sigma_p**2))
                   +(pyvt[k]**2)/((sigma_p**2)) + (prat[k]**2)/((sign**2))))
        eta_v[k] = (sig_z_v[k]**2)*((pxvt[k]*xv[k])/(sigma_p**2)+(pyvt[k]*yv[k])/(sigma_p**2))
        a0_v[k]  = a0*exp(-((xv[k]**2)/(2*(sigma_p**2)))-((yv[k]**2)/(2*(sigma_p**2))))*exp((eta_v[k]**2)/(2*(sig_z_v[k]**2)))
        # Terzic et al. 2019, shift factor 1/k (Eqs. (4) and (5))
        shift_v[k] = (((En0/mc2)/gav[k])**2)/prat_t[k]

        #--------------------------------------------------------
        # Write out parameters a0, eta, sig_z, p, p~, shift
        #--------------------------------------------------------
        line = r'%15.8e  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e' % (a0_v[k],eta_v[k],sig_z_v[k],sign,prat[k],prat_t[k],shift_v[k],pxvt[k],pyvt[k])
        distFile.write(line)
        distFile.write('\n')
    distFile.close()
    print("Finished computing parameters for each electron...")

    #---------------------------------------------------------------
    # Tabulate inner A(xi) integrals (Terzic et al. 2019: Eq. (3)).
    # Computed once and used repeatedly in d^2E/dOmega domega.
    #---------------------------------------------------------------
    xi = np.mgrid[-ntothm1:(Ntot-1)-ntothm1:Ntot*1j]
    ax,axi,axi2,xi_2 = SENSE_module.tabulate1_ax(iTypeEnv, exp_f, exp_xi, modType,fmdfunc,fmd_xi,fmd_f,Nout,Ntot,sign,1.0,a0,a0chirp,lambda_RF)

    
    #---------------------------
    # Dump the chirping profile 
    #---------------------------
    if (modType != 1): #changed to txt from out
       chirpFile = open('fmod_SENSE.txt', 'w')
       for i in range (0, Ntot):
            fxi = SENSE_module.fmod(xi[i],modType,sign,a0chirp,lambda_RF,fmdfunc,fmd_xi,fmd_f)
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
    cpu = mp.cpu_count()
    pool = mp.Pool(processes=cpu)    # parallel execute on cpu CPUs
    print("Now executing spectrum computation on ", cpu, " CPUs")
    for x in range(0, Npart):        # parallelize over Npart
        pool.apply_async(SENSE_module.loopParticles,args=(a0_v[x]
                         ,shift_v[x],x0[x],y0[x],Nout,Ntot
                         ,wtilde_min,wtilde_max,c1,omega0,beta
                         ,d_omega,N_MC,rad_v,phi_v,ax,axi,axi2,xi)
                         ,callback=logResult)
    pool.close()
    pool.join()                      # ends processes when completed

    #------------------------------
    # Organize spectrum for output
    #------------------------------
    Yall = [0 for x in range(Nout)]
    for i in range(0, Nout):         # sum up all individual spectra 
        for n in range(0, Npart):
            Yall[i] += results[n][i]

    for i in range(0, Nout):
        if (xplot[i] == 0.0):
            Yall[i] = 0.0
        else:
            Yall[i] = Yall[i]/xplot[i]   # dN/dE' [1/eV] 
                                    # for dE/dE' [eV/eV], comment this line out
    Ynorm = max(Yall)

    yplotN=[]
    yplotNn=[]
    for i in range(0, Nout):
        yplotN.append(yfac*Yall[i]/Npart)   # dN/dE' [1/eV]
        yplotNn.append(Yall[i]/Ynorm)       # normalized to 1

    #-----------------------------------
    # Dump the total spectrum to a file
    #-----------------------------------
    f = open('output_SENSE.txt','w')
    for i in range(0, Nout):
        line = r'%15.8e  %15.8e  %15.8e  %15.8e' % (xplot[i],xplotN[i],yplotN[i],yplotNn[i])
        f.write(line)
        f.write('\n')
    f.close()
