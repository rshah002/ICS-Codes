import random
import numpy as np
import math
from libc.math cimport exp, cos, sin, sqrt, asin, acos, atan2, fabs
import scipy.integrate as integrate

#---------------------
# Physical constants
#---------------------
cdef double pi = np.pi
cdef double e = np.e
cdef double hbar = 6.582e-16          # Planck's constant [eV*s]
cdef double e0 = 8.85e-12             # epsilon_0 [C^2/Nm^2]
cdef double c = 2.998e8               # speed of light [m/s]
cdef double r = 2.818e-15             # electron radius [m]
cdef double ec = 511e3                # electron rest energy [eV]
cdef double q = 1.602e-19             # electron charge [C]
cdef double eVtoJ = 1.602176565e-19   # eV to Joules conversion
cdef double me  = 9.10938356e-31      # electron mass [kg]
cdef double mu0 = 4*pi*1e-7           # [N/A^2]


#-------------------------------
# Laser beam envelope function
#-------------------------------
def alpha_x(aPeak, xi, sign, iTypeEnv, exp_f, exp_xi):  #Beth- added exp_f and exp_xi parameters
    xis2 = 0.0
    if (iTypeEnv == 1):       # Gaussian 
        xis2 = (xi ** 2)/(2.0 * (sign ** 2))
        value = aPeak*exp(-xis2)
    elif (iTypeEnv == 2):     # rectangle 
        if ((xi > 0.0) & (xi < 2*pi*sign)):
            value = aPeak#/1.176
        else:
            value = 0.0
    #Beth 6/23/2020- add 3rd option to create function from data file 
    elif (iTypeEnv == 3):
        if ((xi< exp_xi[0]) or (xi > exp_xi[-1])) :
            value=0
        else:
            value=exp_f(xi)
    elif (iTypeEnv == 4):           #Ryan 7/15/2022 - add box_function
        if ((xi< -(sign/2)) or (xi > (sign/2))):
            value = 0
        else:
            sum_iter = 40                             # Summation Iterations

            #B = (aPeak*2*pi*me*c)/(lam_u*q)  ((aPeak)*(4/pi))   #Calculate Magnetic Field Amplitude from K and lam_u

            summ_wave = 0                           # Initialize sum to 0
            for n in range(1,sum_iter+1):               # Calculate sum for sum_iter terms
                b_wave=cos(2*pi*((2*n)-1)*(xi+0.25))/(((2*n)-1)**2)         # Store value in b_wave  !!!!!!!!!!!!!! Removed Normalizing Square Wave Amplitude !!!!!!!!
                summ_wave=summ_wave+b_wave          # Sum values wrt z
            value = summ_wave*aPeak # !!!!!!! Added *aPeak !!!!!!!!!!
    elif (iTypeEnv == 5):           #Ryan 7/15/2022 - add wille vec pot
        if ((xi< -(sign/2)) or (xi > (sign/2))):
            value = 0
        else:
            wille=cos(2*pi*(xi+0.25))        
            value = wille*aPeak # !!!!!!! Added *aPeak !!!!!!!!!!
    else:
        xis2 = (xi ** 2)/(2.0 * (sign **2))
        value = aPeak*exp(-xis2)
    return value


#-------------------------------
# Frequency modulation function 
#-------------------------------
def A(Y,a0):
    return (a0**2)*exp(-2.0*(Y**2))/2.0

def s_1(Y,a0,p):
    return (p*A(Y,a0)/2.0 + (p**3.0)/27.0
           + sqrt((p**4)*A(Y,a0)/27.0 + (p**2)*(A(Y,a0)**2)/4.0))**(1.0/3.0)

def s_2(Y,a0,p):
    return (p*A(Y,a0)/2.0 + (p**3.0)/27.0
           - sqrt((p**4)*A(Y,a0)/27.0 + (p**2)*(A(Y,a0)**2)/4.0))**(1.0/3.0)

def fmod (xi,modType,sign,a0chirp,lambda_RF,fmdfunc,fmd_xi,fmd_f):  #Beth- added fmdfunc, fmd_xi, and fmd_f parameters
    if (modType == 1):
        fmod = 1.0
    elif (modType == 10):
        fmod = 0.0
    elif (modType == 2):    # exact from TDHK 2014 (f(0) = 1)
        cons1 = 1.0/(1.0 + 0.5*(a0chirp**2))
        if (xi == 0):
            fmod = 1.0
        else:
            fmod = cons1*(1.0+(a0chirp**2)*sqrt(pi)*sign/(4.0*xi)*math.erf(xi/sign))
    elif (modType == 3):    # lambda_RF chirp quadratic
        fmod = 1.0 - 4.0*(pi**2)*(xi**2)/(3.0*(lambda_RF**2))
    elif (modType == 4):    # lambda RF chirp sinusoidal
        eps = 1e-5
        if (abs(xi) > eps):
            fmod = 0.6666667*(1.0 + (lambda_RF/(8.0*pi*xi))*sin(4.0*pi*xi/lambda_RF))
        else:
            fmod = 1.0
    elif (modType == 5):    # exact from Maroli et al. 2018
        p = lambda_RF
        Y = xi/(sqrt(2.0)*sign)
        eps = 1e-5
        tmp0 = integrate.quad(lambda x: s_1(x,a0chirp,p) + s_2(x,a0chirp,p),0,eps)
        tmp  = integrate.quad(lambda x: s_1(x,a0chirp,p) + s_2(x,a0chirp,p),0,Y)
        if (abs(Y) > eps):
            fmod = (p/3.0 + tmp[0]/Y)/(p/3.0 + tmp0[0]/eps)
        else:
            fmod = 1.0
    elif (modType == 6):    # empirical from GSU 2013
        fmod = (2.0/3.0)*(sqrt(pi)*sign/(4.0*xi))*math.erf(xi/sign)
        if (xi == 0):
            fmod = 1
    elif (modType == 7):    # exact from TDHK 2014 (f(+/-inf) = 1)
        cons1 = 1.0
        if (xi == 0):
            fmod = 1.0
        else:
            fmod = cons1*(1.0+(a0chirp**2)*sqrt(pi)*sign/(4.0*xi)*math.erf(xi/sign))
    elif (modType == 8):    # saw-tooth chirp 
        cons1 = 1.0
        fmod = 1.0 - a0chirp*abs(xi)
        if (fmod < 0.0):
           fmod = 0.0
    elif (modType == 9):    # data-generated chirp -Beth
        if ((xi< fmd_xi[0]) or (xi > fmd_xi[-1])) :
            fmod=fmd_f[0]
        else:
            fmod=fmdfunc(xi)
    else:
        fmod = 1.0
    return fmod


#---------------------------------------------------------------
# Tabulate inner A(xi) integrals (Terzic et al. 2019: Eq. (3)).
# Computed once and used repeatedly in d^2E/dOmega domega.
#---------------------------------------------------------------
def tabulate1_ax(iTypeEnv, exp_f, exp_xi, modType, fmdfunc, fmd_xi, fmd_f, Nout, Ntot, sign, aval, aPeak, a0chirp, lambda_RF): #Beth- added fmdfunc, fmd_xi, and fmd_f parameters
    nLambda = float(Nout - 1)
    halfStep = 0.5/nLambda

    ax_a = np.zeros(Ntot)
    axi_a = np.zeros(Ntot)
    axp_a = np.zeros(Ntot)
    axi2_a = np.zeros(Ntot)

    xi_a = np.zeros(Ntot)
    for i in range(-1, Ntot - 1):
        xip = (i - Ntot/2)/nLambda
        xi_a[i+1] = xip

    xip = Ntot/-2.0/nLambda
    alphax = alpha_x(aval, xip, sign, iTypeEnv, exp_f, exp_xi)
    f = fmod(xip, modType, sign, a0chirp, lambda_RF, fmdfunc, fmd_xi, fmd_f)
    ax_a[0] = alphax * cos(2.0*pi*f*xip)
    for i in range(Ntot - 1):
        xip = (i - Ntot/2)/nLambda
        alphax = alpha_x(aval, xip, sign, iTypeEnv, exp_f, exp_xi)
        f = fmod(xip, modType, sign, a0chirp, lambda_RF, fmdfunc, fmd_xi, fmd_f)
        xis2 = (xip ** 2)/(2.0 * (sign** 2))
        ax_a[i+1] = alphax*cos(2.0*pi*f*xip)
        alphax = alpha_x(aval, xip, sign, iTypeEnv, exp_f, exp_xi)
        f = fmod(xip - 0.5, modType, sign, a0chirp, lambda_RF, fmdfunc, fmd_xi, fmd_f)
        axp_a[i] = alphax*cos(2.0*pi*f*(xip-halfStep))
#       axp_a[i] = alphax*cos(2.0*pi*f*(xip-0.5))
        axi_a[i+1] = axi_a[i] + (ax_a[i] + ax_a[i+1] + 4.0*axp_a[i])/6.0
        axi2_a[i+1] = axi2_a[i] + ((ax_a[i]**2) + (ax_a[i+1]**2) + 4.0*(axp_a[i]**2))/6.0
    return ax_a, axi_a, axi2_a, xi_a


#-----------------------------------------------------
# Evaluate outermost integrand using Simpson's rule.
# Terzic et al. 2019: Eqs. (1)-(3)
#-----------------------------------------------------
cdef double dE2(int Nout, int Ntot, double ct, double phi, double omega, double[:] xi, double[:] ax, double[:] axi, double[:] axi2, double c1, double betaz):
    cdef double nLambda = Nout - 1
    cdef double omLam = omega/nLambda
    cdef double st = sqrt(1.0-ct**2)
    cdef double cp = cos(phi)
    cdef double sp = sin(phi)
    cdef double c2 = (1.0 - betaz * ct)/(1 + betaz)
    cdef double c3 = st * cp/c1
    cdef double c4 = (1.0 + ct)/(2.0*(c1**2))
    cdef double c6 = (ct - betaz)/(1 - betaz*ct)

    cdef double arg
    cdef double ca
    cdef double sa
    cdef double n1x = 0.0
    cdef double n2x = 0.0
    cdef double n1z = 0.0
    cdef double n2z = 0.0
    cdef int k
    for k in range(Ntot):
        arg = omLam * (xi[k]*c2 - axi[k]*c3 + axi2[k]*c4)
        ca = cos(arg)
        sa = sin(arg)
        n1x += ax[k] * ca
        n2x += ax[k] * sa
        n1z += (ax[k]**2) * ca
        n2z += (ax[k]**2) * sa
    cdef double Dx = sqrt(n1x**2 + n2x**2)/(c1*nLambda)
    cdef double Dz = sqrt(n1z**2 + n2z**2)/(2*c1*c1*c2*nLambda)
    cdef double Esigma = (omega*Dx*sp)**2
    cdef double Epi = (omega*(Dx*c6*cp + Dz*st))**2
    cdef double Yval = Epi + Esigma
    return Yval


#--------------------------------------------------------------
# Convert to Monte Carlo-sampled circle representing aperture
#--------------------------------------------------------------
def convert_coord(radius,phi,x0,y0):
    x0p = x0 + np.array(radius)*np.cos(phi)
    y0p = y0 + np.array(radius)*np.sin(phi)
    rad_p = np.sqrt(x0p**2 + y0p**2)
    cth = np.cos(rad_p)
    ph = np.zeros(len(x0p))
    for i in range(len(x0p)):
        ph[i] = atan2(y0p[i],x0p[i])
    return cth, ph


#----------------------------------------------------
# Evaluate the outer integrand using Simpson's rule
#---------------------------------------------------------------------
# Integrals over angles theta and phi are done by summing N_MC points
# randomly sampling the circular area of the aperture.
#---------------------------------------------------------------------
def loopParticles(a0_v,shift_v,x0,y0,Nout,Ntot,omega_omega0_min,omega_omega0_max,c1,omega0,beta,d_omega,N_MC,rad_v,phi_v,ax_v,axi_v,axi2_v,xi):
#------------------------------------------------------------------
# Compute the contribution of each electron to the aperture by
# evaluating Terzic et al. 2019: Eqs. (1)-(3) for each of N_MC
# points, each with its own scattering angle (theta, phi), within
# the finite aperture given in the input file config.in.
#------------------------------------------------------------------
    Y = [0 for x in range(Nout)]
    ntothm1 = Ntot/2.0 - 1.0
    ax   = a0_v*ax_v
    axi  = a0_v*axi_v
    axi2 = (a0_v**2)*axi2_v
    cth, ph = convert_coord(rad_v,phi_v,x0,y0)
    for n in range(Nout):       # loop over scattering frequencies
        omega = omega0 * (omega_omega0_min + n*d_omega) * shift_v
        for i in range(N_MC):   # loop over N_MC angles (theta, phi)
            yp = dE2(Nout,Ntot,cth[i],ph[i],omega,xi,ax,axi,axi2,c1,beta)
#------------------------------------------------------------------
# Amplitude of the interpolated result d2E/dodO must be scaled by
# frequency scale squared (Terzic et al. 2016, Eq. 5)
#------------------------------------------------------------------
            Y[n] += yp/(shift_v**2)
    return Y
