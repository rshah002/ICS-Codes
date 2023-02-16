from scipy.interpolate import interp1d
from scipy.integrate  import quad
import math
import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------
#Import laser envelope data and config parameters
#--------------------------------------------------
data_file = open("Laser_Envelope_Data.txt", "r")
exp_data = []
for line in data_file:
    exp_data.append(line.strip().split())
exp_xi = []
exp_a = []
for line in exp_data:
    exp_xi.append(float(line[0]))
    exp_a.append(float(line[1]))
exp_af=interp1d(exp_xi,exp_a,kind='cubic')

data_file = open("config.txt", "r")
args = []
for line in data_file:
    i = 0
    while (line[i:i + 1] != " "):
        i += 1
    num = float(line[0:i])
    args.append(num)
sign = args[7]
a0 = args[9]


#---------------------------------
#Find laser pulse center of mass
#---------------------------------

area=quad(exp_af, exp_xi[0], exp_xi[-1])
def integrand(x):
    return (x * exp_af(x))

temp1=quad(integrand, exp_xi[0], 0)
temp2=quad(integrand, 0, exp_xi[-1])
xicm= (temp1[0]+temp2[0])/area[0]


#----------------------------
#Generate chirping function
#----------------------------

def fintegrand(x):
    return (a0**2) *(exp_af(x))**2

const=1/(1+ 0.5*(a0**2))


def fmod(x):
    if (x<exp_xi[0]):
        integral=quad(fintegrand, xicm, exp_xi[0])
    elif (x>exp_xi[-1]):
        integral=quad(fintegrand, xicm, exp_xi[-1])
    else:
        integral=quad(fintegrand, xicm, x)
    
    if abs(x)< 0.5:
        value=const*(1+(fintegrand(x)/2))
    else:
        value=const*(1+integral[0]/(2*x))
    return value

def fmodflat(x):
    return 1.0

def gaussf(x):      #analytical gaussian chirp (not generalized)
    cons1 = 1.0/(1.0 + 0.5*(a0**2))
    if (x == 0):
        fmod=1.0
    else:
        fmod=cons1*(1.0+(a0**2)*math.sqrt(np.pi)*sign/(4.0*x)*math.erf(x/sign))
    return fmod

def lorentzf(x):        #analytical lorentz chirp (not generalized)
    if (x == 0):
        fmod=1.0
    else: 
        fmod= const* (1 + ((math.sqrt(sign) * a0**2)/(4*x*math.sqrt(2))) * (math.sqrt(2*sign)*x/(sign + 2* x**2) +  math.atan(x*math.sqrt(2)/math.sqrt(sign))))
    return fmod

def secantf(x):
    fmod= const* (1 + ((a0**2)/(2*x*sign))* math.tanh(sign*x))
    return fmod


#-----------------------------------
#Write generated data to a file
#-----------------------------------

x=[]
y=[]
x=np.arange(exp_xi[0],exp_xi[-1],1)

for xi in x:
    y.append(fmod(xi))
file= open('Fmod_Data.txt', 'w')
for i in range(len(x)):
    file.write(str(x[i])+'\t'+str(y[i])+'\n')
file.close()

