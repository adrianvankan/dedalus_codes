#PLOTS SCRIPT PLOTS TIME SERIES OF ENERGY, ENSTROPHY AND POLARITY

import numpy as np
from matplotlib import pyplot as plt
#from scipy.stats import levy_stable
#from mpmath import *
import matplotlib as mpl
from matplotlib import rc
#from datetime import datetime
#import numba as nb
import h5py

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)

#set font sizes
SMALL_SIZE = 22
MEDIUM_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize
################################################################################
inds = [1]
t0 = 0 #time offset (if not starting from 0)
for ind in inds:
  f = h5py.File('profiles/profiles_s1'+'.h5','r')
  print(list(f.keys()))
  
  dset  = f['tasks']['1D_zonal_velocity']
  theta = np.linspace(-np.pi/2,np.pi/2,len(dset[0,0,:]))
  plt.figure(1,layout='constrained')
  for i in range(np.shape(dset)[0]):
      plt.plot(theta,dset[i,0,:])
if False:
  plt.figure(1)
  plt.savefig('u_mean.png')
plt.show()

