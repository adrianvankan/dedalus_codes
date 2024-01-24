import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import levy_stable
#from mpmath import *
import matplotlib as mpl
from matplotlib import rc
from datetime import datetime
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
t0 = 0
for ind in inds:
  f = h5py.File('scalar_data/scalar_data_s'+str(ind)+'/scalar_data_s'+str(ind)+'_p0.h5','r')
  print(list(f.keys()))

  dset  = f['tasks']
  print(list(dset))
  Ek    = dset['Ek'] 

  plt.figure(1)
  dt = 0.001
  time = t0 + dt*np.arange(0,len(Ek[:,0,0]))
  plt.plot(time,(Ek[:,0,0]),'.-');
  plt.xlabel('$t$'); plt.ylabel('$E$')
  plt.tight_layout()
  t0 = time[-1]
  
  plt.figure(2)
  Ek    = dset['Enstrophy']
  time = t0 + dt*np.arange(0,len(Ek[:,0,0]))
  plt.plot(time,(Ek[:,0,0]),'.-');
  plt.xlabel('$t$'); plt.ylabel('$\\Omega$') 
  plt.tight_layout()
  t0 = time[-1]
  plt.tight_layout()
  t0 = time[-1]

if False:
  plt.savefig('Tke_vs_time.pdf')
plt.show()
