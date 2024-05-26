import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs
import dedalus.public as d3
import logging
import scipy
import matplotlib as mpl
from matplotlib import rc

################################################################################
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

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

ind = 1
f = h5py.File('snapshots/snapshots_s1.h5')
print(list(f['scales']))

u_phi_c = f['tasks']['u_phi_c']
u_theta_c = f['tasks']['u_theta_c']

# Parameters
Nphi    = 256
Ntheta  = 128
dealias = 3/2
dtype   = np.float64
R       = 1

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist   = d3.Distributor(coords, dtype=dtype)
basis  = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

#######################
Nt         = len(u_phi_c[:,0,0]);
print('total number of time steps=',Nt)
it_sta     = Nt - 10
it_end     = Nt

cnt = 0
E_spec_avg = np.zeros(Ntheta)
for it in range(it_sta,it_end):
  cnt += 1
  u_c    = u_phi_c[it,:,:]
  v_c    = u_theta_c[it,:,:]
  E_spec = np.zeros(Ntheta)
  for i in range(u_c.shape[0]):
    for j in range(u_c.shape[1]):
      groups = basis.elements_to_groups((False, False), (np.array((i,)),np.array((j,))))
      m   = int(groups[0][0])
      ell = int(groups[1][0])
      E_spec[ell] += 0.5*(u_c[i,j]**2 + v_c[i,j]**2)
  E_spec_avg += E_spec
E_spec_avg /= cnt
############################################################
ells = np.arange(Ntheta)+1

plt.figure(layout='constrained')
plt.loglog(ells,E_spec_avg); plt.ylabel('$E(\\ell)$'); plt.xlabel('$\\ell$')

plt.figure(layout='constrained'); plt.ylabel('$\\ell^2(\\ell+1)^2E(\\ell)$'); plt.xlabel('$\\ell$')
plt.loglog(ells,E_spec_avg*ells**2*(ells+1)**2); plt.show()
