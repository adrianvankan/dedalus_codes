# This Dedalus script solves the stochastically forced 2D Navier-Stokes equations on the sphere
# Author: Adrian van Kan
# Date  : 06 June 2024

import numpy as np
import dedalus.public as d3
import logging
import scipy
import dedalus.extras.flow_tools as flow_tools
from mpi4py import MPI
import dedalus.tools.logging as mpi_logging
import matplotlib.pyplot as plt
import time

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)
figkw = {'figsize':(6,4), 'dpi':100}

###### Numerical Parameters ####
Nphi      = 256        # Number of m modes (zonal)
Ntheta    = 128         # Number of l modes (meridional)
dealias   = 3/2
R         = 1.0        #Sphere radius (non-dim)
timestep  = 5e-4       #initial timestep
stop_sim_time = 1000   #stop time
dtype     = np.float64
seed0     = 1
max_timestep = 5e-4

restart    = False
cp_path    = 'checkpoints/checkpoints_s3.h5'

##### Dimensional Parameters ####
Omega     = 100           #Planetary rotation rate
nu        = 3e-3          #Viscosity
eps       = 1             #Energy injection rate

#####################
fact = 0.0
ang_mom_conserving_viscous_term = True
if ang_mom_conserving_viscous_term == True:
  fact = 1.0
######################
forcing_in_spectral_space = True
######################

# Specify momentum forcing range in meridional wavenumber l, with all |m| <= l being forced
ml_vec = []
ls = [10] #these l (meridional) modes are forced
for l in ls:
   for m in range(0,l+1): #these m (zonal) modes are forced
      ml_vec.append([m,l])

#Timestepper settings
timestepper = d3.RK443       

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist   = d3.Distributor(coords, dtype=dtype)
basis  = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

# Substitutions
zcross = lambda A: d3.MulCosine(d3.skew(A))

# Fields
u = dist.VectorField(coords, name='u',   bases=basis)
f = dist.VectorField(coords, name='f',   bases=basis)
p = dist.Field(      name='p',   bases=basis)
psi_f  = dist.Field(      name='psi_f',   bases=basis)
psi    = dist.Field(      name='psi',   bases=basis)
tau_p  = dist.Field(  name='tau_p')
g      = dist.Field(      name='g',   bases=basis)
#h      = dist.Field(      name='h',   bases=basis)

# Define unit vectors
eph   = dist.VectorField(coords, name='ephi', bases = basis)
eth   = dist.VectorField(coords, name='ephi', bases = basis)
eph['g'][0] = 1; eph['g'][1] = 0
eth['g'][0] = 0; eth['g'][1] = 1

# Coordinates
phi, theta = dist.local_grids(basis)
lat = np.pi / 2 - theta + 0*phi
phi_mat   = phi   + 0*theta
theta_mat = theta + 0*phi

g['g'] = np.cos(lat) # Auxiliary function

# Substitutions
zcross = lambda A: d3.MulCosine(d3.skew(A))

# Problem (nondimensional)
problem = d3.IVP([u, p, tau_p], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(u) +  grad(p) - nu*lap(u) - 2*fact*nu*u  = - 2*Omega*zcross(u)  - u@grad(u) + d3.skew(d3.grad(psi_f))")
problem.add_equation("ave(p) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

#Initial condition
if restart == False:
  l = 5
  m = 5 
  amp_psi  = 0.000
  psi['g'] = amp_psi * np.real(scipy.special.sph_harm(m,l,phi,theta)) 
  #For order of arguments of sph_harm see https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm.html
  file_handler_mode = 'overwrite'
  initial_timestep = max_timestep
  # Initial conditions: generate incompressible velocity field from stream function
  problem2 = d3.LBVP([u], namespace=locals())
  problem2.add_equation("u = d3.skew(d3.grad(psi)) ")
  solver2 = problem2.build_solver()
  solver2.solve()
else:
  write, initial_timestep = solver.load_state(cp_path)
  file_handler_mode = 'append'
 
# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=10000,mode=file_handler_mode)
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity',layout='g')
snapshots.add_task(eph@u, name='u_phi', layout='g')
snapshots.add_task(eth@u, name='u_theta',layout = 'g')
snapshots.add_task(eph@u, name='u_phi_c', layout='c')
snapshots.add_task(eth@u, name='u_theta_c', layout='c')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity_c', layout='c')
checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=200, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

# 1D Profiles
profiles = solver.evaluator.add_file_handler('profiles',sim_dt=0.05,max_writes = 50000,mode=file_handler_mode)
profiles.add_task(-d3.Average(eph@u, coords[0]),name='1D_zonal_velocity')

# Scalar Data
analysis1 = solver.evaluator.add_file_handler("scalar_data", sim_dt=0.01, max_writes = 1000000, mode=file_handler_mode)
analysis1.add_task(d3.Average(0.5*(u@u),coords), name="Ek")
analysis1.add_task(d3.Average((-d3.div(d3.skew(u)))**2, coords), name='Enstrophy')
analysis1.add_task( (d3.Average((eph@u)**2,coords)- d3.Average((eth@u)**2,coords))/(d3.Average(u@u,coords)+1e-8), name='polarity')

# Flow properties
flow_prop_cad = 25
flow = d3.GlobalFlowProperty(solver, cadence = flow_prop_cad)
flow.add_property(d3.Average(0.5*(u@u),coords), name = 'avgEkin')
flow.add_property(d3.Average(-nu*u@d3.lap(u),coords), name='diss_E')
flow.add_property(d3.Average(((eph@u)*(eph@u)-(eth@u)*(eth@u)))/(d3.Average((u@u),coords)+1e-8), name='polarity')

theta_arr = theta + 0*phi
phi_arr   = phi + 0*theta

#GlobalArrayReducer
reducer   = flow_tools.GlobalArrayReducer(comm=MPI.COMM_WORLD)

# CFL
CFL = d3.CFL(solver, initial_dt=initial_timestep, cadence=25, safety=0.8, threshold=0.8, max_change=100000, min_change=0.00001, max_dt=max_timestep)
CFL.add_velocity(u)

#set MPI rank and size
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Initialise random seed to seed0
if forcing_in_spectral_space == True:
   np.random.seed(seed0+rank)
elif forcing_in_spectral_space == False:
   np.random.seed(seed0)

it0 = solver.iteration
#############
# MAIN LOOP #
#############
g.preset_scales(dealias)
psi_f.preset_scales(dealias)
try:
    logger.info('Starting main loop')
    while solver.proceed:
        psi_f['g'] = 0
        if solver.iteration == it0:#not ( len(psi_f['g']) == len(phi_mat) and len(psi_f['g']) == len(theta_mat)) : 
          phi,theta = basis.local_grids(dist=dist,scales=(dealias,dealias));
          phi_mat   = phi   + 0*theta
          theta_mat = theta + 0*phi  
          
          for lf in ls:
            phase = 2*np.pi*np.random.rand(1)
            m, ell, *_ = dist.coeff_layout.local_group_arrays(basis.domain(dist), scales=1)
            #FOR DBG: print('array size',np.shape(psi_f['c'][(m>=0)*(ell == lf)]))
            shape_psi_f =np.shape(psi_f['c'][ell == lf])
            psi_f['c'][ell == lf] = 2*np.random.rand((shape_psi_f[0]))-1
          d3.Average(psi_f).evaluate()
          #plt.pcolormesh(psi_f['g']); plt.show()
        
        ts = time.time()
        if solver.iteration > it0:
          #####################################
          if forcing_in_spectral_space == True:
            for lf in ls: 
              shape_psi_f =np.shape(psi_f['c'][ell == lf])
              psi_f['c'][ell == lf] = 2*np.random.rand((shape_psi_f[0]))-1
          ##########################################        
          if forcing_in_spectral_space == False:
            for ml in ml_vec:
              m,l = ml
              phase = 2*np.pi*np.random.rand(1)
              psi_f['g'] += np.real(np.exp(1.0j*phase)*scipy.special.sph_harm(m,l,phi_mat,theta_mat))
          ##########################################  #For order of arguments of sph_harm, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm.html
          #d3.Average(psi_f).evaluate()
          #plt.clf();plt.title(str(rank)); plt.pcolormesh(np.transpose(psi_f['g']),cmap='seismic'); plt.colorbar(); plt.show()
          #compute momentum forcing from skew gradient of psi_f
          f     = d3.skew(d3.grad(psi_f)).evaluate()
          f_var = reducer.global_mean(g['g']*(f['g'][0]**2 + f['g'][1]**2))/reducer.global_mean(g['g'])
          #print(f_var)
          timestep = CFL.compute_timestep()
          psi_f['g'] = psi_f['g']/f_var**(0.5) * (2 * eps / timestep)**0.5 
          
        solver.step(timestep)
        #print('done with a timestep')
        if (solver.iteration) % flow_prop_cad == 0:
            avgEkin  = flow.max('avgEkin')
            diss_E   = flow.max('diss_E')
            m        = flow.max('polarity')
            logger.info('Iteration=%i, Time=%e, dt=%e, Ekin=%f, Ekin/time=%f, Diss_E =%f, m=%f' %(solver.iteration, solver.sim_time, timestep, avgEkin, avgEkin/solver.sim_time, diss_E, m))
 
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

