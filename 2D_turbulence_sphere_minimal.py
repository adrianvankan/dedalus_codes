import numpy as np
import dedalus.public as d3
import logging
import scipy
import dedalus.extras.flow_tools as flow_tools
from mpi4py import MPI
import dedalus.tools.logging as mpi_logging

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)
figkw = {'figsize':(6,4), 'dpi':100}

# Parameters
Nphi      = 256
Ntheta    = 256
dealias   = 3/2
R         = 1.0       #Sphere radius
nu        = 1.0e-7    #Viscosity
alpha     = 0.1       #Rayleigh friction coefficient
timestep  = 0.001     #initial timestep
stop_sim_time = 25.0  #stop time
dtype     = np.float64
eps       = 1.0 #energy injection rate
seed0     = 10

#Timestepper settings
timestepper   = d3.RK443

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist   = d3.Distributor(coords, dtype=dtype)
basis  = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

# Fields
u = dist.VectorField(coords, name='u',   bases=basis)
f = dist.VectorField(coords, name='f',   bases=basis)
p = dist.Field(      name='p',   bases=basis)
psi_f = dist.Field(      name='psi_f',   bases=basis)
psi   = dist.Field(      name='psi',   bases=basis)
tau_p = dist.Field(  name='tau_p')
const = dist.Field(  name='const', bases = basis)
g     = dist.Field(      name='g',   bases=basis)

# Coordinates
phi, theta = dist.local_grids(basis)
lat = np.pi / 2 - theta + 0*phi

# Substitutions
zcross = lambda A: d3.MulCosine(d3.skew(A))

# Problem
problem = d3.IVP([u, p, tau_p], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(u) + grad(p) + nu*lap(lap(u)) + alpha*u = - u@grad(u) + d3.skew(d3.grad(psi_f))")
problem.add_equation("ave(p) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.05, max_writes=100)
snapshots.add_task(p, name='pressure')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

# Scalar Data
analysis1 = solver.evaluator.add_file_handler("scalar_data", sim_dt=0.005)
analysis1.add_task(d3.Average(0.5*(u@u),coords), name="Ek")
snapshots.add_task(d3.Average((-d3.div(d3.skew(u)))**2, coords), name='Enstrophy')

# Flow properties
flow_prop_cad = 1
flow = d3.GlobalFlowProperty(solver, cadence = 10)
flow.add_property(d3.Average(0.5*(u@u),coords), name = 'avgEkin')
flow.add_property(d3.Average(g,coords), name='avg_g')
flow.add_property(d3.Average(nu*u@d3.lap(d3.lap(u)),coords), name='diss')

#Initial condition (spherical harmonic |l| <= m)
l = 5
m = 5
amp = 0
psi['g'] = amp*np.real(scipy.special.sph_harm(l,m,phi,theta))

# Initial conditions: incompressible velocity field
problem2 = d3.LBVP([u], namespace=locals())
problem2.add_equation("u = d3.skew(d3.grad(psi)) ")
solver2 = problem2.build_solver()
solver2.solve()

#GlobalArrayReducer
reducer   = flow_tools.GlobalArrayReducer(comm=MPI.COMM_WORLD)

# CFL
max_timestep = 0.01
CFL = d3.CFL(solver, initial_dt=timestep, cadence=10, safety=1.0, threshold=0.5, max_change=5, min_change=0.2, max_dt=max_timestep)
CFL.add_velocity(u)

# Main loop
cnt = 0;

# Initialise random seed to seed0
np.random.seed(seed0)


# Specify forcing range: azimuthal angular WN, i.e. |l| <= meridional angular WN, i.e. m
lm_vec = []
ms = [11,12]
lmin = 1
for m in ms:
   for l in range(lmin,m+1):
      lm_vec.append([l,m])

try:
    logger.info('Starting main loop')
    while solver.proceed:
        if cnt >= 1: #skip first time step needed for initialising all arrays to correct size
         phi,theta = basis.local_grids((dealias,dealias));
         lat = np.pi / 2 - theta + 0*phi

         psi_f['g'] = np.zeros_like(psi_f['g'])
         for lm in lm_vec:
            l,m = lm
            phase = 2*np.pi*np.random.rand(1)
            psi_f['g'] += 1/(1+m)/l*np.real(np.exp(1.0j*phase)*scipy.special.sph_harm(l,m,phi,theta))

         f  = d3.skew(d3.grad(psi_f)).evaluate()
         g['g'] = np.cos(lat)
         fvar = reducer.global_mean(g['g']*(f['g'][0]**2+f['g'][1]**2))/reducer.global_mean(g['g'])

         psi_f['g'] = psi_f['g']/np.sqrt(fvar) * np.sqrt(2*eps/timestep)

        timestep = CFL.compute_timestep()

        solver.step(timestep)
        if (solver.iteration) % 10 == 0:
            avgEkin  = flow.max('avgEkin')
            diss     = flow.max('diss')
            logger.info('Iteration=%i, Time=%e, dt=%e, Ekin=%f, Ekin/time=%f, Diss. rate=%f'  %(solver.iteration, solver.sim_time, timestep,  avgEkin, avgEkin/solver.sim_time, diss))
        cnt += 1

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

