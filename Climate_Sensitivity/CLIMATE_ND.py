import numpy as np
import dedalus.public as d3
import logging
import scipy
import dedalus.extras.flow_tools as flow_tools
from mpi4py import MPI
import dedalus.tools.logging as mpi_logging
import matplotlib.pyplot as plt
from time import time as timer
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)
figkw = {'figsize':(6,4), 'dpi':100}

###### Numerical Parameters ####
Nphi      = 128       # Number of l modes
Ntheta    = 64        # Number of m modes
dealias   = 3/2
R         = 1.0        #Sphere radius (non-dim)
timestep  = 1.0        #initial timestep
stop_sim_time = 10     #stop time
dtype     = np.float64
seed0     = 1
max_timestep = 1e-3

restart = False
cp_path    = 'checkpoints/checkpoints_s3.h5'

##### Dimensional Parameters ####
Rearth    = 6.371e6       #Radius of the Earth in [m]
nu_E      = 1e-10#1.0e-7  #Eddy Viscosity
kap_E     = 10*nu_E      #Eddy Diffusivity
nu_R      = 5e-6          #Rayleigh friction coefficient

C         = 1.5*10**5     #**8     #Ocean mixed layer heat capacity in J/K per m² of surface area
sig_sb    = 5.67*10**(-8) #Stefan-Boltzmann's constant
rho_a     = 10**4         #Air mass in kg per m² of surface area of Earth
eps       = 0.591         #Emissivity
alpha_c   = 0.7           #Globally averaged co-albedo of Earth
S_sun     = 1360          #Solar irradiance

#### NONDIMENSIONAL GROUPS #####
F0         = S_sun*alpha_c/np.pi
T0         = (F0/eps/sig_sb)**(1/4)

###### Convection Scheme Parameters #######
T_c_dim    = 301 #temperature threshold in K 
T_c        = T_c_dim/T0
print('T_c =',T_c)
tau_c      = 0.1 #time after which the convective relaxation is done
d_ang      = 2 * np.pi / 10 #34
nbox_phi, nbox_theta  = int(2*np.pi/d_ang), int(np.pi/d_ang)
epsilon_f_arr  = np.zeros((nbox_phi, nbox_theta))
phi_mid_vec   = (np.arange(nbox_phi)   + 0.5) * d_ang
theta_mid_vec = (np.arange(nbox_theta) + 0.5) * d_ang
#lat_mid_vec   = np.pi/2 - theta_mid_vec
#lat_min_vec   = lat_mid_vec - d_ang/2
#lat_max_vec   = lat_mid_vec + d_ang/2
#phi_min_vec   = phi_mid_vec - d_ang/2
#phi_max_vec   = phi_mid_vec + d_ang/2
#print('phi_mid_vec = '+str(phi_mid_vec))
#print('theta_mid_vec = '+str(theta_mid_vec))

# Specify momentum forcing range: azimuthal angular WN m such that |m| <= meridional angular WN  l
ml_vec = []
ls = [nbox_theta]
mmin = 0  #max(ms)-2
for l in ls:
   for m in range(mmin,l+1):
      ml_vec.append([m,l])

theta_c    = np.arccos(T_c**4)
fraction   = 0.5*(theta_c - np.sin(theta_c)*np.cos(theta_c))
inv_frac   = 1/fraction
print('fraction=',fraction)
F1         = F0 * fraction
u0         = np.sqrt(F1/rho_a/nu_R)
ta         = Rearth/u0           # checked that this = np.sqrt(Rearth**2 * rho * nu_R / F0)
ta_by_trad = F1*ta/(C*T0)
ta_by_trad_mod = ta_by_trad * inv_frac
ta_by_tfr  = F1*ta/(rho_a*u0**2) # checked that this = ta*nu_R 
thermal_by_kinetic = C*T0/(rho_a*u0**2)
print('ta_by_trad =',ta_by_trad)
#print('ta=',ta,np.sqrt(Rearth**2 * rho_a * nu_R / F1))
print('ta_by_tfr =',ta_by_tfr,ta*nu_R)

#Timestepper settings
timestepper = d3.RK222  

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist   = d3.Distributor(coords, dtype=dtype)
basis  = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

# Fields
u = dist.VectorField(coords, name='u',   bases=basis)
f = dist.VectorField(coords, name='f',   bases=basis)
f_temp = dist.VectorField(coords, name='f_temp',   bases=basis)
p = dist.Field(      name='p',   bases=basis)
psi_f  = dist.Field(      name='psi_f',   bases=basis)
psi    = dist.Field(      name='psi',   bases=basis)
tau_p  = dist.Field(  name='tau_p')
const  = dist.Field(  name='const', bases = basis)
g      = dist.Field(      name='g',   bases=basis)
T      = dist.Field(      name='T',   bases=basis)
temp   = dist.Field(     name='temp', bases=basis)  #temporary field for computing box averages
temp2  = dist.Field(     name='temp2', bases=basis) #another temporary field for computing box averages
temp3  = dist.Field(     name='temp3', bases=basis) #yet another temporary field for building random forcing

# Coordinates
phi, theta = dist.local_grids(basis)
lat = np.pi / 2 - theta + 0*phi

#g['g'] = np.cos(lat) # Auxiliary function

# Substitutions
zcross = lambda A: d3.MulCosine(d3.skew(A))

# Problem (nondimensional)
problem = d3.IVP([u, p, T, tau_p], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(u) + ta_by_tfr * ( grad(p) + u - nu_E*lap(lap(lap(u))))  =  - u@grad(u) + d3.skew(d3.grad(psi_f))")
problem.add_equation("dt(T) -  ta_by_trad*kap_E * lap(lap(lap(T))) =  ta_by_trad_mod * g - ta_by_trad_mod *  T**4  - ta_by_trad * nu_E * u@lap(lap(lap(u))) + ta_by_trad * u@u - u@grad(T)")
problem.add_equation("ave(p) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

#Initial condition (close to radiative equilibrium)
if restart == False:
  T['g']   = np.cos(lat)**(1/4)
  T['g'][T['g']>T_c] = T_c
  file_handler_mode = 'overwrite'
  initial_timestep = max_timestep
else:
  write, initial_timestep = solver.load_state(cp_path)
  file_handler_mode = 'append'
 
# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=1000,mode=file_handler_mode)
snapshots.add_task(p, name='pressure')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task(T, name='Temperature')
checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=0.5, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)
# Scalar Data
analysis1 = solver.evaluator.add_file_handler("scalar_data", sim_dt=0.001)
analysis1.add_task(d3.Average(0.5*(u@u),coords), name="Ek")
analysis1.add_task(d3.Average((-d3.div(d3.skew(u)))**2, coords), name='Enstrophy')
analysis1.add_task(d3.Average(T,coords), name="Tmean")
analysis1.add_task(d3.Average(T,coords)+d3.Average(0.5*(u@u),coords), name="Etot")

# Flow properties
flow_prop_cad = 10
flow = d3.GlobalFlowProperty(solver, cadence = flow_prop_cad)
flow.add_property(d3.Average(0.5*(u@u),coords), name = 'avgEkin')
flow.add_property(d3.Average(g,coords), name='avg_g')
flow.add_property(d3.Average(ta_by_tfr *nu_E*u@d3.lap(d3.lap(u)),coords), name='diss_E')
flow.add_property(d3.Average(ta_by_trad*u@u,coords), name='diss_R')
flow.add_property(d3.Average(T,coords), name='glob_avg_temp')

theta_arr = theta + 0*phi
phi_arr   = phi + 0*theta
#CHECK SHAPE OF INITIALTEMPERATURE PROFILE
if False:
  print(np.shape(theta_arr),np.shape(phi_arr),np.shape(T['g']))  
  plt.plot(theta[0,:],T['g'][0,:])
  plt.contourf(theta_arr,phi_arr,T['g']); 
  plt.show()

# Initial conditions: incompressible velocity field
#problem2 = d3.LBVP([u], namespace=locals())
#problem2.add_equation("u = d3.skew(d3.grad(psi)) ")
#solver2 = problem2.build_solver()
#solver2.solve()

#GlobalArrayReducer
reducer   = flow_tools.GlobalArrayReducer(comm=MPI.COMM_WORLD)

# CFL
CFL = d3.CFL(solver, initial_dt=initial_timestep, cadence=2, safety=0.8, threshold=0.8, max_change=100000, min_change=0.00001, max_dt=max_timestep)
CFL.add_velocity(u)

# Main loop
cnt_conv   = 0   #counts number of convection steps

# Initialise random seed to seed0
np.random.seed(seed0)

#set MPI rank and size
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

conds     = []# np.empty((nbox_phi,nbox_theta))
not_conds = []# np.empty((nbox_phi,nbox_theta))
envelopes = []
box_areas = []
#lat_mids  = []; lat_mins  = []; lat_maxs = []
#phi_mins  = []; phi_maxs   = [];
#############
# MAIN LOOP #
#############
try:
    logger.info('Starting main loop')
    while solver.proceed:
        switch = False
        if 1:          
          #IN FIRST TIMESTEPENSURE ARRAYS HAVE CORRECT SIZE AND PRECOMPUTE CONSTANT ARRAYS in first timestep
          k = len(psi_f['g'])
          if k == Nphi:
            d3.Average(psi_f).evaluate() 
            d3.Average(g).evaluate()
            d3.Average(f_temp@f_temp).evaluate()
            d3.Average(temp).evaluate()
            d3.Average(temp2).evaluate()
            d3.Average(temp3).evaluate()
            phi,theta = basis.local_grids(dist=dist,scales=(dealias,dealias));
            phi_mat   = phi   + 0*theta
            theta_mat = theta + 0*phi
            lat = np.pi / 2 - theta + 0*phi
            g['g'] = np.cos(lat)
            grid_mean_cos_lat = reducer.global_mean(g['g'])
           
            for iphi, phi_mid in enumerate(phi_mid_vec):
              for itheta, theta_mid in enumerate(theta_mid_vec):
                lat_mid   =  np.pi/2 - theta_mid; #lat_mids.append(lat_mid)
                lat_min   =  lat_mid - d_ang/2;   #lat_mins.append(lat_min)
                lat_max   =  lat_mid + d_ang/2;   #lat_maxs.append(lat_max)
                phi_min   =  phi_mid - d_ang/2;   #phi_mins.append(phi_mins)
                phi_max   =  phi_mid + d_ang/2;   #phi_maxs.append(phi_max)

                #Boolean array to identify current box (in argument)
                cond = np.zeros_like(lat,dtype=bool)
                cond[((lat >= lat_min) & (lat <= lat_max) & (phi_mat >= phi_min) & (phi_mat <= phi_max))] = True
                conds.append(np.array(cond))
                #Boolean arrays to identify complement of current box (in argument)
                not_cond = np.ones_like(lat,dtype=bool)
                not_cond[((lat >= lat_min) & (lat <= lat_max) & (phi_mat >= phi_min) & (phi_mat <= phi_max))] = False
                not_conds.append(np.array(not_cond))
                envelope = np.exp(- ( np.sin(lat-lat_mid) **2 /(d_ang**2 / 4) + np.sin(phi_mat-phi_mid)**2 / (2*d_ang**2 / 4)))
                envelopes.append(envelope)
                #Boolean array to identify box (with correct dimensions, unlike cond)
                temp2['g']           = np.ones_like(temp2['g'])
                temp2['g'][not_cond] = 0
                box_areas.append(reducer.global_mean(g['g']*temp2['g']))
          #conds = np.array(conds); not_conds = np.array(not_conds)
          #conds.reshape(nbox_phi,nbox_theta)
          #not_conds.reshape(nbox_phi,nbox_theta)
          psi_f['g'] = np.zeros_like(psi_f['g'])
          time = solver.sim_time
          
          ts0 = timer()
          #BUILD GLOBAL RANDOM FORCING 
          temp3['g'] = 0
          for ml in ml_vec:
            m,l = ml
            phase = 2*np.pi*np.random.rand(1)
            temp3['g'] += np.real(np.exp(1.0j*phase)*scipy.special.sph_harm(m,l,phi_mat,theta_mat)) #* envelope

          #print('build global forcing '+str(timer()-ts0))
          ts_other_than_global_forcing = timer()
          #Loop over angular boxes
          counter = 0
          for iphi, phi_mid in enumerate(phi_mid_vec):
            for itheta, theta_mid in enumerate(theta_mid_vec):
               lat_mid   =  np.pi/2 - theta_mid
               lat_min   =  lat_mid - d_ang/2
               lat_max   =  lat_mid + d_ang/2
               phi_min   =  phi_mid - d_ang/2
               phi_max   =  phi_mid + d_ang/2

               ts1 = timer()
               #Check whether it's time for convection to take place
               if time > cnt_conv * tau_c:
                 #print('convection time!')#,time,cnt_conv*tau_c)
                 epsilon_f_arr[iphi,itheta] = 0 

                 switch = True
                 #Boolean array to identify current box
                 cond = conds[counter]  
                 #cond = np.zeros_like(lat,dtype=bool)
                 #cond[((lat >= lat_min) & (lat <= lat_max) & (phi_mat >= phi_min) & (phi_mat <= phi_max))] = True
                 
                 not_cond = not_conds[counter]
                 #not_cond = np.ones_like(lat,dtype=bool)
                 #not_cond[((lat >= lat_min) & (lat <= lat_max) & (phi_mat >= phi_min) & (phi_mat <= phi_max))] = False
            
                 #temperature data restricted to the box
                 temp['g']           = T['g']
                 temp['g'][not_cond] = 0
                  
                 #Boolean array to identify box (with correct dimensions, unlike cond)
                 temp2['g']           = np.ones_like(temp2['g'])
                 temp2['g'][not_cond] = 0 
            
                 #print('defining arrays takes',str(te2-ts2))
                 #Compute area-averaged temperature in box:
                 T_box_mean = reducer.global_mean(g['g']*temp['g'])/box_areas[counter]#/reducer.global_mean(g['g']*temp2['g'])

                 #CHECK IF TEMPERATURE IN GIVEN BOX EXCEEDS T_c  --> Set T -> T_c where threshold exceeded
                 condTgtTc = np.zeros_like(lat,dtype=bool)
                 condTgtTc[((lat >= lat_min) & (lat <= lat_max) & (phi_mat >= phi_min) & (phi_mat <= phi_max) & (T['g']>T_c))] = True
                 condTleqTc = np.zeros_like(lat,dtype=bool)
                 condTleqTc[((lat >= lat_min) & (lat <= lat_max) & (phi_mat >= phi_min) & (phi_mat <= phi_max) & (T['g']<=T_c))] = True

                 if T_box_mean > T_c:
                     #print('box mean temp='+str(T_box_mean))
                     #COMPUTE ENERGY INJECTION RATE AS ENERGY RELEASED IN CONVECTION DIVIDED BY CONVECTIVE TIME
                     temp['g'][condTleqTc] = T_c
                     #deltaH       = reducer.global_mean((temp['g'] - T_c*temp2['g'])*g['g']) / grid_mean_cos_lat #reducer.global_mean(g['g']) #difference in total thermal energy density due to reset
                     deltaH        = (reducer.global_mean(temp['g'])- T_c*box_areas[counter]) / grid_mean_cos_lat
                     epsilon_f_arr[iphi,itheta] = thermal_by_kinetic * deltaH / tau_c  #energy injection rate in this box
                     #print('eps = '+str(thermal_by_kinetic*deltaH / tau_c),' iphi= ', iphi)

                     T['g'][condTgtTc] = T_c   #only now, actually reset temperature to threshold where it has been exceeded
              
               #print('DBG length of loop', timer()-ts1)
               # Compute forcing localised to given box
               temp['g'] = 0 #use as temporary stream function below
               
               if epsilon_f_arr[iphi,itheta] > 1e-5:
                 #CONVOLVE GLOBAL FORCING WITH ENVELOPE
                 ts2 = timer()
                 envelope = envelopes[counter]#np.exp(- ( np.sin(lat-lat_mid) **2 /(d_ang**2 / 4) + np.sin(phi_mat-phi_mid)**2 / (2*d_ang**2 / 4)))
                 #print('DBG read envelopes '+str(timer()-ts2)); ts3 = timer()
                 temp['g'] = envelope * temp3['g']
                 #print('DBG multiply envelope * random forcing '+str(timer()-ts3)); ts4 = timer()
                 f_temp  = d3.skew(d3.grad(temp)).evaluate()
                 #print('DBG generate vectorial force '+str(timer()-ts4)); 
                 #abc = d3.Average(f_temp@f_temp,coords).evaluate()
                 ts5 = timer()
                 f_temp_var = reducer.global_mean(g['g']*(f_temp['g'][0]**2 + f_temp['g'][1]**2))/ grid_mean_cos_lat #reducer.global_mean(g['g'])
                 #print('DBG normalise forcing '+str(timer()-ts5)); ts6 = timer()
                 temp['g'] = temp['g'] / np.sqrt(f_temp_var) * np.sqrt(2*epsilon_f_arr[iphi,itheta]/timestep)
                 psi_f['g'] += temp['g']
                 #print('DBG generate streamfunction '+str(timer()-ts6))
               #COUNT THROUGH CONVECTION BOXES
               counter += 1 

               #print(timer() - ts)
          
          #if convective event has occurred, increase counter by 1 before continuing
          if switch == True: cnt_conv += 1
          
          print('length of loop of boxes'+str(timer() - ts_other_than_global_forcing))
          ts7 = timer()
          #compute forcing from superposition of forcing function
          f  = d3.skew(d3.grad(psi_f)).evaluate()
          
        timestep = CFL.compute_timestep()
        #print('define forcing and CFL',str(timer()-ts7))
        ts8 = timer()
        solver.step(timestep)
        #print('timestepper'+str(timer()-ts8))
        print('all apart from forcing definition '+str(timer()-ts_other_than_global_forcing))
        if (solver.iteration) % flow_prop_cad == 0:
            glob_avg_temp = flow.max('glob_avg_temp')
            avgEkin  = flow.max('avgEkin')
            diss_E     = flow.max('diss_E')
            diss_R     = flow.max('diss_R')
            logger.info('Iteration=%i, Time=%e, dt=%e, Ekin=%f, Ekin/time=%f, Diss_E =%f, Diss_R=%f, <T>_glob=%f'  %(solver.iteration, solver.sim_time, timestep,  avgEkin, avgEkin/solver.sim_time, diss_E, diss_R, glob_avg_temp))
        #cnt += 1
 
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
