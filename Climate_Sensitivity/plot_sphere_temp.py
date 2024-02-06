"""
Plot sphere outputs.

Usage:
    plot_sphere.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def build_s2_coord_vertices(phi, theta):
    phi = phi.ravel()
    phi_vert = np.concatenate([phi, [2*np.pi]])
    phi_vert -= phi_vert[1] / 2
    theta = theta.ravel()
    theta_mid = (theta[:-1] + theta[1:]) / 2
    theta_vert = np.concatenate([[np.pi], theta_mid, [0]])
    return np.meshgrid(phi_vert, theta_vert, indexing='ij')


def main(filename, start, count, output):
   #CONSTANTS
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
    theta_c    = np.arccos(T_c**4)
    print(theta_c)
    Ntheta     = 144

    """Save plot of specified tasks for given range of analysis writes."""
    # Plot settings
    task = 'Temperature'
    cmap = plt.cm.RdBu_r
    dpi = 100
    figsize = (8, 8)
    savename_func = lambda write: 'temp_write_{:06}.png'.format(write)
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        dset = file['tasks'][task]
        phi = dset.dims[1][0][:].ravel()
        theta = dset.dims[2][0][:].ravel()
        phi_vert, theta_vert = build_s2_coord_vertices(phi, theta)
        lat_vec    = np.linspace(-np.pi/2,np.pi,Ntheta)
        iplus = int(3/2*np.abs(lat_vec - theta_c).argmin())
        imin  = int(3/2*np.abs(lat_vec + theta_c).argmin())
        x = np.sin(theta_vert) * np.cos(phi_vert)
        y = np.sin(theta_vert) * np.sin(phi_vert)
        z = np.cos(theta_vert)
        for index in range(start, start+count):
            #plt.title('$t$='+str(0.1*index))
            data_slices = (index, slice(None), slice(None))
            data = dset[data_slices]
            clim = np.max(np.abs(data))
            norm = matplotlib.colors.Normalize(2*clim/3, clim)
            fc = cmap(norm(data))
            fc[:, theta.size//2, :] = [0,0,0,1]  # black equator
            fc[:,iplus, :]          = [0,0,1,1]  # northern end of convective zone
            fc[:,imin, :]           = [0,0,1,1]  # northern end of convective zone

            if index == start:
                surf = ax.plot_surface(x, y, z, facecolors=fc, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False, zorder=5)
                ax.set_box_aspect((1,1,1))
                ax.set_xlim(-0.7, 0.7)
                ax.set_ylim(-0.7, 0.7)
                ax.set_zlim(-0.7, 0.7)
                ax.axis('off')
            else:
                surf.set_facecolors(fc.reshape(fc.size//4, 4))
            
            #plt.clf()
            time = float(0.1*index)
            fig.suptitle('Temperature at $t$='+'%.2f' % time, fontsize=18)
            #plt.title('$t$='+str(0.05*index))
            #plt.tight_layout()
            
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            plt.title('')
    plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)

