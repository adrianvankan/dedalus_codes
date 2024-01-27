"""
Plot sphere outputs in the Mercator projection.

Usage:
    plot_sphere_Mercator.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""
    # Plot settings
    task = 'vorticity'
    cmap = plt.cm.RdBu_r
    dpi = 100
    figsize = (8, 8)
    savename_func = lambda write: 'write_Mercator_{:06}.png'.format(write)
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        dset = file['tasks'][task]
        phi = dset.dims[1][0][:].ravel() 
        theta = dset.dims[2][0][:].ravel()
        #print(phi)
        lat = (np.pi/2 - theta)*180/np.pi
        phi = (phi - np.pi)*180/np.pi
        #print(phi)
        phi_mat,lat_mat = np.meshgrid(phi,lat)
        for index in range(start, start+count):
            plt.clf()
            fig = plt.figure(1,figsize=figsize)
            ax = fig.add_subplot(projection=ccrs.Mercator())            
            data_slices = (index, slice(None), slice(None))
            data = dset[data_slices]
            plt.contourf(phi_mat,lat_mat,np.transpose(data),transform=ccrs.PlateCarree(),cmap=cmap,levels=50)
            ax.gridlines(draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')

            #SET THIS TO CORRECT VALUE 
            dt   = 0.25
            time = float(dt*index)
            fig.suptitle('Vorticity at $t$='+'%.2f' % time, fontsize=18)
            
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            plt.tight_layout()
            fig.savefig(str(savepath), dpi=dpi)
                 
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

