# %%
import xarray as xr
import numpy as np
import sys, os
import math

sys.path.append('../')
import atmospy
import string

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import (cm, colors, cycler)
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

figpath = '/user/home/xz19136/Figures/mars_analysis/'
path = '/user/work/xz19136/Isca_data/'
theta, center, radius, verts, circle = atmospy.stereo_plot()
theta0 = 200.
kappa = 1/4.0

if plt.rcParams["text.usetex"]:
    fmt = r'%r \%'
else:
    fmt = '%r'

# %%

if __name__ == "__main__":
    x = np.genfromtxt("INSOLN.LA2004.MARS.ASC",dtype="U12",usecols=(0,1,2),
                     converters={})
    t = []
    t2 = []
    eps = []
    gamma = []
    for i in range(len(x)):
        t.append(-float(x[i][0])*1e-3)
        t2.append(-float(x[i][0])*1e-3)
        #t2.append(-(float(x[i][0])-0.1)*1e-3)
        #t2.append(-(float(x[i][0])-0.2)*1e-3)
        #t2.append(-(float(x[i][0])-0.3)*1e-3)
        #t2.append(-(float(x[i][0])-0.4)*1e-3)
        #t2.append(-(float(x[i][0])-0.5)*1e-3)
        #t2.append(-(float(x[i][0])-0.6)*1e-3)
        #t2.append(-(float(x[i][0])-0.7)*1e-3)
        #t2.append(-(float(x[i][0])-0.8)*1e-3)
        #t2.append(-(float(x[i][0])-0.9)*1e-3)
        gamma.append(float(x[i][1]))
        eps.append(x[i][2])
    
    t2 = t2[slice(None,None,20)]
    print(t2)

    l = [np.rad2deg(float(eps[i])) for i in range(len(eps))]

    l = np.interp(t2, t, l)
    t2 = [-i for i in t2]
    d1 = xr.open_dataset('/user/work/xz19136/Isca_data/mars_analysis/summary_of_lats/curr-ecc_summary_of_lats.nc')
    d1 = d1.isel(phi=-1).squeeze()
    d1['exp_name'] = [10,15,20,25,30,35,40,45,50]
    d1 = d1.interp({'exp_name':l},method='linear',kwargs={'fill_value':'extrapolate'})

    d2 = xr.open_dataset('/user/work/xz19136/Isca_data/mars_analysis/summary_of_lats/0-ecc_summary_of_lats.nc')
    d2 = d2.isel(phi=-1).squeeze()
    d2['exp_name'] = [10,15,20,25,30,35,40,45,50]
    d2 = d2.interp({'exp_name':l},method='linear',kwargs={'fill_value':'extrapolate'})

    ### plot keff timeseries ###
    fig, axs = plt.subplots(nrows=3,ncols=1, figsize = (10,10),dpi=300)
    
    colors = plt.cm.viridis(np.linspace(0,1,4))
    matplotlib.rcParams['axes.prop_cycle'] = (
            cycler('color', colors) * \
            cycler('linestyle', ['-','--'])
        )
    
    cmap = plt.get_cmap("viridis", 3)
    norm = matplotlib.colors.BoundaryNorm(np.arange(4)+0.5,3)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # this line may be ommitted for matplotlib >= 3.1


# %%
    axs[0].plot(t2, l,linewidth=0.5, c = 'k')
    
    
    axs[1].plot(t2, d1.isel(hem=0).to_array().squeeze(),linewidth=0.5,
                c = 'xkcd:teal',label='$\gamma=0.093$')
    axs[1].plot(t2, d2.isel(hem=0).to_array().squeeze(),linewidth=0.5,
                linestyle='--',c = 'xkcd:crimson',label='$\gamma=0.000$')
    axs[1].set_ylim([1,2.9])
    axs[1].set_title('NH')
    
    axs[2].plot(t2, d1.isel(hem=1).to_array().squeeze(),linewidth=0.5,
                c = 'xkcd:teal')
    axs[2].plot(t2, d2.isel(hem=1).to_array().squeeze(),linewidth=0.5,
                linestyle='--',c = 'xkcd:crimson')
    axs[2].set_ylim([1,2.9])
    axs[2].set_title('SH')
    axs[0].set_ylabel('Obliquity, $\\varepsilon$')
    axs[1].legend()
    axs[2].set_xlabel('Time (Myr)')

    for i, ax in enumerate(fig.axes):
        ax.text(-0.01, 1.03, string.ascii_lowercase[i]+')', transform=ax.transAxes)
        ax.set_xlim([-20,0])
        if ax != axs[-1]:
            ax.set_xticklabels([])
        if ax != axs[0]:
            ax.set_ylabel('Normalized effective diffusivity')

    fig.savefig(figpath \
                + 'pld-timeseries.pdf', dpi=300,
                bbox_inches='tight')
#%%
##########################
if __name__ == "__main__":
    d1 = xr.open_dataset('/user/work/xz19136/Isca_data/mars_analysis/summary_of_lats/dust_summary_of_lats_300_temp.nc')
    d1['exp_name'] = [0,1,2,3,4]
    fig, axs = plt.subplots(nrows=2,ncols=2, figsize = (8,6),dpi=300)
    x1 = d1.exp_name.values
    y1 = d1.isel(hem=0).to_array().squeeze().values
    x2 = d1.exp_name.values
    y2 = d1.isel(hem=1).to_array().squeeze().values
    axs[1,0].plot(x1,y1,color='xkcd:teal')
    axs[1,1].plot(x2,y2,color='xkcd:teal')

    d1 = xr.open_dataset('/user/work/xz19136/Isca_data/mars_analysis/summary_of_lats/curr-ecc_summary_of_lats_300_temp.nc')
    d1['exp_name'] = [10,15,20,25,30,35,40,45,50]
    x1 = d1.exp_name.values
    y1 = d1.isel(hem=0).to_array().squeeze().values
    x2 = d1.exp_name.values
    y2 = d1.isel(hem=1).to_array().squeeze().values
    axs[0,0].plot(x1,y1,color='xkcd:teal',label='$\gamma=0.093$')
    axs[0,1].plot(x2,y2,color='xkcd:teal')
    #d1 = d1.interp({'exp_name':l},method='linear',)#kwargs={'fill_value':'extrapolate'})
#
    d2 = xr.open_dataset('/user/work/xz19136/Isca_data/mars_analysis/summary_of_lats/0-ecc_summary_of_lats_300_temp.nc')
    d2['exp_name'] = [10,15,20,25,30,35,40,45,50]
    x1 = d2.exp_name.values
    y1 = d2.isel(hem=0).to_array().squeeze().values
    x2 = d2.exp_name.values
    y2 = d2.isel(hem=1).to_array().squeeze().values
    axs[0,0].plot(x1,y1,color='xkcd:crimson',linestyle='--',label='$\gamma=0.000$')
    axs[0,1].plot(x2,y2,color='xkcd:crimson',linestyle='--')
    #d2 = d2.interp({'exp_name':l},method='linear',kwargs={'fill_value':'extrapolate'})
#
#   
    for i, ax in enumerate(fig.axes):
        ax.set_ylim([130,180])
        ax.text(-0.01, 1.03, string.ascii_lowercase[i]+')', transform=ax.transAxes)
    for i in [0,1]:
        axs[i,0].set_ylabel('Temperature (K)')
        axs[i,1].set_yticklabels([])
        axs[1,i].set_xticks([0,1,2,3,4])
        axs[0,i].set_xticks([10,15,20,25,30,35,40,45,50])
        axs[1,i].set_xticklabels(['1/2','1','2','4','8'])
        axs[1,i].set_xlabel('Dust Scale ($\lambda$)')
        axs[0,i].set_xlabel('Obliquity $\\varepsilon$ ($^\circ$)')
    axs[0,0].legend()
    axs[0,0].set_title('NH')
    axs[0,1].set_title('SH')
    fig.tight_layout()
    fig.savefig(figpath \
                + 'pld-polar_temp.pdf', dpi=300,
                bbox_inches='tight')
# %%
if __name__ == "__main__":
    ### plot temp timeseries ###
    fig, axs = plt.subplots(nrows=3,ncols=1, figsize = (10,10),dpi=300)
    
    colors = plt.cm.viridis(np.linspace(0,1,4))
    matplotlib.rcParams['axes.prop_cycle'] = (
            cycler('color', colors) * \
            cycler('linestyle', ['-','--'])
        )
    
    cmap = plt.get_cmap("viridis", 3)
    norm = matplotlib.colors.BoundaryNorm(np.arange(4)+0.5,3)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # this line may be ommitted for matplotlib >= 3.1
#
#
    
    axs[0].plot(t2, l,linewidth=0.5, c = 'k')
    
    
    axs[1].plot(t2, d1.isel(hem=0).to_array().squeeze(),linewidth=0.5,
                c = 'xkcd:teal',label='$\gamma=0.093$')
    axs[1].plot(t2, d2.isel(hem=0).to_array().squeeze(),linewidth=0.5,
                linestyle='--',c = 'xkcd:crimson',label='$\gamma=0.000$')
    #axs[1].set_ylim([1,2.9])
    axs[1].set_title('NH')
    
    axs[2].plot(t2, d1.isel(hem=1).to_array().squeeze(),linewidth=0.5,
                c = 'xkcd:teal')
    axs[2].plot(t2, d2.isel(hem=1).to_array().squeeze(),linewidth=0.5,
                linestyle='--',c = 'xkcd:crimson')
    #axs[2].set_ylim([1,2.9])
    axs[2].set_title('SH')
    axs[0].set_ylabel('Obliquity, $\\varepsilon$')
    axs[1].legend()
    axs[2].set_xlabel('Time (Myr)')
    for i, ax in enumerate(fig.axes):
        ax.text(-0.01, 1.03, string.ascii_lowercase[i]+')', transform=ax.transAxes)
        ax.set_xlim([-20,0])
        
        if ax != axs[-1]:
            ax.set_xticklabels([])
        if ax != axs[0]:
            ax.set_ylim([130,170])
            ax.set_ylabel('Temperature (K)')
    fig.savefig(figpath \
                + 'pld-timeseries_temp.pdf', dpi=300,
                bbox_inches='tight')
    #########################


# %%
