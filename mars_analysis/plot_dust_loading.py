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

figpath = '/user/home/xz19136/Figures/'
path = '/user/work/xz19136/Isca_data/'
theta, center, radius, verts, circle = atmospy.stereo_plot()
theta0 = 200.
kappa = 1/4.0

if plt.rcParams["text.usetex"]:
    fmt = r'%r \%'
else:
    fmt = '%r'



def plot_dust_lat():
    dclim = xr.open_dataset(
        '/user/home/xz19136/dust_clim.nc',
        decode_times = False)
    dclim = dclim.mean(dim="longitude")
    dMY28 = xr.open_dataset(
        '/user/home/xz19136/dustscenario_MY28_v2-0.nc',
        decode_times = False)
    dMY28 = dMY28.mean(dim="longitude")
    
    fig, axs = plt.subplots(nrows=1,ncols=3, figsize = (12,4),)
    
    lims = [0,2]

    boundaries, cmap, norm = atmospy.new_cmap(lims, extend='max', i = 10, override=True, cols='OrRd')
    for i, ax in enumerate(fig.axes):
        ax.text(0, 1.05, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
            size='large')
        
        ax.set_ylim([-90,90])
        ax.set_xlabel('Sol')

    axs[0].set_title('$\lambda = 1$')
    axs[0].set_ylabel('Latitude ($^\circ$N)')
    axs[1].set_title('$\lambda = 4$')
    axs[1].set_yticklabels([])
    axs[2].set_title('MY28')
    axs[2].set_yticklabels([])

    my28winter = dMY28.cdodtot.isel(Time=slice(400,600)).where(dMY28.latitude>-60,
            drop=True).where(dMY28.latitude<60,drop=True).max(dim={"Time","latitude"})
    climwinter = dclim.cdod.isel(Time=slice(400,600)).where(dclim.latitude>-60,
            drop=True).where(dclim.latitude<60,drop=True).max(dim={"Time","latitude"})
    print(my28winter.values, climwinter.values)
    for i in [1/2,1,2,4,8]:
        print(i*climwinter.values)

    my28winter = dMY28.cdodtot.isel(Time=slice(100,300)).where(dMY28.latitude>-60,
            drop=True).where(dMY28.latitude<60,drop=True).max(dim={"Time","latitude"})
    climwinter = dclim.cdod.isel(Time=slice(100,300)).where(dclim.latitude>-60,
            drop=True).where(dclim.latitude<60,drop=True).max(dim={"Time","latitude"})
    for i in [1/2,1,2,4,8]:
        print(i*climwinter.values)

    c1=axs[0].contourf(dclim.Time, dclim.latitude, dclim.cdod.squeeze().transpose(),
                cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])
    c2=axs[1].contourf(dclim.Time, dclim.latitude, 4*dclim.cdod.squeeze().transpose(),
                cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])
    c3=axs[2].contourf(dMY28.Time, dMY28.latitude, dMY28.cdodtot.squeeze().transpose(),
                cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ticks=boundaries,
        ax = axs, label='CDOD', extend='max')

def plot_dust_lat_bad():
    dclim = xr.open_dataset(
        '/user/home/xz19136/Isca/exp/socrates_mars/input/cdod_clim_scenario.nc',
        decode_times = False)
    dclim = dclim.isel(time=slice(0,669))
    dMY28 = xr.open_dataset(
        '/user/home/xz19136/Isca/exp/socrates_mars/input/cdod_clim_MY28.nc',
        decode_times = False)
    dMY28 = dMY28.isel(time=slice(0,669))
    
    fig, axs = plt.subplots(nrows=1,ncols=3, figsize = (10,4),)

    lims = [0,1]

    boundaries, cmap, norm = atmospy.new_cmap(lims, extend='max', i = 10, override=True, cols='OrRd')
    for i, ax in enumerate(fig.axes):
        ax.text(0, 1.05, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
            size='large')
        
        ax.set_ylim([-90,90])
        ax.set_xlabel('Sol')

    axs[0].set_title('$\lambda = 1$')
    axs[1].set_title('$\lambda = 4$')
    axs[2].set_title('MY28')

    c1=axs[0].contourf(dclim.time, dclim.lat, dclim.cdod.squeeze().transpose(),
                cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])
    c2=axs[1].contourf(dclim.time, dclim.lat, 4*dclim.cdod.squeeze().transpose(),
                cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])
    c3=axs[2].contourf(dMY28.time, dMY28.lat, dMY28.cdod.squeeze().transpose(),
                cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])
if __name__ == "__main__":
    plot_dust_lat()
# %%
