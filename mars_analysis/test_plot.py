# %%
import xarray as xr
import numpy as np
import sys, os

sys.path.append('../')

from atmospy import stereo_plot, lait, calc_PV_max, new_cmap, \
                    get_timeslice, nf, get_exps

import string

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import (cm, colors, cycler)
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

figpath = '/user/home/xz19136/Figures/mars_analysis/xsections/'
path = '/user/work/xz19136/Isca_data/'
theta, center, radius, verts, circle = stereo_plot()
theta0 = 200.
kappa = 1/4.0

eps = np.arange(10,55,5)
gamma = [0.093,0.00]

if plt.rcParams["text.usetex"]:
    fmt = r'%r \%'
else:
    fmt = '%r'


d  = xr.open_mfdataset(path+'soc_mars_mola_topo_lh_eps_25' + \
    '_gamma_0.093_clim_latlon_7.4e-05/run0015/atmos_monthly.nc', decode_times=False)

ds = xr.open_mfdataset(path+'soc_mars_mola_topo_lh_eps_25' + \
    '_gamma_0.093_cdod_clim_scenario_7.4e-05/run0015/atmos_monthly.nc', decode_times=False)

print(d)
print(ds)
c = plt.contourf(d.temp.sel(pfull=0.5,method="nearest").squeeze()-ds.temp.sel(pfull=0.5,method="nearest").squeeze())
plt.colorbar(c)
# %%
