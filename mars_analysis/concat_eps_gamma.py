# %%
import xarray as xr
import numpy as np
import sys, os

sys.path.append('../')

from atmospy import open_files, stereo_plot

import string

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import (cm, colors, cycler)
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


path = '/user/work/xz19136/Isca_data/'
theta, center, radius, verts, circle = stereo_plot()
theta0 = 200.
kappa = 1/4.0

if plt.rcParams["text.usetex"]:
    fmt = r'%r \%'
else:
    fmt = '%r'

def concat_parameter(isentropic = False):
    g_keff = []
    for gamma in [0.093, 0.00]:
        e_keff = []
        for eps in [10,15,20,25,30,35,40,45,50]:#,30,35,40,45,50]:
            exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps, gamma)
            print(exp_name)
            ds, d = open_files(path,exp_name, isentropic)
            ds = ds.interp({'new':d.lat.values})
            ds['epsilon'] = eps
            
            e_keff.append(ds)
        
        e_keff = xr.concat(e_keff, dim='epsilon')
        e_keff['gamma'] = gamma
        g_keff.append(e_keff)


    g_keff = xr.concat(g_keff, dim='gamma')
    if isentropic:
        ise = '_isentropic'
    else:
        ise = ''
    g_keff.to_netcdf(path+'mars_analysis/parameter_keff_test_tracer%s.nc' %ise, mode='w')

def concat_attribution(isentropic = False):
    c = []
    for t in ['', '_mola_topo']:
        b = []
        for dust in ['', '_cdod_clim_scenario_7.4e-05']:
            a = []
            for l in ['', '_lh']:
                exp_name = 'tracer_soc_mars%s%s_eps_25_gamma_0.093%s' % (t, l, dust)
                print(exp_name)
                ds, d = open_files(path, exp_name, isentropic)
                ds = ds.interp({'new':d.lat.values})
                if l == '_lh':
                    ds['lh'] = 1
                else:
                    ds['lh'] = 0

                a.append(ds)
            a = xr.concat(a, dim='lh')
            if dust == '':
                a['dust'] = 0
            else:
                a['dust'] = 1
            b.append(a)
        b = xr.concat(b, dim = "dust")
        if t == '':
            b['topo'] = 0
        else:
            b['topo'] = 1
        c.append(b)
    
    c = xr.concat(c, dim = "topo")

    if isentropic:
        ise = '_isentropic'
    else:
        ise = ''
    c.to_netcdf(path+'mars_analysis/attribution_keff_test_tracer%s.nc' %ise, mode='w')

def concat_dust(isentropic = False):
    
    a = []
    for dust in [3.7e-5, 7.4e-5,1.48e-4,2.96e-4]:
        exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                    '25_gamma_0.093_cdod_clim_scenario_%s' % str(dust)

        print(exp_name)
        ds, d = open_files(path, exp_name, isentropic)
        ds = ds.interp({'new':d.lat.values})
        ds['dust_scale'] = dust
        a.append(ds)
    a = xr.concat(a, dim="dust_scale")
    if isentropic:
        ise = '_isentropic'
    else:
        ise = ''
    a.to_netcdf(path+'mars_analysis/dust_keff_test_tracer%s.nc' %ise, mode='w')

if __name__ == "__main__":
    isentropic = True
    concat_dust(isentropic=isentropic)
    
    concat_parameter(isentropic=isentropic)
    concat_attribution(isentropic=isentropic)

            #for t in ['test_tracer']:#, 'PV']:
# %%
