# %%
import xarray as xr
import numpy as np
import sys, os
import math

sys.path.append('/user/home/xz19136/Py_Scripts/atmospy/')

import analysis_functions as funcs
from test_tracer_plot import open_files
from plot_keff_cross_sections import get_exps
import string

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import (cm, colors, cycler)
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


path = '/user/work/xz19136/Isca_data/'
theta, center, radius, verts, circle = funcs.stereo_plot()
theta0 = 200.
kappa = 1/4.0

if plt.rcParams["text.usetex"]:
    fmt = r'%r \%'
else:
    fmt = '%r'

def get_PV_lats_isentropic(di, hem='nh'):
    '''
    Lait-scale PV and then return the latitude of maximum PV on given 
    pressure levels'''
    laitPV = funcs.lait(di.PV,di.level,theta0,kappa=kappa)

    l = []
    s = []
    for a in range(len(di.time)):
        try:
            x = laitPV.isel(time=a)
        
            x = x.where(x != np.nan, drop = True)
            if hem == 'nh':
                x = x.sortby('lat',ascending=True)
                phi_PV, PV_max = funcs.calc_jet_lat(x, x.lat)
            else:
                x = x.sortby('lat',ascending=False)
                phi_PV, PV_max = funcs.calc_jet_lat(-x, x.lat)                    
            l.append(phi_PV)
            s.append(PV_max)
        except:
            l.append(np.nan)
            s.append(np.nan)
            #l.append(x.lat[np.argmax(np.abs(x.lat))])
        if l[-1] == np.nan:
            l[-1] = x.lat[np.argmax(np.abs(x.lat))]
            s[-1] = x[np.argmax(np.abs(x.lat))]
    return l, s

def get_PV_max_isentropic(di, hem='nh'):
    '''
    Lait-scale PV and then return the latitude of maximum PV on given 
    pressure levels'''
    laitPV = funcs.lait(di.PV,di.level,theta0,kappa=kappa)

    l = []
    s = []
    for a in range(len(di.time)):
        try:
            x = laitPV.isel(time=a)
        
            x = x.where(x != np.nan, drop = True)
            if hem == 'nh':
                phi_PV, PV_max = funcs.calc_jet_lat(x, x.lat)
            else:
                phi_PV, PV_max = funcs.calc_jet_lat(-x, x.lat)                    
            l.append(phi_PV)
            s.append(PV_max)
        except:
            l.append(np.nan)
            s.append(np.nan)
            #l.append(x.lat[np.argmax(np.abs(x.lat))])
        if l[-1] == np.nan:
            l[-1] = x.lat[np.argmax(np.abs(x.lat))]
            s[-1] = x[np.argmax(np.abs(x.lat))]
    return l, s


def plot_PV_max_evolution(exps=['curr-ecc','0-ecc','dust'], \
    smooth=None,level=300,ext='png'):
    '''
    Plot effective diffusivity evolution at a given point,
    in order to understand strength of the transport barrier and mixing
    within the vortex.
    '''
    
    fig, axs = plt.subplots(nrows=len(exps),ncols=2, figsize = (15,4*len(exps)),dpi=300)
    for i in range(len(exps)):
        exp = exps[i]
        exp_names, titles, _, _ = get_exps(exp)
        
        colors = plt.cm.viridis(np.linspace(0,1,int(len(exp_names)-1)))
        if exp == 'curr-ecc':
            exp = '$\gamma = 0.093$'
        elif exp == '0-ecc':
            exp = '$\gamma = 0.000$'
        elif exp == 'dust':
            exp = 'Dust Scale'
        elif exp == 'attribution':
            exp = 'Attribution'
        l = 0
        for j in range(len(exp_names)):

            exp_name = exp_names[j]
            print(exp_name)
            
            ds = xr.open_dataset(path+exp_name+'/atmos_isentropic.nc', decode_times=False)

            ds = ds[["PV","mars_solar_long"]].sel(level=level,method="nearest")
            ds = ds.mean(dim="lon")

            _, PV_max_n = get_PV_lats_isentropic(ds.where(ds.lat > 0, drop=True),hem='nh')
            _, PV_max_s = get_PV_lats_isentropic(ds.where(ds.lat < 0, drop=True),hem='sh')

            if smooth is not None:
                time = funcs.moving_average(ds.time, smooth)
                PV_max_n = funcs.moving_average(PV_max_n, smooth)
                PV_max_s = funcs.moving_average(PV_max_s, smooth)
                ls = funcs.moving_average(ds.mars_solar_long, smooth)
            else:
                time = ds.time
                ls = ds.mars_solar_long

            PV_max_n =  PV_max_n*10**4
            PV_max_s = -PV_max_s*10**4

            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05':
                col = 'k'
                lnstl = '--'
            else:
                lnstl = '-'
                col = colors[l]
                l += 1
            ax = axs[i,0]
            ax.plot(time, PV_max_n,label=titles[j],color=col,linestyle=lnstl)
            ax.set_ylim([3,11])
            ax.set_xlim([time[0],time[250]])
            ax.text(
                    -0.16, 0.5, exp,
                    ha='right',
                    va='center',
                    transform=ax.transAxes,
                    rotation='vertical',
                    fontsize='large',
            )
            ax = axs[i,1]
            ax.plot(time, PV_max_s,label=titles[j],color=col,linestyle=lnstl)
            ax.set_ylim([-2,-10])
            if exp == 'Attribution':
                ax.set_ylim([-2,-12])
            ax.set_xlim([time[300],time[-50]])

        for i, ax in enumerate(fig.axes):
            ax.text(0.0, 1.03, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
                size='large')
            xlocs = [i for i in ax.get_xticks()]
            if smooth is not None:
                lsi = np.interp(xlocs, time, ls)
            else:
                lsi = ls.interp(time=xlocs,kwargs={"fill_value":"extrapolate"})#, d.time)
            if i < 2*len(exps)-2:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(['%i' % i for i in lsi])
                ax.set_xlabel('$\mathrm{L_s}$',fontsize='large')
            ax.set_ylabel('max PV (MPVU)')
            if i % 2 == 1:
                ax.legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
                 borderaxespad=0, fontsize='large')
            
            if i == 0:
                ax.set_title('NH')
            elif i == 1:
                ax.set_title('SH')



    fig.savefig('/user/home/xz19136/Figures/mars_analysis/PV/' \
                + 'PV_maxstrength.%s' %ext, dpi=300,
                bbox_inches='tight')


def plot_PV_lat_evolution(exps=['curr-ecc','0-ecc','dust'], \
    smooth=None,level=300,ext='png'):
    '''
    Plot effective diffusivity evolution at a given point,
    in order to understand strength of the transport barrier and mixing
    within the vortex.
    '''
    
    fig, axs = plt.subplots(nrows=len(exps),ncols=2, figsize = (15,4*len(exps)),dpi=300)
    for i in range(len(exps)):
        exp = exps[i]
        exp_names, titles, _, _ = get_exps(exp)
        
        colors = plt.cm.viridis(np.linspace(0,1,int(len(exp_names)-1)))
        if exp == 'curr-ecc':
            exp = '$\gamma = 0.093$'
        elif exp == '0-ecc':
            exp = '$\gamma = 0.000$'
        elif exp == 'dust':
            exp = 'Dust Scale'
        elif exp == 'attribution':
            exp = 'Attribution'
        l = 0
        for j in range(len(exp_names)):

            exp_name = exp_names[j]
            print(exp_name)
            
            ds = xr.open_dataset(path+exp_name+'/atmos_isentropic.nc', decode_times=False)

            ds = ds[["PV","mars_solar_long"]].sel(level=level,method="nearest")
            ds = ds.mean(dim="lon")

            phiPV_n, _ = get_PV_lats_isentropic(ds.where(ds.lat > 0, drop=True),hem='nh')
            phiPV_s, _ = get_PV_lats_isentropic(ds.where(ds.lat < 0, drop=True),hem='sh')

            if smooth is not None:
                time = funcs.moving_average(ds.time, smooth)
                phiPV_n = funcs.moving_average(phiPV_n, smooth)
                phiPV_s = funcs.moving_average(phiPV_s, smooth)
                ls = funcs.moving_average(ds.mars_solar_long, smooth)
            else:
                time = ds.time
                ls = ds.mars_solar_long
            
            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05':
                col = 'k'
                lnstl = '--'
            else:
                lnstl = '-'
                col = colors[l]
                l += 1
            ax = axs[i,0]
            ax.plot(time, phiPV_n,label=titles[j],color=col,linestyle=lnstl)
            ax.set_ylim([60,90])
            ax.set_xlim([time[0],time[250]])
            ax.text(
                    -0.16, 0.5, exp,
                    ha='right',
                    va='center',
                    transform=ax.transAxes,
                    rotation='vertical',
                    fontsize='large',
            )
            ax = axs[i,1]
            ax.plot(time, phiPV_s,label=titles[j],color=col,linestyle=lnstl)
            ax.set_ylim([-60,-90])
            ax.set_xlim([time[300],time[-50]])

        for i, ax in enumerate(fig.axes):
            ax.text(0.0, 1.03, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
                size='large')
            xlocs = [i for i in ax.get_xticks()]
            if smooth is not None:
                lsi = np.interp(xlocs, time, ls)
            else:
                lsi = ls.interp(time=xlocs,kwargs={"fill_value":"extrapolate"})#, d.time)
            if i < 2*len(exps)-2:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(['%i' % i for i in lsi])
                ax.set_xlabel('$\mathrm{L_s}$',fontsize='large')
            ax.set_ylabel('latitude ($^\circ$N)')
            if i % 2 == 1:
                ax.legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
                 borderaxespad=0, fontsize='large')
            
            if i == 0:
                ax.set_title('NH')
            elif i == 1:
                ax.set_title('SH')



    fig.savefig('/user/home/xz19136/Figures/mars_analysis/PV/' \
                + 'PV_maxlat.%s' %ext, dpi=300,
                bbox_inches='tight')


if __name__ == "__main__":
    eps = np.arange(10,55,5)
    gamma = [0.093,0.00]
    plot_PV_lat_evolution(exps=['curr-ecc','0-ecc','dust'],smooth=10,ext='pdf')
    plot_PV_max_evolution(exps=['curr-ecc','0-ecc','dust'],smooth=10,ext='pdf')
# %%
