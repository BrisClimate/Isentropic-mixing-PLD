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

def get_jet_strength(di, lats):
    '''
    Lait-scale PV and then return the latitude of maximum PV on given 
    pressure levels'''
    
    l = []
    for a in range(len(di.time)):

            x = di.isel(time=a)
        
            x = np.interp(lats.isel(time=a), x.lat, x)                  
            l.append(x)
            
    return l


def plot_jet_lat_evolution(exps=['curr-ecc','0-ecc','dust'], \
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
            
            ds = xr.open_dataset(path+exp_name+'/EDJ.nc', decode_times=False)
            
            phi_n = ds.phi_n.squeeze()
            phi_s = ds.phi_s.squeeze()


            if smooth is not None:
                time = funcs.moving_average(ds.time, smooth)
                phi_n = funcs.moving_average(phi_n, smooth)
                phi_s = funcs.moving_average(phi_s, smooth)
                ls = funcs.moving_average(ds.mars_solar_long, smooth)
            else:
                time = ds.time
                ls = ds.mars_solar_long

            # if you want to do a sanity check include this
            #if exp_name == exp_names[0]:
            #    axs[i,0].contourf(ds.time,ds.lat,ds.u50.transpose(),cmap='RdBu_r')
            #    axs[i,1].contourf(ds.time,ds.lat,ds.u50.transpose(),cmap='RdBu_r')
            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05':
                col = 'k'
                lnstl = '--'
            else:
                lnstl = '-'
                col = colors[l]
                l += 1
            ax = axs[i,0]
            ax.plot(time, phi_n,label=titles[j],color=col,linestyle=lnstl)
            ax.set_ylim([43,79])
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
            ax.plot(time, phi_s,label=titles[j],color=col,linestyle=lnstl)
            ax.set_ylim([-35,-80])
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



    fig.savefig('/user/home/xz19136/Figures/mars_analysis/jet/' \
                + 'jet_lat.%s' % ext, dpi=300,
                bbox_inches='tight')


def plot_jet_strength_evolution(exps=['curr-ecc','0-ecc','dust'], \
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
            
            ds = xr.open_dataset(path+exp_name+'/EDJ.nc', decode_times=False)

            #phi_n = ds.jets_n.squeeze()
            #phi_s = ds.jets_s.squeeze()
            jet_n = ds.jet_n.squeeze()
            jet_s = ds.jet_s.squeeze()


            #jet_n = get_jet_strength(ds.u50, phi_n)
            #jet_s = get_jet_strength(ds.u50, phi_s)

            if smooth is not None:
                time = funcs.moving_average(ds.time, smooth)
                jet_n = funcs.moving_average(jet_n, smooth)
                jet_s = funcs.moving_average(jet_s, smooth)
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
            ax.plot(time, jet_n,label=titles[j],color=col,linestyle=lnstl)
            ax.set_ylim([70,160])
            if exp == 'Attribution':
                ax.set_ylim([50,160])
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
            ax.plot(time, jet_s,label=titles[j],color=col,linestyle=lnstl)
            ax.set_ylim([38,120])
            if exp == 'Attribution':
                ax.set_ylim([20,120])
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
            ax.set_ylabel('jet strength (m s$^{-1}$)')
            if i % 2 == 1:
                ax.legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
                 borderaxespad=0, fontsize='large')
            
            if i == 0:
                ax.set_title('NH')
            elif i == 1:
                ax.set_title('SH')



    fig.savefig('/user/home/xz19136/Figures/mars_analysis/jet/' \
                + 'jet_strength.%s' % ext, dpi=300,
                bbox_inches='tight')

if __name__ == "__main__":
    eps = np.arange(10,55,5)
    gamma = [0.093,0.00]
    plot_jet_strength_evolution(exps=['curr-ecc','0-ecc','dust'],smooth=10,ext='pdf')
    plot_jet_lat_evolution(     exps=['curr-ecc','0-ecc','dust'],smooth=10,ext='pdf')

# %%
