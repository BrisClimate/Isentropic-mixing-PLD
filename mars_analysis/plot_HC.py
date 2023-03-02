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

def get_HC_strength(d):
    lat = d.lat.values
    lev = d.pfull
    
    d = d.where(lev < 5, drop = True)
    d = d.where(lev > 0.5, drop = True)

    psi = d.psi.weighted(d.pfull)
    psi = psi.mean(dim="pfull")

    s = []
    for a in range(len(d.time)):
        Psi = psi.isel(time=a)
        try:
            _, HC_max = funcs.calc_jet_lat(Psi, lat)
            s.append(HC_max)
        except:
            s.append(np.nan)
    return s

def get_HC_edge(d):
    lat = d.lat.values
    lev = d.pfull
    
    d = d.where(lev < 5, drop = True)
    d = d.where(lev > 0.5, drop = True)

    psi = d.psi.weighted(d.pfull)
    psi = psi.mean(dim="pfull")

    s = []
    for a in range(len(d.time)):
        Psi = psi.isel(time=a)
        try:
            HC_edge = funcs.Calculate_ZeroCrossing(Psi, lat)
            s.append(HC_edge)
        except:
            s.append(np.nan)
    return s


def plot_HC_edge_evolution(exps=['curr-ecc','0-ecc','dust'], \
    smooth=None,ext='png'):
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
        l=0
        for j in range(len(exp_names)):

            exp_name = exp_names[j]
            print(exp_name)
            
            ds = xr.open_dataset(path+exp_name+'/psi.nc', decode_times=False)
            #print('%.2f' % (sum([math.isnan(i) for i in ds.psi_0n])/len(ds.psi_0n)))
            #print('%.2f' % (sum([math.isnan(i) for i in ds.psi_0s])/len(ds.psi_0s)))

            psi0n = get_HC_edge(ds.where(ds.lat>0,drop=True))
            psi0s = get_HC_edge(ds.where(ds.lat<0,drop=True).sortby('lat',ascending=False))
            if smooth is not None:
                time = funcs.moving_average(ds.time, smooth)
                psi0n = funcs.moving_average(psi0n, smooth)
                psi0s = funcs.moving_average(psi0s, smooth)
                ls = funcs.moving_average(ds.mars_solar_long, smooth)
            else:
                time = ds.time
                psi0n = psi0n
                psi0s = psi0s
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
            ax.plot(time, psi0n,label=titles[j],color=col,linestyle=lnstl)
            ax.set_ylim([35,70])
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
            ax.plot(time, psi0s,label=titles[j],color=col,linestyle=lnstl)
            ax.set_ylim([-25,-75])
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



    fig.savefig('/user/home/xz19136/Figures/mars_analysis/HC/' \
                + 'HC_edge.%s' % ext, dpi=300,
                bbox_inches='tight')


def plot_HC_strength_evolution(exps=['curr-ecc','0-ecc','dust'], \
    smooth=None, ext='png'):
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
            
            ds = xr.open_dataset(path+exp_name+'/psi.nc', decode_times=False)
            #print('%.2f' % (sum([math.isnan(i) for i in ds.psi_0n])/len(ds.psi_0n)))
            #print('%.2f' % (sum([math.isnan(i) for i in ds.psi_0s])/len(ds.psi_0s)))
            ds = ds.transpose('lat','pfull','time')
            psi_0n = get_HC_strength(ds.where(ds.lat>0,drop=True))
            psi0n = [i/10**8 for i in psi_0n]
            psi_0s = get_HC_strength(-ds.where(ds.lat<0,drop=True))
            psi0s = [-i/10**8 for i in psi_0s]

            if smooth is not None:
                time = funcs.moving_average(ds.time, smooth)
                psi0n = funcs.moving_average(psi0n, smooth)
                psi0s = funcs.moving_average(psi0s, smooth)
                ls = funcs.moving_average(ds.mars_solar_long, smooth)
            else:
                time = ds.time
                psi0n = psi0n
                psi0s = psi0s
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
            ax.plot(time, psi0n,label=titles[j],color=col,linestyle=lnstl)
            ax.set_ylim([0,50])
            if exp == 'Attribution':
                ax.set_ylim([0,100])
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
            ax.plot(time, psi0s,label=titles[j],color=col,linestyle=lnstl)
            ax.set_ylim([0,-50])
            if exp == 'Attribution':
                ax.set_ylim([0,-75])
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
            ax.set_ylabel('$\psi$ ($10^8$ kg s$^{-1}$)')
            if i % 2 == 1:
                ax.legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
                 borderaxespad=0, fontsize='large')
            
            if i == 0:
                ax.set_title('NH')
            elif i == 1:
                ax.set_title('SH')



    fig.savefig('/user/home/xz19136/Figures/mars_analysis/HC/' \
                + 'HC_strength.%s' % ext, dpi=300,
                bbox_inches='tight')

def plot_psi_crosssection(exp_name, tind=101,edge=False,ext='png'):
    ds = xr.open_dataset(path+exp_name+'/psi.nc', decode_times=False)

    ds = ds.isel(time=slice(tind-15,tind+15)).mean(dim="time").squeeze()

    lims = [-2,2]

    boundaries, cmap, norm = funcs.new_cmap(lims, extend='both',i=10)
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

    c1 = axs.contourf(ds.lat, ds.pfull, ds.psi.transpose()/10**9,
        norm=norm,cmap=cmap,levels=[boundaries[0]-100]+boundaries+[boundaries[-1]+100])
    fig.suptitle(exp_name)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),extend = 'both',
        orientation='vertical', label='streamfunction')
    axs.set_ylim([6,0.01])
    axs.set_yscale('log')
    if edge:
        axs.plot(ds.psi_0n, 1, marker='s')
        axs.plot(ds.psi_0s, 1, marker='s')


    fig.savefig('/user/home/xz19136/Figures/mars_analysis/HC/xsect_' + \
        '%s_%03d.%s' % (exp_name,tind,ext),
        bbox_inches='tight', dpi=300)
    

if __name__ == "__main__":
    eps = np.arange(10,55,5)
    gamma = [0.093,0.00]
    plot_HC_edge_evolution(exps=['curr-ecc','0-ecc','dust'],smooth=10,ext='pdf')
    plot_HC_strength_evolution(exps=['curr-ecc','0-ecc','dust'],smooth=10,ext='pdf')
    exps = []
    
    #for l in ['', '_lh']:
    #    for d in ['', '_cdod_clim_scenario_7.4e-05']:
    #        for t in ['', '_mola_topo']:
    #            exps.append('tracer_soc_mars%s%s_eps_25_gamma_0.093%s' % (t, l, d))

    for ep in eps:
        for gam in gamma:
            exps.append('tracer_soc_mars_mola_topo_lh_eps_' + \
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (ep, gam))

    for dust_scale in [3.7e-5, 7.4e-05, 1.48e-4, 2.96e-4]:
      exps.append('tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_'+str(dust_scale))
    
    #for exp_name in exps:
    #    plot_psi_crosssection(exp_name,edge=True,tind=450)
    #for tind in np.arange(90,120):
    #    plot_psi_crosssection('tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_0.000296',
    #    edge = True, tind = tind)
# %%
