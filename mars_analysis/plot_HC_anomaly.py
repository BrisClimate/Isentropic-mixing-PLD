# %%
import xarray as xr
import numpy as np
import sys, os
import math

sys.path.append('../')

from atmospy import nf, get_exps, stereo_plot, \
        Calculate_ZeroCrossing, calc_jet_lat, moving_average, new_cmap, \
        moving_average_2d, lait

import string

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import (cm, colors, cycler)
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


path = '/user/work/xz19136/Isca_data/'
figpath = '/user/home/xz19136/Figures/mars_analysis/HC/'
theta, center, radius, verts, circle = stereo_plot()
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
            _, HC_max = calc_jet_lat(Psi, lat)
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
            HC_edge = Calculate_ZeroCrossing(Psi, lat)
            s.append(HC_edge)
        except:
            s.append(np.nan)
    return s


def plot_HC_edge_evolution(exps=['curr-ecc','0-ecc','dust'], \
    smooth=None,ext='png',savedata=False):
    '''
    Plot effective diffusivity evolution at a given point,
    in order to understand strength of the transport barrier and mixing
    within the vortex.
    '''
    nrows = len(exps)+1
    ncols = 2
    if exps.count('MY28'):
        nrows -= 1
        
    fig, axs = plt.subplots(nrows=nrows,ncols=2, figsize = (15,4*(nrows-1)),dpi=300)
    
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
        elif exp == 'high_res_dust' or exp == 'vert_dust_only':
            exp = 'Dust Vertical Res'
        elif exp == 'long-dust' or exp == 'long-dust_only':
            exp = 'Longitudinal Dust'
        elif exp == 'attribution':
            exp = 'Attribution'
        l=0
        
            
        if savedata:
            for j in range(len(exp_names)):

                exp_name = exp_names[j]
                print(exp_name)

                try:
                    ds = xr.open_dataset(path+exp_name+'/psi.nc', decode_times=False)
                except:
                    continue
                #print('%.2f' % (sum([math.isnan(i) for i in ds.psi_0n])/len(ds.psi_0n)))
                #print('%.2f' % (sum([math.isnan(i) for i in ds.psi_0s])/len(ds.psi_0s)))

                psi0n = get_HC_edge(ds.where(ds.lat>0,drop=True))
                psi0s = get_HC_edge(ds.where(ds.lat<0,drop=True).sortby('lat',ascending=False))
                
                if smooth is not None:
                    time  = moving_average(ds.time, smooth)
                    psi0n = moving_average(psi0n, smooth)
                    psi0s = moving_average(psi0s, smooth)
                    if exp != '$\gamma = 0.000$':
                        ls    = moving_average(ds.mars_solar_long, smooth)
                else:
                    time = ds.time
                    psi0n = psi0n
                    psi0s = psi0s
                    if exp != '$\gamma = 0.000$':
                        ls = ds.mars_solar_long

                ds = xr.Dataset(data_vars=dict(
                        psi_n = (["time"], psi0n),
                        psi_s = (["time"], psi0s),
                        ls   = (["time"], ls.data),
                    ),
                    coords = dict(
                        time = time.data,
                    ),
                )
                #try:
                ds.to_netcdf(path+'mars_analysis/HC_lats/%s.nc' % exp_name)
                #except:
                #    print('Didn\'t save')

        if exp == '$\gamma = 0.093$' or exp == 'Dust Scale':
            d = xr.open_dataset(path+'mars_analysis/HC_lats/tracer_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.093_cdod_clim_scenario_7.4e-05.nc',
                                decode_times=False)
        elif exp == '$\gamma = 0.000$':
            d = xr.open_dataset(path+'mars_analysis/HC_lats/tracer_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.000_cdod_clim_scenario_7.4e-05.nc',
                                decode_times=False)
        elif exp == 'Dust Vertical Res':
            d = xr.open_dataset(path+'mars_analysis/HC_lats/tracer_vert_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.093_cdod_clim_scenario_7.4e-05.nc',
                                decode_times=False)
        elif exp == 'Longitudinal Dust':
            d = xr.open_dataset(path+'mars_analysis/HC_lats/tracer_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.093_clim_latlon_7.4e-05.nc',
                                decode_times=False)

        for j in range(len(exp_names)):
            exp_name = exp_names[j]
            try:
                ds = xr.open_dataset(path+'mars_analysis/HC_lats/%s.nc' % exp_name,
                                 decode_times=False)
            except:
                continue
            psi0n = ds.psi_n - d.psi_n
            psi0s = ds.psi_s - d.psi_s

            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_vert_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05'\
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_clim_latlon_7.4e-05':
                psi0n = d.psi_n
                psi0s = d.psi_s

            ls = ds.ls
            time = ds.time

            
            if exp == 'MY28':
                ax = axs[0,0]
            elif len(exps) == 1:
                ax = axs[0]
            else:
                ax = axs[i,0]

            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05':
                col = 'xkcd:teal'
                lnstl = '-'
                ax = axs[-1,0]
            elif exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05':
                col = 'xkcd:crimson'
                lnstl = '--'
                ax = axs[-1,0]
            elif exp_name == 'tracer_vert_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05':
                col = 'xkcd:crimson'
                lnstl = '--'
                ax = axs[-1,0]
            elif exp_name == 'tracer_MY28_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05':
                col = 'xkcd:crimson'
                lnstl = '-.'
            elif exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_clim_latlon_7.4e-05':
                col = 'xkcd:gold'
                lnstl = '-.'
                ax = axs[-1,0]
            else:
                lnstl = '-'
                col = colors[l]
                l += 1


            ax.plot(time, psi0n,label=titles[j],color=col,linestyle=lnstl)
            
            
            tind = (np.abs(ds.ls.values-270)).argmin(axis=0)
            
            ax.set_xlim([time[tind-45],time[tind+120]])
            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_vert_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05'\
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_clim_latlon_7.4e-05':
                axs[i,0].hlines(0,time[tind-45],time[tind+120],color='k',alpha=0.5)
            if exp != 'MY28' and exp != 'long-dust' and (ax != axs[-1,0]) and (ax != axs[-1,1]):
                ax.text(
                    -0.16, 0.5, exp,
                    ha='right',
                    va='center',
                    transform=ax.transAxes,
                    rotation='vertical',
                    fontsize='large',
                )
            if exp == 'MY28':
                ax = axs[0,1]
            elif len(exps) == 1:
                ax = axs[1]
            else:
                ax = axs[i,1]
            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_vert_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05'\
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_clim_latlon_7.4e-05':
                ax = axs[-1,1]
                titles[j] = exp

            ax.plot(time, psi0s,label=titles[j],color=col,linestyle=lnstl)
            
            tind = (np.abs(ds.ls.values-90)).argmin(axis=0)
            
            ax.set_xlim([time[tind-45],time[tind+120]])

        for i, ax in enumerate(fig.axes):
            ax.text(0.0, 1.03, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
                size='large')
            xlocs = [i for i in ax.get_xticks()]
            if smooth is not None:
                lsi = np.interp(xlocs, time, ls)
            else:
                lsi = ls.interp(time=xlocs,kwargs={"fill_value":"extrapolate"})#, d.time)
            if i < 2*(len(exps)+1)-2:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(['%i' % i for i in lsi])
                ax.set_xlabel('$\mathrm{L_s}$',fontsize='large')
            if exps.count('MY28') and i >= 2*len(exps)-4:
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

    for i in range(nrows-1):
        axs[i,0].set_ylim([-10,40])
        axs[i,1].set_ylim([10,-40])
        axs[i,1].hlines(0,time[tind-45],time[tind+120], color='k',alpha=0.5)
    
    axs[-1,0].set_ylim([35,57])
    axs[-1,1].set_ylim([-35,-57])



    fig.savefig(figpath \
                + 'HC_edge_%s.%s' % (exps[-1],ext), dpi=300,
                bbox_inches='tight')

def plot_HC_strength_evolution(exps=['curr-ecc','0-ecc','dust'], \
    smooth=None,ext='png',savedata=False):
    '''
    Plot effective diffusivity evolution at a given point,
    in order to understand strength of the transport barrier and mixing
    within the vortex.
    '''
    nrows = len(exps)+1
    ncols = 2
    if exps.count('MY28'):
        nrows -= 1
        
    fig, axs = plt.subplots(nrows=nrows,ncols=2, figsize = (15,4*(nrows-1)),dpi=300)
    
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
        elif exp == 'high_res_dust' or exp == 'vert_dust_only':
            exp = 'Dust Vertical Res'
        elif exp == 'long-dust' or exp == 'long-dust_only':
            exp = 'Longitudinal Dust'
        elif exp == 'attribution':
            exp = 'Attribution'
        l=0
        
            
        if savedata:
            for j in range(len(exp_names)):

                exp_name = exp_names[j]
                print(exp_name)

                try:
                    ds = xr.open_dataset(path+exp_name+'/psi.nc', decode_times=False)
                except:
                    continue
                ds = ds.transpose('lat','pfull','time')

                psi_0n = get_HC_strength(ds.where(ds.lat>0,drop=True))
                psi0n = [i/10**8 for i in psi_0n]
                psi_0s = get_HC_strength(-ds.where(ds.lat<0,drop=True))
                psi0s = [-i/10**8 for i in psi_0s]

                if smooth is not None:
                    time  = moving_average(ds.time, smooth)
                    psi0n = moving_average(psi0n, smooth)
                    psi0s = moving_average(psi0s, smooth)
                    if exp != '$\gamma = 0.000$':
                        ls    = moving_average(ds.mars_solar_long, smooth)
                else:
                    time = ds.time
                    psi0n = psi0n
                    psi0s = psi0s
                    if exp != '$\gamma = 0.000$':
                        ls = ds.mars_solar_long

                ds = xr.Dataset(data_vars=dict(
                        psi_n = (["time"], psi0n),
                        psi_s = (["time"], psi0s),
                        ls   = (["time"], ls.data),
                    ),
                    coords = dict(
                        time = time.data,
                    ),
                )
                try:
                    ds.to_netcdf(path+'mars_analysis/HC_strength/%s.nc' % exp_name)
                except:
                    print('Didn\'t save')

        if exp == '$\gamma = 0.093$' or exp == 'Dust Scale':
            d = xr.open_dataset(path+'mars_analysis/HC_strength/tracer_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.093_cdod_clim_scenario_7.4e-05.nc',
                                decode_times=False)
        elif exp == '$\gamma = 0.000$':
            d = xr.open_dataset(path+'mars_analysis/HC_strength/tracer_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.000_cdod_clim_scenario_7.4e-05.nc',
                                decode_times=False)
        elif exp == 'Dust Vertical Res':
            d = xr.open_dataset(path+'mars_analysis/HC_strength/tracer_vert_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.093_cdod_clim_scenario_7.4e-05.nc',
                                decode_times=False)
        elif exp == 'Longitudinal Dust':
            d = xr.open_dataset(path+'mars_analysis/HC_strength/tracer_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.093_clim_latlon_7.4e-05.nc',
                                decode_times=False)

        for j in range(len(exp_names)):
            exp_name = exp_names[j]
            try:
                ds = xr.open_dataset(path+'mars_analysis/HC_strength/%s.nc' % exp_name,
                                 decode_times=False)
            except:
                continue
            psi0n = ds.psi_n - d.psi_n
            psi0s = ds.psi_s - d.psi_s

            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_vert_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05'\
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_clim_latlon_7.4e-05':
                psi0n = d.psi_n
                psi0s = d.psi_s

            ls = ds.ls
            time = ds.time

            
            if exp == 'MY28':
                ax = axs[0,0]
            elif len(exps) == 1:
                ax = axs[0]
            else:
                ax = axs[i,0]

            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05':
                col = 'xkcd:teal'
                lnstl = '-'
                ax = axs[-1,0]
            elif exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05':
                col = 'xkcd:crimson'
                lnstl = '--'
                ax = axs[-1,0]
            elif exp_name == 'tracer_vert_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05':
                col = 'xkcd:crimson'
                lnstl = '--'
                ax = axs[-1,0]
            elif exp_name == 'tracer_MY28_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05':
                col = 'xkcd:crimson'
                lnstl = '-.'
            elif exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_clim_latlon_7.4e-05':
                col = 'xkcd:gold'
                lnstl = '-.'
                ax = axs[-1,0]
            else:
                lnstl = '-'
                col = colors[l]
                l += 1


            ax.plot(time, psi0n,label=titles[j],color=col,linestyle=lnstl)
            
            
            tind = (np.abs(ds.ls.values-270)).argmin(axis=0)
            
            ax.set_xlim([time[tind-45],time[tind+120]])
            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_vert_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05'\
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_clim_latlon_7.4e-05':
                axs[i,0].hlines(0,time[tind-45],time[tind+120],color='k',alpha=0.5)
            
            if exp != 'MY28' and exp != 'long-dust' and (ax != axs[-1,0]) and (ax != axs[-1,1]):
                ax.text(
                    -0.16, 0.5, exp,
                    ha='right',
                    va='center',
                    transform=ax.transAxes,
                    rotation='vertical',
                    fontsize='large',
                )
            if exp == 'MY28':
                ax = axs[0,1]
            elif len(exps) == 1:
                ax = axs[1]
            else:
                ax = axs[i,1]
            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_vert_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05'\
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_clim_latlon_7.4e-05':
                ax = axs[-1,1]
                titles[j] = exp

            ax.plot(time, psi0s,label=titles[j],color=col,linestyle=lnstl)
            
            tind = (np.abs(ds.ls.values-90)).argmin(axis=0)
            
            ax.set_xlim([time[tind-45],time[tind+120]])

        for i, ax in enumerate(fig.axes):
            ax.text(0.0, 1.03, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
                size='large')
            xlocs = [i for i in ax.get_xticks()]
            if smooth is not None:
                lsi = np.interp(xlocs, time, ls)
            else:
                lsi = ls.interp(time=xlocs,kwargs={"fill_value":"extrapolate"})#, d.time)
            if i < 2*(len(exps)+1)-2:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(['%i' % i for i in lsi])
                ax.set_xlabel('$\mathrm{L_s}$',fontsize='large')
            if exps.count('MY28') and i >= 2*len(exps)-4:
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

    for i in range(nrows-1):
        axs[i,0].set_ylim([-22,32])
        axs[i,1].set_ylim([22,-32])
        axs[i,1].hlines(0,time[tind-45],time[tind+120], color='k',alpha=0.5)
    
    axs[-1,0].set_ylim([ 10, 36])
    axs[-1,1].set_ylim([-10,-36])



    fig.savefig(figpath \
                + 'HC_strength_%s.%s' % (exps[-1],ext), dpi=300,
                bbox_inches='tight')
    
def plot_psi_cross_section(exps=['curr-ecc','0-ecc','dust'], \
    smooth=None,ext='png',savedata=False,hem='NH',average=30):
    '''
    Plot effective diffusivity evolution at a given point,
    in order to understand strength of the transport barrier and mixing
    within the vortex.
    '''
    for i in range(len(exps)):
        exp = exps[i]
        
        exp_names, titles, _, _ = get_exps(exp)
        
        if exp == 'curr-ecc':
            exp = '$\gamma = 0.093$'
            nrow = 3
            ncol = 3
        elif exp == '0-ecc':
            exp = '$\gamma = 0.000$'
            nrow = 3
            ncol = 3
        elif exp == 'dust':
            exp = 'Dust Scale'
            nrow = 3
            ncol = 2
            titles = ['$\lambda = $' + i for i in titles]
        elif exp == 'high_res_dust':
            exp = 'Dust Vertical Res'
            nrow = 5
            ncol = 2
        elif exp == 'attribution':
            exp = 'Attribution'
            nrow = 2
            ncol = 4
        elif exp == 'MY28':
            nrow = 1
            ncol = 1
        elif exp == 'long-dust':
            nrow = 5
            ncol = 2
        
        fig, axs = plt.subplots(nrows=nrow,ncols=ncol, figsize = (4*ncol,2.5*nrow))
            
        if savedata:
            for j in range(len(exp_names)):
                exp_name = exp_names[j]
                print(exp_name)
                if os.path.isfile(path+'mars_analysis/HC_x/%s_%s.nc' % (exp_name,hem)):
                    continue

                try:
                    ds = xr.open_dataset(path+exp_name+'/psi.nc', decode_times=False)
                except:
                    continue

                if exp != '$\gamma = 0.000$':
                    if hem == 'NH':
                        tind = (np.abs(ds.mars_solar_long.values-270)).argmin(axis=0)
                        #i = np.take_along_axis(ds, np.abs(ds.mars_solar_long.values-270).argmin(axis=0), axis=0)
                    elif hem == 'SH':
                        tind = np.abs(ds.mars_solar_long.values-90).argmin(axis=0)

                tslice = slice(tind-average,tind+average)
                ds = ds.isel(time=tslice).mean(dim="time")

                if hem == 'NH':
                    ds = ds.where(ds.lat>0,drop=True)
                elif hem == 'SH':
                    ds = ds.where(ds.lat<0,drop=True)

                ds = ds.psi
                try:
                    ds.to_netcdf(path+'mars_analysis/HC_x/%s_%s.nc' % (exp_name,hem))
                except:
                    print('Didn\'t save')

        if exp == '$\gamma = 0.093$' or exp == 'Dust Scale':
            d = xr.open_dataset(path+'mars_analysis/HC_x/tracer_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.093_cdod_clim_scenario_7.4e-05_%s.nc' % hem,
                                decode_times=False)
        elif exp == '$\gamma = 0.000$':
            d = xr.open_dataset(path+'mars_analysis/HC_x/tracer_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.000_cdod_clim_scenario_7.4e-05_%s.nc' % hem,
                                decode_times=False)
        elif exp == 'Dust Vertical Res':
            d = xr.open_dataset(path+'mars_analysis/HC_x/tracer_vert_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.093_cdod_clim_scenario_7.4e-05_%s.nc' % hem,
                                decode_times=False)
        elif exp == 'Longitudinal Dust':
            d = xr.open_dataset(path+'mars_analysis/HC_x/tracer_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.093_clim_latlon_7.4e-05_%s.nc' % hem,
                                decode_times=False)

        for j, ax in enumerate(fig.axes):
            try:
                exp_name = exp_names[j]
            except:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            try:
                ds = xr.open_dataset(path+'mars_analysis/HC_x/%s_%s.nc' % (exp_name,hem),
                                 decode_times=False)
            except:
                continue
            psi = ds.psi - d.psi


            lat = ds.lat
            pfull= ds.pfull

            lims = [-15,15]
            levels = [-10,-5,5,10]
            if exp == 'long-dust' and hem == 'NH':
                lims = [-8, 8]
            boundaries, cmap, norm = new_cmap(lims, extend='both', i=20,cols='RdBu_r')
            ax.set_ylim([6,0.01])
            ax.set_yscale('log')
            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_vert_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05'\
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_clim_latlon_7.4e-05':
                psi = d.psi

                
                levels = [0,50,100,120]
                if exp == 'long-dust' and hem == 'NH':
                    lims = [10, 50]
                if hem == 'NH':
                    cols = 'Spectral_r'
                    lims = [-5,15]
                else:
                    cols = 'Spectral_r'
                    lims = [-15,5]
                levels = [-10,-5,0,5,10]
                boundaries, cmap, norm = new_cmap(lims, extend='both',
                                override=True, i=16,cols=cols,user_choice=True)
                cb = fig.colorbar(
                    cm.ScalarMappable(norm=norm, cmap=cmap),extend = 'both',
                    ticks=boundaries[slice(None,None,4)],ax = ax,aspect=20,pad=-0.15,
                    #location='left',
                    )
                cb.set_label(label='$\psi$ ($10^8$ kg s$^{-1}$)',fontsize='large')
            
            c1 = ax.contourf(ds.lat, ds.pfull, psi.transpose()/10**8,
                norm=norm,cmap=cmap,levels=[boundaries[0]-100]+boundaries+[boundaries[-1]+100])
            c0 = ax.contour(ds.lat, ds.pfull, psi.transpose()/10**8,
                            colors='k',linewidths=0.5,levels=levels)
            
            c0.levels = [nf(val) for val in c0.levels]
            ax.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
            

            if exp == '$\gamma = 0.093$' or exp == '$\gamma = 0.000$':
                ax.set_title(titles[j]+'$^\circ$')
            elif exp == 'MY28':
                ax.set_title('')
            else:
                ax.set_title(titles[j])
            

        for i, ax in enumerate(fig.axes):
            try:
                exp_name = exp_names[i]
            except:
                continue
            if i >= ncol*nrow:
                continue
            ax.text(0.0, 1.03, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
                size='large')
            if exp != 'Dust Scale':
                if i < ncol*(nrow-1):
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('latitude ($^\circ$N)',fontsize='large')
            else:
                if i < ncol*(nrow-1)-1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('latitude ($^\circ$N)',fontsize='large')
            if exps.count('MY28') and i >= 2*len(exps)-4:
                ax.set_xlabel('latitude ($^\circ$N)',fontsize='large')

            if i % ncol == 0:
                ax.set_ylabel('pressure (hPa)')
            else:
                ax.set_yticklabels([])
        fig.suptitle(exp)
        fig.subplots_adjust(hspace=0.35,wspace=0.58)#,left=0.1,right=0.3)
        cb = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),extend = 'both',
            ticks=boundaries[slice(None,None,2)],ax = axs,aspect=30,#location='left',
            pad=0.05)
        cb.set_label(label='$\psi$ ($10^8$ kg s$^{-1}$)',fontsize='large')
        
        fig.savefig(figpath \
                    + 'HC_x_%s_%s.%s' % (exps[-1],hem,ext), dpi=300,
                    bbox_inches='tight')


def plot_psi_cross_section_long(exps=['long-dust'], \
    smooth=None,ext='png',savedata=False,hem='NH',average=30):
    '''
    Plot effective diffusivity evolution at a given point,
    in order to understand strength of the transport barrier and mixing
    within the vortex.
    '''
    for i in range(len(exps)):
        exp = exps[i]
        
        exp_names, titles, _, _ = get_exps(exp)
        
        if exp == 'high_res_dust':
            exp = 'Dust Vertical Res'
            nrow = 4
            ncol = 2
        elif exp == 'long-dust':
            nrow = 4
            ncol = 2
        
        fig, axs = plt.subplots(nrows=nrow,ncols=ncol, figsize = (4*ncol,2.5*nrow))
            
        if savedata:
            for j in range(len(exp_names)):
                exp_name = exp_names[j]
                print(exp_name)
                if os.path.isfile(path+'mars_analysis/HC_x/%s_%s.nc' % (exp_name,hem)):
                    continue

                try:
                    ds = xr.open_dataset(path+exp_name+'/psi.nc', decode_times=False)
                except:
                    continue


                if hem == 'NH':
                    tind = (np.abs(ds.mars_solar_long.values-270)).argmin(axis=0)
                    #i = np.take_along_axis(ds, np.abs(ds.mars_solar_long.values-270).argmin(axis=0), axis=0)
                elif hem == 'SH':
                    tind = np.abs(ds.mars_solar_long.values-90).argmin(axis=0)

                tslice = slice(tind-average,tind+average)
                ds = ds.isel(time=tslice).mean(dim="time")

                if hem == 'NH':
                    ds = ds.where(ds.lat>0,drop=True)
                elif hem == 'SH':
                    ds = ds.where(ds.lat<0,drop=True)

                ds = ds.psi
                try:
                    ds.to_netcdf(path+'mars_analysis/HC_x/%s_%s.nc' % (exp_name,hem))
                except:
                    print('Didn\'t save')
        
        l = []
        for j in range(len(exp_names)):
            try:
                exp_name = exp_names[j]
                print(j)
            except:
                continue
            
            ds = xr.open_dataset(path+'mars_analysis/HC_x/%s_%s.nc' % (exp_name,hem),
                                 decode_times=False)
            
            l.append(ds)
        print(len(l))
        for i, ax in enumerate(fig.axes):
            try:
                exp_name = exp_names[j]
            except:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            lims = [-5,5]
            levels = [-20,-10,0,10,20]
            if exp == 'long-dust' and hem == 'NH':
                lims = [-8, 8]
            boundaries, cmap, norm = new_cmap(lims, extend='both', i=20,cols='RdBu_r')
            
            if i % 2 == 0:
                psi = l[i].psi
                if hem == 'NH':
                    cols = 'Reds'
                    lims = [-5,25]
                else:
                    cols = 'Blues_r'
                    lims = [-25,5]
                boundaries, cmap, norm = new_cmap(lims, extend='both',
                                override=True, i=10,cols=cols,user_choice=True)
            else:
                print(exp_names[i])
                print(exp_names[i-1])
                psi = l[i].psi-l[i-1].psi
            
            c1 = ax.contourf(ds.lat, ds.pfull, psi.transpose()/10**8,
                norm=norm,cmap=cmap,levels=[boundaries[0]-100]+boundaries+[boundaries[-1]+100])
            c0 = ax.contour(ds.lat, ds.pfull, psi.transpose()/10**8,
                            colors='k',linewidths=0.5,levels=[-10,-5,-2,2,5,10])
            
            c0.levels = [nf(val) for val in c0.levels]
            ax.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
            ax.set_ylim([6,0.01])
            ax.set_yscale('log')

            ax.set_title(titles[i],fontsize='small')

            if ax == axs[-1,0]:
                cb = fig.colorbar(
                    cm.ScalarMappable(norm=norm, cmap=cmap),extend = 'both',
                    ticks=boundaries[slice(None,None,2)],ax = axs[:,0],aspect=40,
                    )
                cb.set_label(label='$\psi$ ($10^8$ kg s$^{-1}$)',fontsize='large')
            elif ax == axs[-1,1]:
                cb = fig.colorbar(
                    cm.ScalarMappable(norm=norm, cmap=cmap),extend = 'both',
                    ticks=boundaries[slice(None,None,2)],ax = axs[:,1],aspect=40,
                    )
                cb.set_label(label='$\psi$ ($10^8$ kg s$^{-1}$)',fontsize='large')
                
            
            ax.text(-0.05, 1.06, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
                size='large')
            if i < ncol*(nrow-1):
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('latitude ($^\circ$N)')
            if exps.count('MY28') and i >= 2*len(exps)-4:
                ax.set_xlabel('latitude ($^\circ$N)')

            if i % ncol == 0:
                ax.set_ylabel('pressure (hPa)')
            else:
                ax.set_yticklabels([])
        #fig.suptitle(exp)
        #fig.subplots_adjust(hspace=0.35,wspace=0.58)#,left=0.1,right=0.3)
        
        
        fig.savefig(figpath \
                    + 'HC_x_%s_%s.%s' % (exps[-1],hem,ext), dpi=300,
                    bbox_inches='tight')
    
def plot_psi_cross_section_one(exp_name, ext='png',average=30):
    '''
    Plot effective diffusivity evolution at a given point,
    in order to understand strength of the transport barrier and mixing
    within the vortex.
    '''
    
        
    fig, axs = plt.subplots(nrows=2,ncols=2, figsize = (6,5),
        gridspec_kw= {'height_ratios': [5, 2], 'width_ratios':[5,1]})
        
    
    d_psi = xr.open_dataset(path+exp_name+'/psi.nc', decode_times=False)
    du = xr.open_dataset(path+exp_name+'/EDJ.nc', decode_times=False)
    dPV = xr.open_dataset(path+exp_name+'/atmos_isentropic.nc', decode_times=False)
    d = xr.open_dataset(path+exp_name+'/atmos.nc', decode_times=False)
    dt = xr.open_dataset(path+'mars_analysis/HC_lats/'+exp_name+'.nc')
    

    tindn = (np.abs(d_psi.mars_solar_long.values-270)).argmin(axis=0)
    tslicen = slice(tindn-average,tindn+average)

    d_psi = d_psi.where(d_psi.lat>0,drop=True)
    dPV = dPV.where(dPV.lat>0,drop=True)

    d_psi = d_psi.isel(time=tslicen).mean(dim="time").psi
    dun = du.isel(time=tslicen).mean(dim="time").u50


    dtn = dt.isel(time=tslicen).mean(dim="time").psi_n
    du_phi = du.isel(time=tslicen).mean(dim="time").phi_n
    du_max = du.isel(time=tslicen).mean(dim="time").jet_n
    
    dPV = dPV[["PV","mars_solar_long"]].sel(level=300,method="nearest").mean(dim='lon')
    dPV['mars_solar_long'] = dPV.mars_solar_long.isel(lat=0)
    tindn = (np.abs(dPV.mars_solar_long.values-270)).argmin(axis=0)
    tslicen = slice(tindn-average,tindn+average)
    dPV = dPV.isel(time=tslicen).mean(dim="time").PV
    dPV = dPV*10**4
    dPV = lait(dPV,300,theta0,kappa=kappa)
    dPV_max = dPV.max().values
    dPV_phi = dPV.lat.isel(lat=dPV.argmax().values)
    
    d = d.ucomp.mean(dim='lon')
    d = d.isel(time=tslicen).mean(dim="time")

    dt = [dtn, du_phi]

    lims = [-15,15]
    levels = [-20,-10,0,10,20]

    p = d_psi.pfull.mean().values

    boundaries, cmap, norm = new_cmap(lims, extend='both', i=20,cols='RdBu_r')


    ax = axs[0,0]
    di = d_psi
    c1 = ax.contourf(di.lat, di.pfull, di.transpose()/10**8,
    norm=norm,cmap=cmap,levels=[boundaries[0]-100]+boundaries+[boundaries[-1]+100])
    c0 = ax.contour(di.lat, di.pfull, di.transpose()/10**8,
                    colors='k',linewidths=0.5,levels=[-9,-6,-3,3,6,9])
    ax.contour(d.lat,d.pfull,d.transpose(),colors='k',levels=[-50,0,50,100,150])
    #ax.plot(dt[0], p, marker='x',color='purple',ms=12)
    ax.annotate('$\phi_{HC}$',(dt[0], p),(dt[0]+1, p-0.9),color='purple',fontsize='large')
    ax.plot(dt[1], 0.5, marker='x',color='k',ms=12)
    ax.annotate('$\phi_{u_{max}}$',(dt[1], 0.5),(dt[1]-1, 0.35),color='k',fontsize='large')
    ax.errorbar(dt[0], p, yerr=[[3.551],[2]],color='purple',capsize=2)
    c0.levels = [nf(val) for val in c0.levels]
    ax.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
    cb = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),extend = 'both',
        ticks=boundaries[slice(None,None,2)],ax = axs[0,1],aspect=25,shrink=1.5,
        pad = -1.5)
    cb.set_label(label='$\psi$ ($10^8$ kg s$^{-1}$)',fontsize='large')

    ax.set_ylim([6,0.01])
    ax.set_yscale('log')

    axs[0,0].set_ylabel('pressure (hPa)')
    ax.set_xticklabels([])

    ax = axs[1,0]

    di = dPV
    ax.plot(di.lat,di,c='tab:orange')
    ax.plot(dPV_phi, dPV_max, ms=6, marker='x',c='tab:orange')
    ax.annotate('$\phi_{PV}$',(dPV_phi, dPV_max),(dPV_phi+3, dPV_max+0.5),
                color='tab:orange',fontsize='large')
    ax.set_ylabel('PV$_{max}$ (MPVU)',c='tab:orange')
    ax.set_xlabel('latitude ($^\circ$N)')

    ax = axs[1,0].twinx()
    

    ds = xr.open_dataset(path+exp_name+'/keff_isentropic_test_tracer.nc', decode_times=False)
    ds = ds.nkeff.sel(level=300,method="nearest")
    keff = ds.where(ds.new > 0, drop=True).isel(time=tslicen).mean(dim="time")
    keff_n = keff.where(keff.new > 40, drop=True)
    dn_weighted = keff_n.weighted(np.cos(np.deg2rad(keff_n.new)))
    keff_n = dn_weighted.mean(dim="new")

    ax.plot(keff.new,np.log(keff),c='tab:blue')    
    ax.errorbar(65, np.log(keff_n), xerr=[[25], [keff.new.max().values-65]],
                color='tab:blue',capsize=2)
    ax.annotate('$\kappa_{{eff}_{300}}$',(65, np.log(keff_n)),(59, np.log(keff_n)+0.08),
                color='tab:blue',fontsize='large')
    ax.set_ylabel('$\kappa_{eff}$',c='tab:blue',fontsize='large')
    ax.set_xlabel('latitude ($^\circ$N)')

    for i in [0,1]:
        axs[i,1].set_frame_on(False)
        axs[i,1].set_xticks([])
        axs[i,1].set_yticks([])
        axs[i,0].set_xlim([0,90])

    fig.tight_layout()
    fig.subplots_adjust(wspace=-0.01)
    

    fig.savefig(figpath \
                + 'HC_x_%s.%s' % (exp_name,ext), dpi=300,
                bbox_inches='tight')


if __name__ == "__main__":
    eps = np.arange(10,55,5)
    gamma = [0.093,0.00]
    savedata=True
    plot_psi_cross_section_one(exp_name=
        'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05',
        ext='pdf')
    #plot_psi_cross_section_long(exps=['high_res_dust'],ext='pdf',savedata=savedata,
    #                       hem='NH')
    #plot_psi_cross_section(exps=['0-ecc'],ext='pdf',savedata=savedata,
    #                       hem='NH')
    #plot_HC_edge_evolution(    exps=['dust','vert_dust_only','long-dust_only'],smooth=5,
    #                    ext='pdf',savedata=savedata)
    #plot_HC_strength_evolution(exps=['dust','vert_dust_only','long-dust_only'],smooth=5,
    #                    ext='pdf',savedata=savedata)
# %%
