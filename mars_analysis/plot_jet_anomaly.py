# %%
import xarray as xr
import numpy as np
import sys, os
import math

sys.path.append('../')

from atmospy import get_exps, stereo_plot, \
        Calculate_ZeroCrossing, calc_jet_lat, moving_average

import string

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import (cm, colors, cycler)
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


path = '/user/work/xz19136/Isca_data/'
figpath = '/user/home/xz19136/Figures/mars_analysis/jet/'
theta, center, radius, verts, circle = stereo_plot()
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
    smooth=None,ext='png',level=300,savedata=False):
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
                    ds = xr.open_dataset(path+exp_name+'/EDJ.nc', decode_times=False)
                except:
                    continue
                #print('%.2f' % (sum([math.isnan(i) for i in ds.psi_0n])/len(ds.psi_0n)))
                #print('%.2f' % (sum([math.isnan(i) for i in ds.psi_0s])/len(ds.psi_0s)))

                phi_n = ds.phi_n.squeeze()
                phi_s = ds.phi_s.squeeze()


                if smooth is not None:
                    time  = moving_average(ds.time, smooth)
                    phi_n = moving_average(phi_n, smooth)
                    phi_s = moving_average(phi_s, smooth)
                    if exp != '$\gamma = 0.000$':
                        ls    = moving_average(ds.mars_solar_long, smooth)
                else:
                    time = ds.time
                    if exp != '$\gamma = 0.000$':
                        ls = ds.mars_solar_long
            
                ds = xr.Dataset(data_vars=dict(
                        jet_n = (["time"], phi_n),
                        jet_s = (["time"], phi_s),
                        ls   = (["time"], ls.data),
                    ),
                    coords = dict(
                        time = time.data,
                    ),
                )
                ds.to_netcdf(path+'mars_analysis/jet_lats/%s.nc' % exp_name)

        if exp == '$\gamma = 0.093$' or exp == 'Dust Scale':
            d = xr.open_dataset(path+'mars_analysis/jet_lats/tracer_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.093_cdod_clim_scenario_7.4e-05.nc',
                                decode_times=False)
        elif exp == '$\gamma = 0.000$':
            d = xr.open_dataset(path+'mars_analysis/jet_lats/tracer_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.000_cdod_clim_scenario_7.4e-05.nc',
                                decode_times=False)
        elif exp == 'Dust Vertical Res':
            d = xr.open_dataset(path+'mars_analysis/jet_lats/tracer_vert_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.093_cdod_clim_scenario_7.4e-05.nc',
                                decode_times=False)
        elif exp == 'Longitudinal Dust':
            d = xr.open_dataset(path+'mars_analysis/jet_lats/tracer_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.093_clim_latlon_7.4e-05.nc',
                                decode_times=False)

        for j in range(len(exp_names)):
            exp_name = exp_names[j]
            try:
                ds = xr.open_dataset(path+'mars_analysis/jet_lats/%s.nc' % exp_name,
                                 decode_times=False)
            except:
                continue
            phi_n = ds.jet_n - d.jet_n
            phi_s = ds.jet_s - d.jet_s

            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_vert_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05'\
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_clim_latlon_7.4e-05':
                phi_n = d.jet_n
                phi_s = d.jet_s

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


            ax.plot(time, phi_n,label=titles[j],color=col,linestyle=lnstl)
            
            
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

            ax.plot(time, phi_s,label=titles[j],color=col,linestyle=lnstl)
            
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
        axs[i,0].set_ylim([-11.5, 24.5])
        axs[i,1].set_ylim([ 11.5,-24.5])
        axs[i,1].hlines(0,time[tind-45],time[tind+120], color='k',alpha=0.5)
    
    axs[-1,0].set_ylim([51,66])
    axs[-1,1].set_ylim([-51,-66])



    fig.savefig(figpath \
                + 'jet_lat_%s.%s' % (exps[-1],ext), dpi=300,
                bbox_inches='tight')

def plot_jet_strength_evolution(exps=['curr-ecc','0-ecc','dust'], \
    smooth=None,ext='png',savedata=False,level=300):
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
                    ds = xr.open_dataset(path+exp_name+'/EDJ.nc', decode_times=False)
                except:
                    continue
                jet_n = ds.jet_n.squeeze()
                jet_s = ds.jet_s.squeeze()

                if smooth is not None:
                    time  = moving_average(ds.time, smooth)
                    jet_n = moving_average(jet_n, smooth)
                    jet_s = moving_average(jet_s, smooth)
                    if exp != '$\gamma = 0.000$':
                        ls    = moving_average(ds.mars_solar_long, smooth)
                else:
                    time = ds.time
                    if exp != '$\gamma = 0.000$':
                        ls = ds.mars_solar_long

                ds = xr.Dataset(data_vars=dict(
                        jet_n = (["time"], jet_n),
                        jet_s = (["time"], jet_s),
                        ls   = (["time"], ls.data),
                    ),
                    coords = dict(
                        time = time.data,
                    ),
                )
                ds.to_netcdf(path+'mars_analysis/jet_strength/%s.nc' % exp_name)

        if exp == '$\gamma = 0.093$' or exp == 'Dust Scale':
            d = xr.open_dataset(path+'mars_analysis/jet_strength/tracer_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.093_cdod_clim_scenario_7.4e-05.nc',
                                decode_times=False)
        elif exp == '$\gamma = 0.000$':
            d = xr.open_dataset(path+'mars_analysis/jet_strength/tracer_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.000_cdod_clim_scenario_7.4e-05.nc',
                                decode_times=False)
        elif exp == 'Dust Vertical Res':
            d = xr.open_dataset(path+'mars_analysis/jet_strength/tracer_vert_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.093_cdod_clim_scenario_7.4e-05.nc',
                                decode_times=False)
        elif exp == 'Longitudinal Dust':
            d = xr.open_dataset(path+'mars_analysis/jet_strength/tracer_soc_mars_mola_topo_lh_eps_' + \
                                '25_gamma_0.093_clim_latlon_7.4e-05.nc',
                                decode_times=False)

        for j in range(len(exp_names)):
            exp_name = exp_names[j]
            try:
                ds = xr.open_dataset(path+'mars_analysis/jet_strength/%s.nc' % exp_name,
                                 decode_times=False)
            except:
                continue
            jet_n = ds.jet_n - d.jet_n
            jet_s = ds.jet_s - d.jet_s

            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_vert_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05'\
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_clim_latlon_7.4e-05':
                jet_n = d.jet_n
                jet_s = d.jet_s

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


            ax.plot(time, jet_n,label=titles[j],color=col,linestyle=lnstl)
            
            
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

            ax.plot(time, jet_s,label=titles[j],color=col,linestyle=lnstl)
            
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
            ax.set_ylabel('jet strength (m s$^{-1}$)')
            if i % 2 == 1:
                ax.legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
                 borderaxespad=0, fontsize='large')
            
            if i == 0:
                ax.set_title('NH')
            elif i == 1:
                ax.set_title('SH')

    for i in range(nrows-1):
        axs[i,0].set_ylim([-57,47])
        axs[i,1].set_ylim([-57,47])
        axs[i,1].hlines(0,time[tind-45],time[tind+120], color='k',alpha=0.5)
    
    axs[-1,0].set_ylim([ 73, 156])
    axs[-1,1].set_ylim([ 73, 156])



    fig.savefig(figpath \
                + 'jet_strength_%s.%s' % (exps[-1],ext), dpi=300,
                bbox_inches='tight')


if __name__ == "__main__":
    eps = np.arange(10,55,5)
    gamma = [0.093,0.00]
    savedata=True
    plot_jet_lat_evolution(    exps=['dust','vert_dust_only','long-dust_only'],smooth=5,
                        ext='pdf',savedata=savedata)
    plot_jet_strength_evolution(exps=['dust','vert_dust_only','long-dust_only'],smooth=5,
                        ext='pdf',savedata=savedata)
# %%
