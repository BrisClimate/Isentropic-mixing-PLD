
# %%
import xarray as xr
import numpy as np
import sys, os

sys.path.append('../')

from atmospy import stereo_plot, lait, calc_PV_max, new_cmap, \
                    get_timeslice, nf

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


def get_PV_lats(di, hem='nh'):
    '''
    Lait-scale PV and then return the latitude of maximum PV on given 
    pressure levels'''
    laitPV = lait(di.PV,di.theta,theta0,kappa=kappa)

    l = []
    for a in range(len(di.pfull)):
        try:
            x = laitPV.isel(pfull=a)
        
            x = x.where(x != np.nan, drop = True)
            if hem == 'nh':
                phi_PV, _ = calc_PV_max(x, x.lat)
            else:
                phi_PV, _ = calc_PV_max(-x, x.lat)                    
            l.append(phi_PV)
        except:
            l.append(np.nan)

    return l

def plot_keff_cross_parameter(hem='nh', half=False, PVmax=True, mean=None, tind=101,ext='png'):
    eps = [10,15,20,25,30,35,40,45,50]
    if half == True:
        eps = [10,20,30,40,50]
    gamma = [0.093,0]

    fig, axs = plt.subplots(nrows=2,ncols=len(eps), figsize = (len(eps)*3,7),)
    
    lims = [0,4]

    boundaries, cmap, norm = new_cmap(lims, extend='max', i = 10, override=True, cols='YlGn')

    for i, ax in enumerate(fig.axes):
        ax.text(0.05, 1.05, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
            size='large')
        ax.set_yscale('log')
        ax.set_ylim([5.5,0.01])

    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ticks=boundaries[slice(None,None,2)],
        ax = axs, pad = 0.01,
        label='normalized effective diffusivity', extend='max')

    dkeff = xr.open_dataset(
        	path + 'mars_analysis/' + exps + '_keff_test_tracer.nc',
            decode_times = False,)

    for j in range(len(eps)):
        for i in range(len(gamma)):
            exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps[j], gamma[i])

            ax = axs[i,j]
            ds = dkeff.sel(epsilon=eps[j]).sel(gamma=gamma[i])
            try:
                d = xr.open_dataset(
        	        path + exp_name + '/atmos.nc', decode_times = False,)
                print(exp_name)

                d = d[["theta", "PV", "ucomp", "mars_solar_long"]]
            except:
                continue

            if i == 0:
                ls = d.mars_solar_long.isel(time=tind).values

                ax.set_title('$\epsilon = %i^\circ$' % eps[j])
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('latitude ($^\circ$N)')
            
            if j == 0:
                ytext = '$\gamma = %.3f$' % gamma[i]
                ax.text(
                    -0.46, 0.5, ytext,
                    ha='right',
                    va='center',
                    transform=ax.transAxes,
                    rotation='vertical',
                    fontsize='large',
                )
                ax.set_ylabel('pressure (hPa)')
            else:
                ytext=None
                ax.set_yticklabels([])

            tind, m = get_timeslice(tind, mean)            

            if hem == 'nh':
                d  =  d.where( d.lat >= 0, drop = True)
                ds = ds.where(ds.new >= 0, drop = True)
            else:
                d  =  d.where( d.lat <= 0, drop = True)
                ds = ds.where(ds.new <= 0, drop = True)

            if m != 0:
                tslice = slice(tind-m, tind+m)
                
                di  =  d.isel(time=tslice)
                di  =  di.mean(dim="time")
                dis = ds.nkeff.isel(time=tslice)
                dis = dis.mean(dim="time")
            else:
                di  =  d.isel(time=tind)
                dis = ds.nkeff.isel(time=tind)

            dis = dis.where(dis.level <= 5.5, drop = True)
            di  =  di.where(di.pfull  <= 5.5, drop = True)
            c1=ax.contourf(dis.new, dis.level, np.log(dis),
                    cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])

            ax.contour(di.lat, di.pfull, di.theta.mean(dim=["lon"]).transpose(),
                    levels = [200,300,400,500,600,700,800,900], colors='k', linestyles='--',linewidths=0.5)

            c0 = ax.contour(di.lat, di.pfull, di.ucomp.mean(dim=["lon"]).transpose(),
                    levels=[-50,0,50,100,150], colors='black',linewidths=1)
            c0.levels = [nf(val) for val in c0.levels]
            ax.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
            if PVmax:
                l = get_PV_lats(di.mean(dim=["lon"]), hem=hem)
                ax.plot(l, di.pfull, linestyle='-', color='xkcd:orchid', linewidth=2.5)
    
    if mean is not None:
        mean = ' (%i sol average)' % mean
        sols = '_%i-%isai' % ((tind-m)%30,(tind+m)%30)
    else:
        mean = ''
        sols = '_%isai' % (tind % 30)
    
    if half == True:
        half = '10deg_'
    else:
        half = ''
    
    #fig.suptitle('Zonal mean cross-section of effective diffusivity, Ls = $%i^\circ$%s' % (ls, mean))
    fig.savefig(figpath+'parameter_%s' %half + \
        'xsect_%s_%03d%s.%s' % (hem, tind, sols, ext),
            bbox_inches='tight')

    
    
def plot_keff_cross_attribution(hem='nh', PVmax=True, mean=None, tind=101,ext='png'):
    
    exp_names, titles, nrows, ncols = get_exps('attribution')
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize = (ncols*3,nrows*3),)
    
    lims = [0,4]

    boundaries, cmap, norm = new_cmap(lims, extend='max', i = 10, override=True, cols='YlGn')

    for i, ax in enumerate(fig.axes):
        ax.text(0, 1.05, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
            size='large')
        ax.set_yscale('log')
        ax.set_ylim([5.5,0.01])

    tind, m = get_timeslice(tind, mean)  
    d_keff = xr.open_dataset(
        path + 'mars_analysis/attribution_keff_test_tracer.nc', decode_times = False,)

    if hem == 'nh':
        d_keff = d_keff.where(d_keff.new >= 0, drop = True)
    else:
        d_keff = d_keff.where(d_keff.new <= 0, drop = True)

    if m != 0:
        tslice = slice(tind-m, tind+m)
        
        d_keff = d_keff.nkeff.isel(time=tslice)
        d_keff = d_keff.mean(dim="time")
    else:
        d_keff = d_keff.nkeff.isel(time=tind)

    d_keff = d_keff.where(d_keff.level <= 5.5, drop = True)

    for i, ax in enumerate(fig.axes):
        exp_name = exp_names[i]
        try:
            d = xr.open_dataset(
                path + exp_name + '/atmos.nc', decode_times = False,)
            
            print(exp_name)

            d = d[["theta", "PV", "ucomp", "mars_solar_long"]]
            
        except:
            continue
        
        if i % 4 == 0 or i % 4 == 2:
            ds = d_keff.sel(lh=0)
        else:
            ds = d_keff.sel(lh=1)
        if i % 4 == 0 or i % 4 == 1:
            ds = ds.sel(dust=0)
        else:
            ds = ds.sel(dust=1)
        if i < 4:
            dis = ds.sel(topo=0)
            ax.set_xticklabels([])
        else:
            dis = ds.sel(topo=1)
            
            ax.set_xlabel('latitude ($^\circ$N)')


        ls = d.mars_solar_long.isel(time=tind).values

        ax.set_title(titles[i])
        
        if i % 4 == 0:
            ax.set_ylabel('pressure (hPa)')
        else:
            ax.set_yticklabels([])


        if hem == 'nh':
            d  =  d.where( d.lat >= 0, drop = True)
        else:
            d  =  d.where( d.lat <= 0, drop = True)

        if m != 0:
            tslice = slice(tind-m, tind+m)
            
            di  =  d.isel(time=tslice)
            di  =  di.mean(dim="time")
        else:
            di  =  d.isel(time=tind)

        
        di  =  di.where(di.pfull  <= 5.5, drop = True)
        di = di.mean(dim=["lon"])
        c1=ax.contourf(dis.new, dis.level, np.log(dis),
                cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])

        pfull = di.pfull.values
        lat   =   di.lat.values
        ax.contour(lat, pfull, di.theta.transpose(),
                levels = [200,300,400,500,600,700,800,900], colors='k', linestyles='--',linewidths=0.5)

        c0 = ax.contour(lat, pfull, di.ucomp.transpose(),
                levels=[-50,0,50,100,150], colors='black',linewidths=1)
        c0.levels = [nf(val) for val in c0.levels]
        ax.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
        if PVmax:
            l = get_PV_lats(di, hem=hem)
            ax.plot(l, pfull, linestyle='-', color='xkcd:orchid', linewidth=2.5)
    
    if mean is not None:
        mean = ' (%i sol average)' % mean
        sols = '_%i-%isai' % ((tind-m)%30,(tind+m)%30)
    else:
        mean = ''
        sols = '_%isai' % (tind % 30)
    
    
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ticks=boundaries[slice(None,None,2)],
        ax = axs, pad = 0.01,
        label='normalized effective diffusivity', extend='max')
    
    #fig.suptitle('Zonal mean cross-section of effective diffusivity, Ls = $%i^\circ$%s' % (ls, mean))
    fig.savefig(figpath+'attribution_xsect_' + \
            '%s_%03d%s.%s' % (hem, tind, sols,ext),
            bbox_inches='tight')

    

def plot_keff_cross_dust(hem='nh', PVmax=True, mean=None, tind=101,ext='png'):
    
    exp_names, titles, nrows, ncols = get_exps('dust')
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize = (ncols*4,nrows*3.5),)
    
    lims = [0,4]

    boundaries, cmap, norm = new_cmap(lims, extend='max', i = 10, override=True, cols='YlGn')

    for i, ax in enumerate(fig.axes):
        ax.text(0, 1.05, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
            size='large')
        ax.set_yscale('log')
        ax.set_ylim([5.5,0.01])

    tind, m = get_timeslice(tind, mean)  
    d_keff = xr.open_dataset(
        path + 'mars_analysis/dust_keff_test_tracer.nc', decode_times = False,)

    if hem == 'nh':
        d_keff = d_keff.where(d_keff.new >= 0, drop = True)
    else:
        d_keff = d_keff.where(d_keff.new <= 0, drop = True)

    if m != 0:
        tslice = slice(tind-m, tind+m)
        
        d_keff = d_keff.nkeff.isel(time=tslice)
        d_keff = d_keff.mean(dim="time")
    else:
        d_keff = d_keff.nkeff.isel(time=tind)

    d_keff = d_keff.where(d_keff.level <= 5.5, drop = True)

    for i, ax in enumerate(fig.axes):
        exp_name = exp_names[i]
        try:
            d = xr.open_dataset(
                path + exp_name + '/atmos.nc', decode_times = False,)
            
            print(exp_name)

            d = d[["theta", "PV", "ucomp", "mars_solar_long"]]
            
        except:
            continue
        
        dis = d_keff.isel(dust_scale=i)
        ax.set_xlabel('latitude ($^\circ$N)')


        ls = d.mars_solar_long.isel(time=tind).values

        ax.set_title('Dust Scaling = ' + titles[i])
        
        if i % 4 == 0:
            ax.set_ylabel('pressure (hPa)')
        else:
            ax.set_yticklabels([])


        if hem == 'nh':
            d  =  d.where( d.lat >= 0, drop = True)
        else:
            d  =  d.where( d.lat <= 0, drop = True)

        if m != 0:
            tslice = slice(tind-m, tind+m)
            
            di  =  d.isel(time=tslice)
            di  =  di.mean(dim="time")
        else:
            di  =  d.isel(time=tind)

        
        di  =  di.where(di.pfull  <= 5.5, drop = True)
        di = di.mean(dim=["lon"])
        c1=ax.contourf(dis.new, dis.level, np.log(dis),
                cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])

        pfull = di.pfull.values
        lat   =   di.lat.values
        ax.contour(lat, pfull, di.theta.transpose(),
                levels = [200,300,400,500,600,700,800,900], colors='k', linestyles='--',linewidths=0.5)

        c0 = ax.contour(lat, pfull, di.ucomp.transpose(),
                levels=[-50,0,50,100,150], colors='black',linewidths=1)
        c0.levels = [nf(val) for val in c0.levels]
        ax.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
        if PVmax:
            l = get_PV_lats(di, hem=hem)
            ax.plot(l, pfull, linestyle='-', color='xkcd:orchid', linewidth=2.5)
    
    if mean is not None:
        mean = ' (%i sol average)' % mean
        sols = '_%i-%isai' % ((tind-m)%30,(tind+m)%30)
    else:
        mean = ''
        sols = '_%isai' % (tind % 30)
    
    
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ticks=boundaries[slice(None,None,2)],
        ax = axs, pad = 0.01,
        label='normalized effective diffusivity', extend='max')
    
    #fig.suptitle('Zonal mean cross-section of effective diffusivity, Ls = $%i^\circ$%s' % (ls, mean), y=1.1)
    fig.savefig(figpath+'dust_xsect_' + \
            '%s_%03d%s.%s' % (hem, tind, sols,ext),
            bbox_inches='tight')


def plot_temp_cross_dust(hem='nh', PVmax=True, mean=None, tind=101,ext='png'):
    
    exp_names, titles, nrows, ncols = get_exps('dust')
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize = (ncols*4,nrows*3.5),)
    
    if hem == 'nh':
        lims = [150,240]
    else:
        lims = [120,210]
    boundaries, cmap, norm = new_cmap(lims, extend='both', i = 10, override=True, cols='RdBu_r')

    for i, ax in enumerate(fig.axes):
        ax.text(0, 1.05, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
            size='large')
        ax.set_yscale('log')
        ax.set_ylim([5.5,0.01])

    tind, m = get_timeslice(tind, mean)  
    
    for i, ax in enumerate(fig.axes):
        exp_name = exp_names[i]
        try:
            d = xr.open_dataset(
                path + exp_name + '/atmos.nc', decode_times = False,)
            
            print(exp_name)

            d = d[["theta", "PV", "ucomp", "mars_solar_long", "temp"]]
            
        except:
            continue
        
        ax.set_xlabel('latitude ($^\circ$N)')

        ls = d.mars_solar_long.isel(time=tind).values

        ax.set_title('Dust Scaling = ' + titles[i])
        
        if i % 4 == 0:
            ax.set_ylabel('pressure (hPa)')
        else:
            ax.set_yticklabels([])

        di = d.where(d.pfull  <= 5.5, drop = True)
        di = di.mean(dim=["lon"])

        if hem == 'nh':
            di  =  di.where( di.lat >= 0, drop = True)
        else:
            di  =  di.where( di.lat <= 0, drop = True)

        if m != 0:
            tslice = slice(tind-m, tind+m)
            
            di  =  di.isel(time=tslice)
            di  =  di.mean(dim="time")
        else:
            di  =  di.isel(time=tind)

        
        
        c1=ax.contourf(di.lat, di.pfull, di.temp.transpose(),
                cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])

        pfull = di.pfull.values
        lat   =   di.lat.values
        ax.contour(lat, pfull, di.theta.transpose(),
                levels = [200,300,400,500,600,700,800,900], colors='k', linestyles='--',linewidths=0.5)

        c0 = ax.contour(lat, pfull, di.ucomp.transpose(),
                levels=[-50,0,50,100,150], colors='black',linewidths=1)
        c0.levels = [nf(val) for val in c0.levels]
        ax.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
        if PVmax:
            l = get_PV_lats(di, hem=hem)
            ax.plot(l, pfull, linestyle='-', color='xkcd:orchid', linewidth=2.5)
    
    if mean is not None:
        mean = ' (%i sol average)' % mean
        sols = '_%i-%isai' % ((tind-m)%30,(tind+m)%30)
    else:
        mean = ''
        sols = '_%isai' % (tind % 30)
    
    
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ticks=boundaries[slice(None,None,2)],
        ax = axs, pad = 0.01,
        label='temperature (K)', extend='both')
    
    #fig.suptitle('Zonal mean cross-section of temperature, Ls = $%i^\circ$%s' % (ls, mean), y=1.1)
    fig.savefig(figpath+'dust_temp_xsect_' + \
            '%s_%03d%s.%s' % (hem, tind, sols, ext),
            bbox_inches='tight')



if __name__ == "__main__":
    exps = 'parameter'
    plot_keff_cross_dust(tind = 450,mean=10,hem='sh',ext='pdf')
    plot_keff_cross_dust(tind=115,mean=10,hem='nh',ext='pdf')
    plot_keff_cross_parameter(tind=450,mean=10,hem='sh',half=True,ext='pdf')
    plot_temp_cross_dust(tind=450,mean=10,hem='sh', ext='pdf')
    #plot_keff_cross_dust(tind = 115,mean=10,hem='nh')
    #plot_keff_cross_dust(mean=10)
    plot_keff_cross_parameter(tind=115,mean=10,hem='nh',half=True,ext='pdf')
# %%
