
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
            if hem == 'nh' or hem == 'NH':
                phi_PV, _ = calc_PV_max(x, x.lat)
            else:
                phi_PV, _ = calc_PV_max(-x, x.lat)                    
            l.append(phi_PV)
        except:
            l.append(np.nan)

    return l

    
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
        path + 'mars_analysis/keffs/attribution_keff_test_tracer.nc', decode_times = False,)

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
        ax = axs, pad = 0.1, aspect=50, orientation='horizontal',
        label='normalized effective diffusivity', extend='max')
    
    #fig.suptitle('Zonal mean cross-section of effective diffusivity, Ls = $%i^\circ$%s' % (ls, mean))
    fig.savefig(figpath+'attribution_xsect_' + \
            '%s_%03d%s.%s' % (hem, tind, sols,ext),
            bbox_inches='tight')

def plot_keff_cross_MY28(hem='nh', PVmax=True, mean=None, tind=101,ext='png'):
    exp_names, titles, nrows, ncols = get_exps('MY28')
    
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
        path + 'mars_analysis/keffs/MY28_keff_test_tracer.nc', decode_times = False,)

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
        
        dis = d_keff
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
        ax = axs, pad = 0.2, orientation = 'horizontal', aspect = 40,
        label='normalized effective diffusivity', extend='max')
    
    #fig.suptitle('Zonal mean cross-section of effective diffusivity, Ls = $%i^\circ$%s' % (ls, mean), y=1.1)
    fig.savefig(figpath+'MY28_xsect_' + \
            '%s_%03d%s.%s' % (hem, tind, sols,ext),
            bbox_inches='tight')


def plot_temp_cross_dust(hem='nh', PVmax=True, mean=None, tind=101,ext='png',res=''):
    if res == '':
        exp_names, titles, nrows, ncols = get_exps('dust')
    elif res == 'long':
        exp_names, titles, nrows, ncols = get_exps('long-dust')
        res = 'long_'
    else:
        exp_names, titles, nrows, ncols = get_exps('high_res_dust')
        res = 'vert_'

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
        if res == 'long_':
            ax.set_title(titles[i])
        else:
            ax.set_title('$\lambda = $' + titles[i])
        
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

        print(exp_name, di.temp.sel(pfull=0.3,method='nearest').sel(lat=-90,method="nearest"))
        
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
    fig.savefig(figpath+'%sdust_temp_xsect_' % res + \
            '%s_%03d%s.%s' % (hem, tind, sols, ext),
            bbox_inches='tight')
    

def plot_temp_cross_MY28(hem='nh', PVmax=True, mean=None, tind=101,ext='png'):
    exp_names, titles, nrows, ncols = get_exps('MY28')
    
    fig, axs = plt.subplots(nrows=1,ncols=1, figsize = (ncols*4,nrows*3.5),)
    
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

        ax.set_title(titles[i])
        
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
    fig.savefig(figpath+'MY28_temp_xsect_' + \
            '%s_%03d%s.%s' % (hem, tind, sols, ext),
            bbox_inches='tight')

def plot_keff_cross_dust(PVmax=True, ext='png',res=''):
    if res == '':
        _, titles, nrows, ncols = get_exps('dust')
    else:
        _, titles, nrows, ncols = get_exps('high_res_dust')
        res = 'vert_'
    nrows = 2
    fig, axs = plt.subplots(nrows=ncols,ncols=nrows, figsize = (nrows*5,ncols*2.5),)
    
    lims = [0,4]

    boundaries, cmap, norm = new_cmap(lims, extend='max', i = 10, override=True, cols='YlGn')

    for i, ax in enumerate(fig.axes):
        ax.text(0, 1.05, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
            size='large')
        ax.ticklabel_format(style='plain')
        ax.set_yscale('log')
        ax.set_ylim([5.5,0.01])
        ax.set_yticks([1,0.1,0.01])
        ax.set_yticklabels([1,0.1,0.01])


    d_n = xr.open_dataset(
        path + 'mars_analysis/winter_vars/%sdust_nh.nc' % (res), decode_times = False,)
    
    pfull = d_n.pfull.values
    lat_n   = d_n.lat.values

    d_s = xr.open_dataset(
        path + 'mars_analysis/winter_vars/%sdust_sh.nc' % (res), decode_times = False,)
    lat_s   = d_s.lat.values
    
    for j in [0,1]:
        if j == 1:
            d = d_n
            lat = lat_n
            hem = 'NH'
        else:
            d = d_s
            lat = lat_s
            hem = 'SH'
        for i in range(ncols):
            di = d.isel(dust_scale=i)
            ax = axs[i,j]
            

            if i == 0:
                ax.set_title(hem)
            if i == ncols-1:
                ax.set_xlabel('latitude ($^\circ$N)')                
            else:
                ax.set_xticklabels([])

            if j == 0:
                ytext = '$\lambda = $' + titles[i]
                ax.text(
                    -0.26, 0.5, ytext,
                    ha='right',
                    va='center',
                    transform=ax.transAxes,
                    rotation='vertical',
                    fontsize='large',
                )
                ax.set_ylabel('pressure (hPa)')
            else:
                ax.set_yticklabels([])

            ax.contourf(lat, pfull, np.log(di.keff),
                cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])
            ax.contour(lat, pfull, di.theta.transpose(),
                    levels = [200,300,400,500,600,700,800,900], colors='k', linestyles='--',linewidths=0.5)

            c0 = ax.contour(lat, pfull, di.ucomp.transpose(),
                    levels=[-50,0,50,100,150], colors='black',linewidths=1)
            c0.levels = [nf(val) for val in c0.levels]
            ax.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
            if PVmax:
                l = get_PV_lats(di, hem=hem)
                ax.plot(l, pfull, linestyle='-', color='xkcd:orchid', linewidth=2.5)   
    plt.tight_layout()
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ticks=boundaries,
        ax = axs, pad = 0.05, aspect = 50,
        label='normalized effective diffusivity', extend='max')
    
    #fig.suptitle('Zonal mean cross-section of effective diffusivity, Ls = $%i^\circ$%s' % (ls, mean), y=1.1)
    fig.savefig(figpath+'%sdust_xsect' % res + \
            '.%s' % (ext),
            bbox_inches='tight')

def plot_keff_cross_parameter(hem='nh', half=False, PVmax=True, ext='png'):
    eps = [10,15,20,25,30,35,40,45,50]
    if half == True:
        eps = [10,20,30,40,50]
    gamma = [0.093,0]

    fig, axs = plt.subplots(nrows=len(eps),ncols=2, figsize = (11,len(eps)*2.5),)
    
    lims = [0,4]

    boundaries, cmap, norm = new_cmap(lims, extend='max', i = 10, override=True, cols='YlGn')

    for i, ax in enumerate(fig.axes):
        ax.text(-0.025, 1.05, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
            size='large')
        ax.ticklabel_format(style='plain')
        ax.set_yscale('log')
        ax.set_ylim([5.5,0.01])
        ax.set_yticks([1,0.1,0.01])
        ax.set_yticklabels([1,0.1,0.01])

    

    dkeff = xr.open_dataset(
        	path + 'mars_analysis/winter_vars/parameter_%s.nc' % hem,
            decode_times = False,)
    lat = dkeff.lat.values
    pfull = dkeff.pfull.values

    for j in range(len(eps)):
        for i in range(len(gamma)):
            exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps[j], gamma[i])

            ax = axs[j,i]
            ds = dkeff.sel(epsilon=eps[j]).sel(gamma=gamma[i])
            

            if j == 0:
                ax.set_title('$\gamma = %.3f$' % gamma[i])
            
            if eps[j] == eps[-1]:
                ax.set_xlabel('latitude ($^\circ$N)')
            else:
                ax.set_xticklabels([])
            
            if i == 0:
                ytext = '$\epsilon = %i^\circ$' % eps[j]
                ax.text(
                    -0.16, 0.5, ytext,
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

            
            c1=ax.contourf(lat, pfull, np.log(ds.keff),
                    cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])

            ax.contour(lat, pfull, ds.theta.transpose(),
                    levels = [200,300,400,500,600,700,800,900], colors='k', linestyles='--',linewidths=0.5)

            c0 = ax.contour(lat, pfull, ds.ucomp.transpose(),
                    levels=[-50,0,50,100,150], colors='black',linewidths=1)
            c0.levels = [nf(val) for val in c0.levels]
            ax.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
            if PVmax:
                l = get_PV_lats(ds, hem=hem)
                ax.plot(l, ds.pfull, linestyle='-', color='xkcd:orchid', linewidth=2.5)
    
    plt.tight_layout()
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ticks=boundaries[slice(None,None,2)],
        ax = axs, pad = 0.05,aspect=50,
        label='normalized effective diffusivity', extend='max')
    #fig.suptitle('Zonal mean cross-section of effective diffusivity, Ls = $%i^\circ$%s' % (ls, mean))
    fig.savefig(figpath+'parameter_' + \
        'xsect_%s.%s' % (hem, ext),
            bbox_inches='tight')


def plot_keff_cross_latlon(hem='nh', PVmax=True, ext='png',res=''):
    _, titles, nrows, ncols = get_exps('long-dust')
    
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize = (ncols*4,nrows*3.8),)
    
    lims = [0,4]

    boundaries, cmap, norm = new_cmap(lims, extend='max', i = 10, override=True, cols='YlGn')

    for i, ax in enumerate(fig.axes):
        ax.text(0, 1.05, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
            size='large')
        ax.ticklabel_format(style='plain')
        ax.set_yscale('log')
        ax.set_ylim([5.5,0.01])
        ax.set_yticks([1,0.1,0.01])
        ax.set_yticklabels([1,0.1,0.01])

    d = xr.open_dataset(
        path + 'mars_analysis/winter_vars/latlon_dust_%s.nc' % (hem), decode_times = False,)
    
    pfull = d.pfull.values
    lat   = d.lat.values

    for i, ax in enumerate(fig.axes):
        di = d.isel(long_dust=i)
        ax.set_xlabel('latitude ($^\circ$N)')

        ax.set_title('$\lambda = $' + titles[i])
        
        if i % ncols == 0:
            ax.set_ylabel('pressure (hPa)')
            
        else:
            ax.set_yticklabels([])
        
        ax.contourf(lat, pfull, np.log(di.keff),
            cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])
        ax.contour(lat, pfull, di.theta.transpose(),
                levels = [200,300,400,500,600,700,800,900], colors='k', linestyles='--',linewidths=0.5)

        c0 = ax.contour(lat, pfull, di.ucomp.transpose(),
                levels=[-50,0,50,100,150], colors='black',linewidths=1)
        c0.levels = [nf(val) for val in c0.levels]
        ax.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
        if PVmax:
            l = get_PV_lats(di, hem=hem)
            ax.plot(l, pfull, linestyle='-', color='xkcd:orchid', linewidth=2.5)   
    
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ticks=boundaries[slice(None,None,2)],
        ax = axs, pad = 0.2, orientation = 'horizontal', aspect = 50,
        label='normalized effective diffusivity', extend='max')
    
    #fig.suptitle('Zonal mean cross-section of effective diffusivity, Ls = $%i^\circ$%s' % (ls, mean), y=1.1)
    fig.savefig(figpath+'latlon_xsect_' + \
            '%s.%s' % (hem,ext),
            bbox_inches='tight')


if __name__ == "__main__":
    exps = 'parameter'
    #plot_keff_cross_dust(tind = 450,mean=10,hem='sh',ext='pdf',res='')
    #plot_keff_cross_latlon(hem='sh',ext='pdf',res='')
    #plot_keff_cross_dust(ext='pdf',res='')
    #plot_keff_cross_dust(tind=115,mean=10,hem='nh',ext='pdf',res='vert_')
    #plot_keff_cross_latlon(hem='nh',ext='pdf',res='')
    #plot_keff_cross_dust(hem='nh',ext='pdf',res='')
    #plot_keff_cross_parameter(tind=450,mean=10,hem='sh',half=True,ext='pdf')
    #plot_keff_cross_parameter(hem='sh',half=True,ext='pdf')
    #plot_keff_cross_parameter(hem='nh',half=True,ext='pdf')
    plot_temp_cross_dust(tind=450,mean=10,hem='sh', ext='pdf', res='long')
    #plot_temp_cross_dust(tind=115,mean=10,hem='nh', ext='pdf', res='long')
    #plot_keff_cross_MY28(tind=450,mean=10,hem='sh', ext='pdf')
    #plot_keff_cross_MY28(tind=115,mean=10,hem='nh', ext='pdf')
    #plot_keff_cross_dust(tind = 115,mean=10,hem='nh')
    #plot_keff_cross_dust(mean=10)
    #plot_keff_cross_parameter(tind=115,mean=10,hem='nh',half=True,ext='pdf')
# %%
