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

figpath = '/user/home/xz19136/Figures/mars_analysis/keff/'
path = '/user/work/xz19136/Isca_data/'
theta, center, radius, verts, circle = atmospy.stereo_plot()
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

def get_PV_lats(PV, hem='nh'):
    '''
    Lait-scale PV and then return the latitude of maximum PV on given 
    pressure levels'''
    

    l = []
    for a in range(len(PV.time)):
        #try:
        x = PV.isel(time=a)
        
        x = x.where(x != np.nan, drop = True)
        if hem == 'nh':
            phi_PV, _ = atmospy.calc_jet_lat(x, x.lat)
        else:
            phi_PV, _ = atmospy.calc_jet_lat(-x, x.lat)                    
        l.append(phi_PV)
        #except:
        #    l.append(np.nan)

    return l


def plot_keff_lat(exps=['curr-ecc','0-ecc','dust'], \
        level=300,mean=None,tind=101):
    '''
    Plot effective diffusivity on a given surface,
    in order to understand strength of the transport barrier and mixing
    within the vortex.
    '''
    tind = tind % 360
    tind, m = atmospy.get_timeslice(tind, mean)
    nrows = len(exps)
    if exps.count('MY28'):
        nrows -= 1  
    fig, axs = plt.subplots(nrows=nrows,ncols=2, figsize = (15,4*len(exps)),dpi=300)

    ds = xr.open_dataset(path+'mars_analysis/keffs/parameter_keff_test_tracer_isentropic.nc',
                decode_times=False)
    
    ds0 = ds.sel(level = level, method = "nearest")    
    

    ds = xr.open_dataset(path+'mars_analysis/keffs/dust_keff_test_tracer_isentropic.nc',
                decode_times=False)
    ds1 = ds.sel(level = level, method = "nearest")
    level = int(ds0.level.values)
    ls_n = ds.mars_solar_long.isel(time=tind).isel(dust_scale=0).values
    ls_s = ds.mars_solar_long.isel(time=tind+360).isel(dust_scale=0).values
    ds = xr.open_dataset(path+'mars_analysis/keffs/vert_dust_keff_test_tracer_isentropic.nc',
                decode_times=False)
    ds2 = ds.sel(level = level, method = "nearest")

    ds = xr.open_dataset(path+'mars_analysis/keffs/MY28_keff_test_tracer_isentropic.nc',
                decode_times=False)
    ds3 = ds.sel(level = level, method = "nearest")
    ds3 = ds3.isel(time = slice(60,None))

    

    phi_eff = ds0.new.values
    
    for i in range(len(exps)):
        exp = exps[i]
        exp_names, titles, _, _ = atmospy.get_exps(exp)
        
        colors = plt.cm.viridis(np.linspace(0,1,int(len(exp_names)-1)))
        if exp == 'curr-ecc':
            exp = '$\gamma = 0.093$'
            d_keff = ds0.sel(gamma=0.093)
        elif exp == '0-ecc':
            exp = '$\gamma = 0.000$'
            d_keff = ds0.sel(gamma=0.000)
        elif exp == 'dust':
            exp = 'Dust Scale'
            d_keff = ds1
        elif exp == 'high_res_dust':
            exp = 'Dust Vertical Res'
            d_keff = ds2
        elif exp == 'MY28':
            d_keff = ds3
        elif exp == 'attribution':
            exp = 'Attribution'

        if m != 0:
            tslice = slice(tind-m, tind+m)
        
            d_keff_n = d_keff.nkeff.isel(time=tslice)
            d_keff_n = d_keff_n.mean(dim="time")
            
            tslice = slice(tind+360-m, tind+360+m)
            d_keff_s = d_keff.nkeff.isel(time=tslice)
            d_keff_s = d_keff_s.mean(dim="time")
        else:
            d_keff_s = d_keff.nkeff.isel(time=tind+360)
            d_keff_n = d_keff.nkeff.isel(time=tind)

        l = 0
        for j in range(len(exp_names)):

            exp_name = exp_names[j]
            print(exp_name)
            
            try:
                keff_n = d_keff_n.isel(epsilon=j)
                keff_s = d_keff_s.isel(epsilon=j)
            except:
                try:
                    keff_n = d_keff_n.isel(dust_scale=j)
                    keff_s = d_keff_s.isel(dust_scale=j)
                except:
                    keff_n = d_keff_n
                    keff_s = d_keff_s

            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05'\
                    or exp_name == 'tracer_vert_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05':
                col = 'k'
                lnstl = '--'
            elif exp_name == 'tracer_MY28_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05':
                col = 'xkcd:crimson'
                lnstl = '-.'
            else:
                lnstl = '-'
                col = colors[l]
                l += 1

            if exp == 'MY28':
                ax = axs[i-2,0]
            else:
                ax = axs[i,0]

            ax.plot(phi_eff, np.log(keff_n),label=titles[j],color=col,linestyle=lnstl)

            ax.set_xlim([40,90])
            ax.set_ylim([0, 4.5])
            if exp != 'MY28':
                ax.text(
                    -0.16, 0.5, exp,
                    ha='right',
                    va='center',
                    transform=ax.transAxes,
                    rotation='vertical',
                    fontsize='large',
                )
            if exp == 'MY28':
                ax = axs[i-2,1]
            else:
                ax = axs[i,1]
            ax.plot(phi_eff, np.log(keff_s),label=titles[j],color=col,linestyle=lnstl)

            ax.set_xlim([-40,-90])
            ax.set_ylim([0, 4.5])

        for i, ax in enumerate(fig.axes):
            ax.text(0.0, 1.03, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
                size='large')
            
            if i < 2*len(exps)-2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('effective latitude',fontsize='large')
            
            if exps.count('MY28') and i >= 2*len(exps)-6:
                ax.set_xlabel('effective latitude',fontsize='large')
            if i % 2 == 1:
                ax.legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
                 borderaxespad=0, fontsize='large')
            else:
                ax.set_ylabel('effective diffusivity')
            
            

    if mean is not None:
        mean = ' (%i sol average)' % mean
        sols = '_%i-%isai' % ((tind-m)%30,(tind+m)%30)
    else:
        mean = ''
        sols = '_%isai' % (tind % 30)

    for i, ax in enumerate(fig.axes):
        if i == 0:
            ax.set_title('NH, Ls = $%i^\circ$%s' %(ls_n, mean))
        elif i == 1:
            ax.set_title('SH, Ls = $%i^\circ$%s' %(ls_s, mean))

    fig.suptitle('Effective diffusivity at ' + \
        '%iK' % level,
        y=0.95,
        fontsize='x-large')

    fig.savefig(figpath \
                + 'keff_vs_lat_%iK_%03d%s.png' % (level, tind, sols), dpi=300,
                bbox_inches='tight')


def plot_keff_lat_PV(exps=['curr-ecc','0-ecc','dust'], \
        level=300,mean=None,tind=101,half=False,ext='png'):
    '''
    Plot effective diffusivity on a given surface,
    in order to understand strength of the transport barrier and mixing
    within the vortex.
    '''
    tind = tind % 360
    tind, m = atmospy.get_timeslice(tind, mean)  
    fig1, axs1 = plt.subplots(nrows=len(exps),ncols=3, figsize = (20,4*len(exps)),dpi=300)
    fig2, axs2 = plt.subplots(nrows=len(exps),ncols=3, figsize = (20,4*len(exps)),dpi=300)

    ds = xr.open_dataset(path+'parameter_keff_test_tracer_isentropic.nc',
                decode_times=False)
    
    ds0 = ds.sel(level = level, method = "nearest")    
    

    ds = xr.open_dataset(path+'dust_keff_test_tracer_isentropic.nc',
                decode_times=False)
    ds1 = ds.sel(level = level, method = "nearest")

    level = int(ds0.level.values)
    ls_n = ds.mars_solar_long.isel(time=tind).isel(dust_scale=0).values
    ls_s = ds.mars_solar_long.isel(time=tind+360).isel(dust_scale=0).values

    phi_eff_n = ds0.new.where(ds0.new>40,drop=True).values
    phi_eff_s = ds0.new.where(ds0.new<-40,drop=True).values
    
    for i in range(len(exps)):
        exp = exps[i]
        exp_names, titles, _, _ = atmospy.get_exps(exp)
        
        colors = plt.cm.viridis(np.linspace(0,1,int(len(exp_names)-1)))
        
        if exp == 'curr-ecc':
            exp = '$\gamma = 0.093$'
            d_keff = ds0.sel(gamma=0.093)
        elif exp == '0-ecc':
            exp = '$\gamma = 0.000$'
            d_keff = ds0.sel(gamma=0.000)
        elif exp == 'dust':
            exp = 'Dust Scale'
            d_keff = ds1
        elif exp == 'attribution':
            exp = 'Attribution'

        if exp != 'Dust Scale' and exp != 'Attribution' and half == True:
            colors = plt.cm.viridis(np.linspace(0,1,int(len(exp_names)/4+1)))
        
        if m != 0:
            tslice = slice(tind-m, tind+m)
        
            d_keff_n = d_keff.nkeff.isel(time=tslice)
            d_keff_n = d_keff_n.mean(dim="time").where(d_keff_n.new >40, drop=True)
            
            tslice = slice(tind+360-m, tind+360+m)
            d_keff_s = d_keff.nkeff.isel(time=tslice)
            d_keff_s = d_keff_s.mean(dim="time").where(d_keff_s.new<-40, drop=True)
        else:
            d_keff_s = d_keff.nkeff.isel(time=tind+360).where(d_keff.new >40, drop=True)
            d_keff_n = d_keff.nkeff.isel(time=tind).where(d_keff.new<-40, drop=True)

        l = 0
        for j in range(len(exp_names)):

            exp_name = exp_names[j]
            print(exp_name)
            if exp != 'Dust Scale' and exp != 'Attribution' and half == True:
                if j % 4 == 1 or j % 4 == 2 or j % 4 == 3:
                    continue
            d = xr.open_dataset(
                path + exp_name + '/atmos_isentropic.nc', decode_times = False,)
            

            d = d[["PV", "ucomp", "mars_solar_long"]].sel(level=level,method="nearest")

            p = xr.open_dataset(
                path + exp_name + '/psi.nc', decode_times = False,)
            p = p.psi.sel(pfull=50,method="nearest")

            if m != 0:
                tslice = slice(tind-m, tind+m)
                d_n = d.isel(time=tslice)
                d_n = d_n.mean(dim="time")

                p_n = p.isel(time=tslice)
                p_n = p_n.mean(dim="time")

                tslice = slice(tind+360-m, tind+360+m)
                d_s = d.isel(time=tslice)
                d_s = d_s.mean(dim="time")

                p_s = p.isel(time=tslice)
                p_s = p_s.mean(dim="time")
            else:
                d_s = d.isel(time=tind+360)
                d_n = d.isel(time=tind)

                p_s = p.isel(time=tind+360)
                p_n = p.isel(time=tind)
            
            try:
                keff_n = d_keff_n.isel(epsilon=j)
                keff_s = d_keff_s.isel(epsilon=j)
            except:
                keff_n = d_keff_n.isel(dust_scale=j)
                keff_s = d_keff_s.isel(dust_scale=j)

            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05':
                col = 'k'
            else:
                col = colors[l]
                l += 1
            d_n = d_n.mean(dim="lon").where(d_n.lat>40,drop=True)
            d_s = d_s.mean(dim="lon").where(d_s.lat<-40,drop=True)

            p_n = p_n.where(p_n.lat>40,drop=True)/10**8
            p_s = p_s.where(p_s.lat<-40,drop=True)/10**8

            for ax in [axs1[i,0],axs2[i,0]]:
                ax.set_xlim([40,90])
                ax.text(
                    -0.16, 0.5, exp,
                    ha='right',
                    va='center',
                    transform=ax.transAxes,
                    rotation='vertical',
                    fontsize='large',
                )
                ax.set_ylabel('$\kappa_{\\rm eff}$')
            axs1[i,0].plot(phi_eff_n, np.log(keff_n),label=None,color=col,linestyle='-')
            axs2[i,0].plot(phi_eff_s, np.log(keff_s),label=None,color=col,linestyle='-')
            axs2[i,0].set_xlim([-40,-90])
            for ax in [axs1[i,1],axs2[i,1]]:
                ax.set_xlim([40,90])
                ax.set_ylabel('PV (MPVU)')

            axs1[i,1].plot(d_n.lat.values, d_n.PV*10**4, label=None,color=col,linestyle='-')
            axs2[i,1].plot(d_s.lat.values, -d_s.PV*10**4,label=None,color=col,linestyle='-')
            axs2[i,1].set_xlim([-40,-90])
            for ax in [axs1[i,2],axs2[i,2]]:
                ax.set_xlim([40,90])
                ax.set_ylabel('$u$ (ms$^{-1}$)')
            axs1[i,2].plot(d_n.lat.values, d_n.ucomp, label=titles[j],color=col,linestyle='-')
            axs2[i,2].plot(d_s.lat.values, d_s.ucomp, label=titles[j],color=col,linestyle='-')
            axs2[i,2].set_xlim([-40,-90])
            #for ax in [axs1[i,3],axs2[i,3]]:
            #ax.set_xlim([40,90])
            #ax.set_ylabel('$\psi_{50}$ ($10^8$kg s$^{-1}$)')
            #ax.plot( p_n.lat.values, p_n, label=titles[j],color=col,linestyle='-')
            #ax.plot(-p_s.lat.values,-p_s, label=None,color=col,linestyle='--')
            #ax.set_ylim([-10,10])
            
        l1 = []
        l2 = []
        l3 = []
        for fig in [fig1,fig2]:
            
            for i, ax in enumerate(fig.axes):
                if i % 3 == 0:
                    l1.append(ax.get_ylim()[0])
                    l1.append(ax.get_ylim()[1])
                elif i % 3 == 1:
                    l2.append(ax.get_ylim()[0])
                    l2.append(ax.get_ylim()[1])
                elif i % 3 == 2:
                    l3.append(ax.get_ylim()[0])
                    l3.append(ax.get_ylim()[1])
                ax.text(0.0, 1.03, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
                    size='large')

                if i < 2*len(exps):
                    ax.set_xticklabels([])
                else:

                    ax.set_xlabel('latitude/equivalent latitude ($^\circ$N)',fontsize='large')

                if i % 3 == 2:
                    ax.legend(loc='center left', bbox_to_anchor=(1.07,0.5,),
                     borderaxespad=0, fontsize='large')
                #else:
                #    ax.set_ylabel('$\kappa_{\\rm eff}$, PV, $\psi$')
                if i == 0:
                    ax.set_title('Effective diffusivity')
                elif i == 1:
                    ax.set_title('Zonal mean Potential Vorticity')
                elif i == 2:
                    ax.set_title('Zonal mean zonal wind')
                elif i == 1:
                    ax.set_title('Meridional overturning streamfunction')
        for fig in [fig1,fig2]:
            for i, ax in enumerate(fig.axes):
                if i % 3 == 0:
                    ax.set_ylim([0,np.max(l1)])
                elif i % 3 == 1:
                    ax.set_ylim([np.min(l2),np.max(l2)])
                elif i % 3 == 2:
                    ax.set_ylim([np.min(l3),np.max(l3)])
                
            
    #for ax in [axs1[0,1],axs2[0,1]]:
    #    ax1 = ax.twinx()
    #    ax1.set_yticks([])
    #    ax1.set_ylabel(None)
    #    ax1.plot([],[],color='k',linestyle='-',label='NH, Ls = $%i^\circ$' %(ls_n))
    #    ax1.plot([],[],color='k',linestyle='--',label='SH, Ls = $%i^\circ$' %(ls_s))
    #    ax1.legend(loc='upper left')

    if mean is not None:
        mean = ' (%i sol average)' % mean
        sols = '_%i-%isai' % ((tind-m)%30,(tind+m)%30)
    else:
        mean = ''
        sols = '_%isai' % (tind % 30)

        

    #fig.suptitle('Effective diffusivity at ' + \
    #    '%iK' % level,
    #    y=0.95,
    #    fontsize='x-large')

    fig1.savefig(figpath \
                + 'keff_PV_vs_lat_%iK_%03d%s_nh.%s' % (level, tind, sols, ext), dpi=300,
                bbox_inches='tight')
    fig2.savefig(figpath \
                + 'keff_PV_vs_lat_%iK_%03d%s_sh.%s' % (level, tind, sols, ext), dpi=300,
                bbox_inches='tight')

def plot_keff_lat_hov_dust(level = 300, PVmax=True, winds=True, \
                           smooth=None, res = '', ext='png'):
    if res == '':
        exp_names, titles, nrows, ncols = atmospy.get_exps('dust')
    else:
        exp_names, titles, nrows, ncols = atmospy.get_exps('high_res_dust')
    fig, axs = plt.subplots(nrows=ncols,ncols=nrows, figsize = (nrows*10,ncols*2.5),)
    
    lims = [0,4]

    boundaries, cmap, norm = atmospy.new_cmap(lims, extend='max', i = 10, override=True, cols='YlGn')

    for i, ax in enumerate(fig.axes):
        ax.text(0, 1.05, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
            size='large')
        
        ax.set_ylim([-90,90])

    d_keff = xr.open_dataset(
        path + 'mars_analysis/isentropic_vars/%sdust_%iK.nc' % (res,level), decode_times = False,)
    
    d_keff["time"] = d_keff.mars_solar_long[0].values
    d_keff = d_keff.sortby("time", ascending = True)

    lat = d_keff.lat.values

    for i, ax in enumerate(fig.axes):
        exp_name = exp_names[i]
                
        di = d_keff.isel(dust_scale=i)

        #ax.set_xlabel('L$_s$ ($^\circ$)')
        ax.set_ylabel('equivalent latitude')
        ax.set_title('$\lambda = $' + titles[i])
        
        if i == ncols - 1:
            ax.set_xlabel('L$_s$ ($^\circ$)')
        else:
            ax.set_xticklabels([])

        if smooth is not None:
            time  = atmospy.moving_average(di.time, smooth)
            keff   = atmospy.moving_average_2d(di.keff.transpose(), smooth)
            ucomp = atmospy.moving_average_2d( di.ucomp.transpose(), smooth)
            #PV    = atmospy.moving_average_2d( di.PV.transpose(), smooth)
        else:
            time = di.time
            keff  = di.keff.transpose()
            ucomp = di.ucomp.transpose()
            #PV    = di.PV.transpose()
             

        c1=ax.contourf(time, lat, np.log(keff),
                cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])

        if winds:
            c0 = ax.contour(time, lat, ucomp,
                levels=[-50,0,50,100,150], colors='black',linewidths=1)
            c0.levels = [atmospy.nf(val) for val in c0.levels]
            ax.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
        if PVmax:
            PV = di.PV
            l = get_PV_lats(PV.where(PV.lat>20,drop=True).where(
                di.mars_solar_long>180,drop=True), hem='nh')
            if smooth is not None:
                l    = atmospy.moving_average(l, smooth)
                time = atmospy.moving_average(
                    di.time.where(di.mars_solar_long>180,drop=True), smooth)
            else:
                time = di.time.where(di.mars_solar_long>180,drop=True)
            ax.plot(time, l, linestyle='-', color='xkcd:orchid', linewidth=2.5)

            l = get_PV_lats(PV.where(PV.lat<-20,drop=True).where(
                di.mars_solar_long<180,drop=True), hem='sh')
            if smooth is not None:
                l    = atmospy.moving_average(l, smooth)
                time = atmospy.moving_average(
                    di.time.where(di.mars_solar_long<180,drop=True), smooth)
            else:
                time = di.time.where(di.mars_solar_long<180,drop=True)
            ax.plot(time, l, linestyle='-', color='xkcd:orchid', linewidth=2.5)
        #ax.set_xlim([np.min(di.time),np.max(di.time)])
        #xlocs = [k for k in ax.get_xticks()]
        #ls = di.mars_solar_long.interp(time=xlocs,kwargs={"fill_value":"extrapolate"})#, d.time)
    
        #ax.set_xticklabels(['%i' % k for k in ls])
        ax.set_xticks([0,60,120,180,240,300,360])
        ax.set_ylim([di.lat.min().values,di.lat.max().values])
    
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ticks=boundaries,
        ax = axs, pad = 0.05, aspect=50,
        label='normalized effective diffusivity', extend='max')
    
    fig.savefig(figpath+'%sdust_hov_' % res + \
            '%iK.%s' % (level, ext),
            bbox_inches='tight')
    
def plot_keff_lat_hov_latlon(level = 300, PVmax=True, winds=True, \
                           smooth=None, ext='png'):
    exp_names, titles, nrows, ncols = atmospy.get_exps('long-dust')
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize = (ncols*4,nrows*3.5),)
    
    lims = [0,4]

    boundaries, cmap, norm = atmospy.new_cmap(lims, extend='max', i = 10, override=True, cols='YlGn')

    for i, ax in enumerate(fig.axes):
        ax.text(0, 1.05, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
            size='large')
        
        ax.set_ylim([-90,90])

    d_keff = xr.open_dataset(
        path + 'mars_analysis/keffs/latlon_keff_test_tracer_isentropic.nc', decode_times = False,)

    d_keff = d_keff.sel(level=level,method="nearest")


    for i, ax in enumerate(fig.axes):
        exp_name = exp_names[i]
        try:
            d = xr.open_dataset(
                path + exp_name + '/atmos_isentropic.nc', decode_times = False,)
            
            print(exp_name)

            d = d[["PV", "ucomp", "mars_solar_long"]].sel(level=level,method="nearest")
            
        except:
            continue
        
        dis = d_keff.nkeff
        dis["time"] = d.mars_solar_long.values
        d["time"]   = d.mars_solar_long.values
        
        dis = dis.sortby("time", ascending = True)
        d   =   d.sortby("time", ascending = True)

        ax.set_xlabel('L$_s$ ($^\circ$)')

        ax.set_title(titles[i])
        
        if i % 4 == 0:
            ax.set_ylabel('equivalent latitude')
        else:
            ax.set_yticklabels([])

        di = d.mean(dim=["lon"])
        new = dis.new.values
        if smooth is not None:
            
            time  = atmospy.moving_average(   dis.time, smooth)
            dis   = atmospy.moving_average_2d(     dis.transpose(), smooth)
            ucomp = atmospy.moving_average_2d(di.ucomp.transpose(), smooth)
            #PV    = atmospy.moving_average_2d(   di.PV.transpose(), smooth)
        else:
            time = dis.time
            dis = dis.transpose()
            ucomp = di.ucomp.transpose()
            #PV = di.PV.transpose()
             

        c1=ax.contourf(time, new, np.log(dis),
                cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])

        lat  =  di.lat.values
        if winds:
            c0 = ax.contour(time, lat, ucomp,
                levels=[-50,0,50,100,150], colors='black',linewidths=1)
            c0.levels = [atmospy.nf(val) for val in c0.levels]
            ax.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
        if PVmax:
            PV = di.PV
            l = get_PV_lats(PV.where(PV.lat>20,drop=True).where(
                di.mars_solar_long>180,drop=True), hem='nh')
            if smooth is not None:
                l    = atmospy.moving_average(l, smooth)
                time = atmospy.moving_average(
                    di.time.where(di.mars_solar_long>180,drop=True), smooth)
            else:
                time = di.time.where(di.mars_solar_long>180,drop=True)
            ax.plot(time, l, linestyle='-', color='xkcd:orchid', linewidth=2.5)

            l = get_PV_lats(PV.where(PV.lat<-20,drop=True).where(
                di.mars_solar_long<180,drop=True), hem='sh')
            if smooth is not None:
                l    = atmospy.moving_average(l, smooth)
                time = atmospy.moving_average(
                    di.time.where(di.mars_solar_long<180,drop=True), smooth)
            else:
                time = di.time.where(di.mars_solar_long<180,drop=True)
            ax.plot(time, l, linestyle='-', color='xkcd:orchid', linewidth=2.5)
        #ax.set_xlim([np.min(di.time),np.max(di.time)])
        #xlocs = [k for k in ax.get_xticks()]
        #ls = di.mars_solar_long.interp(time=xlocs,kwargs={"fill_value":"extrapolate"})#, d.time)
    
        ax.set_xticks([0,60,120,180,240,300,360])
        ax.set_ylim([di.lat.min().values,di.lat.max().values])
    
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ticks=boundaries[slice(None,None,2)],
        ax = axs, pad = 0.21, orientation='horizontal',
        label='normalized effective diffusivity', extend='max')
    
    fig.savefig(figpath+'latlon_hov_' + \
            '%iK.%s' % (level, ext),
            bbox_inches='tight')


def plot_keff_lat_hov_MY28(level = 300, PVmax=True, winds=True, \
                           smooth=None, ext='png'):
    exp_names, titles, nrows, ncols = atmospy.get_exps('MY28')
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize = (ncols*4,nrows*3.5),)
    
    lims = [0,4]

    boundaries, cmap, norm = atmospy.new_cmap(lims, extend='max', i = 10, override=True, cols='YlGn')

    for i, ax in enumerate(fig.axes):
        ax.text(0, 1.05, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
            size='large')
        
        ax.set_ylim([-90,90])

    d_keff = xr.open_dataset(
        path + 'mars_analysis/keffs/MY28_keff_test_tracer_isentropic.nc', decode_times = False,)

    d_keff = d_keff.sel(level=level,method="nearest")


    for i, ax in enumerate(fig.axes):
        exp_name = exp_names[i]
        try:
            d = xr.open_dataset(
                path + exp_name + '/atmos_isentropic.nc', decode_times = False,)
            
            print(exp_name)

            d = d[["PV", "ucomp", "mars_solar_long"]].sel(level=level,method="nearest")
            
        except:
            continue
        
        dis = d_keff.nkeff
        dis["time"] = d.mars_solar_long.values
        d["time"]   = d.mars_solar_long.values
        
        dis = dis.sortby("time", ascending = True)
        d   =   d.sortby("time", ascending = True)

        ax.set_xlabel('L$_s$ ($^\circ$)')

        ax.set_title(titles[i])
        
        if i % 4 == 0:
            ax.set_ylabel('equivalent latitude')
        else:
            ax.set_yticklabels([])

        di = d.mean(dim=["lon"])
        new = dis.new.values
        if smooth is not None:
            
            time  = atmospy.moving_average(   dis.time, smooth)
            dis   = atmospy.moving_average_2d(     dis.transpose(), smooth)
            ucomp = atmospy.moving_average_2d(di.ucomp.transpose(), smooth)
            #PV    = atmospy.moving_average_2d(   di.PV.transpose(), smooth)
        else:
            time = dis.time
            dis = dis.transpose()
            ucomp = di.ucomp.transpose()
            #PV = di.PV.transpose()
             

        c1=ax.contourf(time, new, np.log(dis),
                cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])

        lat  =  di.lat.values
        if winds:
            c0 = ax.contour(time, lat, ucomp,
                levels=[-50,0,50,100,150], colors='black',linewidths=1)
            c0.levels = [atmospy.nf(val) for val in c0.levels]
            ax.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
        if PVmax:
            PV = di.PV
            l = get_PV_lats(PV.where(PV.lat>20,drop=True).where(
                di.mars_solar_long>180,drop=True), hem='nh')
            if smooth is not None:
                l    = atmospy.moving_average(l, smooth)
                time = atmospy.moving_average(
                    di.time.where(di.mars_solar_long>180,drop=True), smooth)
            else:
                time = di.time.where(di.mars_solar_long>180,drop=True)
            ax.plot(time, l, linestyle='-', color='xkcd:orchid', linewidth=2.5)

            l = get_PV_lats(PV.where(PV.lat<-20,drop=True).where(
                di.mars_solar_long<180,drop=True), hem='sh')
            if smooth is not None:
                l    = atmospy.moving_average(l, smooth)
                time = atmospy.moving_average(
                    di.time.where(di.mars_solar_long<180,drop=True), smooth)
            else:
                time = di.time.where(di.mars_solar_long<180,drop=True)
            ax.plot(time, l, linestyle='-', color='xkcd:orchid', linewidth=2.5)
        #ax.set_xlim([np.min(di.time),np.max(di.time)])
        #xlocs = [k for k in ax.get_xticks()]
        #ls = di.mars_solar_long.interp(time=xlocs,kwargs={"fill_value":"extrapolate"})#, d.time)
    
        ax.set_xticks([0,60,120,180,240,300,360])
        ax.set_ylim([di.lat.min().values,di.lat.max().values])
    
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ticks=boundaries[slice(None,None,2)],
        ax = axs, pad = 0.21, orientation='horizontal',
        label='normalized effective diffusivity', extend='max')
    
    fig.savefig(figpath+'MY28_hov_' + \
            '%iK.%s' % (level, ext),
            bbox_inches='tight')


def plot_keff_lat_hov_parameter(level = 300, PVmax=True, winds=True, smooth=None, half=False,ext='png'):
    eps = [10,15,20,25,30,35,40,45,50]
    if half == True:
        eps = [10,20,30,40,50]
    gamma = [0.093,0]

    fig, axs = plt.subplots(nrows=len(eps),ncols=2, figsize = (10,len(eps)*2),)
        
    lims = [0,4]

    boundaries, cmap, norm = atmospy.new_cmap(lims, extend='max', i = 10, override=True, cols='YlGn')

    for i, ax in enumerate(fig.axes):
        ax.text(-0.05, 1.05, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
            size='large')
        
        ax.set_ylim([-90,90])

    d_keff = xr.open_dataset(
        path + 'mars_analysis/isentropic_vars/parameter_%iK.nc' % level, decode_times = False,)
    lat = d_keff.lat.values
    for j in range(len(eps)):
        for i in range(len(gamma)):
            exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps[j], gamma[i])

            ax = axs[j,i]
            
            
            dis = d_keff.sel(epsilon=eps[j]).sel(gamma=gamma[i])
            if gamma[i]:
                ls = d_keff.mars_solar_long.sel(epsilon=eps[j]).sel(gamma=gamma[i]).values
            dis["time"] = ls
            
            dis = dis.sortby("time", ascending = True)

            if i == 0:
                ax.set_ylabel('equivalent latitude')
                ytext = '$\epsilon = %i^\circ$' % eps[j]
                ax.text(
                    -0.2, 0.5, ytext,
                    ha='right',
                    va='center',
                    transform=ax.transAxes,
                    rotation='vertical',
                    fontsize='large',
                )
            else:
                ax.set_yticklabels([])
            ax.set_xticks([0,60,120,180,240,300,360])
            ax.set_xticklabels([])
            if j == 0:
                ax.set_title('$\gamma = %.3f$' % gamma[i])
            elif eps[j] == eps[-1]:
                ax.set_xticklabels([0,60,120,180,240,300,360])
                ax.set_xlabel('L$_s$ ($^\circ$)')

            if smooth is not None:

                time  = atmospy.moving_average(   dis.time, smooth)
                keff   = atmospy.moving_average_2d(dis.keff.transpose(), smooth)
                ucomp = atmospy.moving_average_2d(dis.ucomp.transpose(), smooth)
                #PV   = atmospy.moving_average_2d(   di.PV.transpose(), smooth)
            else:
                time = dis.time
                keff = dis.keff.transpose()
                ucomp = dis.ucomp.transpose()
                #PV = di.PV.transpose()


            c1=ax.contourf(time, lat, np.log(keff),
                    cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])

            if winds:
                c0 = ax.contour(time, lat, ucomp,
                    levels=[-50,0,50,100,150], colors='black',linewidths=1)
                c0.levels = [atmospy.nf(val) for val in c0.levels]
                ax.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
            if PVmax:
                PV = dis.PV
                if gamma[i] != 0.:
                    cond1 = (dis.mars_solar_long>220)# or (di.mars_solar_long<50)
                l = get_PV_lats(PV.where(PV.lat>20,drop=True).where(
                    cond1,drop=True), hem='nh')
                if smooth is not None:
                    l  = atmospy.moving_average(l, smooth)
                    time = atmospy.moving_average(
                        dis.time.where(cond1,drop=True), smooth)
                else:
                    time = dis.time.where(cond1,drop=True)
                ax.plot(time, l, linestyle='-', color='xkcd:orchid', linewidth=2.5)

                if gamma[i] != 0.:
                    cond2 = (dis.mars_solar_long<180)
                l = get_PV_lats(PV.where(PV.lat<-20,drop=True).where(
                    cond2,drop=True), hem='sh')
                if smooth is not None:
                    l  = atmospy.moving_average(l, smooth)
                    time = atmospy.moving_average(
                        dis.time.where(cond2,drop=True), smooth)
                else:
                    time = dis.time.where(cond2,drop=True)
                ax.plot(time, l, linestyle='-', color='xkcd:orchid', linewidth=2.5)
            ax.set_xticks([0,60,120,180,240,300,360])
            ax.set_ylim([dis.lat.min().values,dis.lat.max().values])
    
    plt.tight_layout()
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ticks=boundaries,
        ax = axs,pad = 0.025,aspect=50,
        label='normalized effective diffusivity', extend='max')
    if half:
        half=''
    else:
        half='_all'
    fig.savefig(figpath+'parameter_hov_' + \
            '%iK%s.%s' % (level,half,ext),
            bbox_inches='tight')


if __name__ == "__main__":
    eps = np.arange(10,55,5)
    gamma = [0.093,0.00]
    
    #plot_keff_lat(exps = ['curr-ecc','0-ecc','dust','high_res_dust','MY28'],
    #              level=300,mean=10,tind=110)
    #plot_keff_lat_PV(mean=10,tind=110,half=False,level=300, ext='pdf')
    #plot_keff_lat_hov_latlon()
    #plot_keff_lat_hov_dust(level=300, smooth=30,ext='pdf',res='')
    #plot_keff_lat_hov_dust(level=300, smooth=30,ext='pdf',res='vert_')
    #plot_keff_lat_hov_MY28(level=300, smooth=30,ext='pdf')
    plot_keff_lat_hov_parameter(level=300, smooth=30, half=True,ext='pdf')
# %%
