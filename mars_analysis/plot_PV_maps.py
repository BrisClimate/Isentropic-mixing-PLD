# %%
import xarray as xr
import numpy as np
import sys, os
import math

sys.path.append('../')

from atmospy import get_exps, stereo_plot, lait, xr_add_cyclic_point, \
        calc_jet_lat, moving_average, new_cmap, make_stereo_plot
import string

from cartopy import crs as ccrs
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import (cm, colors, cycler)
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

figpath = '/user/home/xz19136/Figures/mars_analysis/PV/maps/'
path = '/user/work/xz19136/Isca_data/'
theta, center, radius, verts, circle = stereo_plot()
theta0 = 200.
kappa = 1/4.0

if plt.rcParams["text.usetex"]:
    fmt = r'%r \%'
else:
    fmt = '%r'

def get_PV_lats_isentropic(di, hem='NH'):
    '''
    Lait-scale PV and then return the latitude of maximum PV on given 
    pressure levels'''
    laitPV = lait(di.PV,di.level,theta0,kappa=kappa)

    l = []
    s = []
    for a in range(len(di.time)):
        try:
            x = laitPV.isel(time=a)
        
            x = x.where(x != np.nan, drop = True)
            if hem == 'NH':
                x = x.sortby('lat',ascending=True)
                phi_PV, PV_max = calc_jet_lat(x, x.lat)
            else:
                x = x.sortby('lat',ascending=False)
                phi_PV, PV_max = calc_jet_lat(-x, x.lat)                    
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

def get_PV_max_isentropic(di, hem='NH'):
    '''
    Lait-scale PV and then return the latitude of maximum PV on given 
    pressure levels'''
    laitPV = lait(di.PV,di.level,theta0,kappa=kappa)

    l = []
    s = []
    for a in range(len(di.time)):
        try:
            x = laitPV.isel(time=a)
        
            x = x.where(x != np.nan, drop = True)
            if hem == 'NH':
                phi_PV, PV_max = calc_jet_lat(x, x.lat)
            else:
                phi_PV, PV_max = calc_jet_lat(-x, x.lat)                    
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
    smooth=None,level=300,ext='png',savedata=False):
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

            if savedata:
            
                ds = xr.open_dataset(path+exp_name+'/atmos_isentropic.nc', decode_times=False)

                ds = ds[["PV","mars_solar_long"]].sel(level=level,method="nearest")
                ds = ds.mean(dim="lon")

                _, PV_max_n = get_PV_lats_isentropic(ds.where(ds.lat > 0, drop=True),hem='NH')
                _, PV_max_s = get_PV_lats_isentropic(ds.where(ds.lat < 0, drop=True),hem='SH')

                if smooth is not None:
                    time     = moving_average(ds.time, smooth)
                    PV_max_n = moving_average(PV_max_n, smooth)
                    PV_max_s = moving_average(PV_max_s, smooth)
                    ls       = moving_average(ds.mars_solar_long, smooth)
                else:
                    time = ds.time
                    ls = ds.mars_solar_long

                PV_max_n =  PV_max_n*10**4
                PV_max_s = -PV_max_s*10**4

                ds = xr.Dataset(data_vars=dict(
                    PV_n = (["time"], PV_max_n),
                    PV_s = (["time"], PV_max_s),
                    ls   = (["time"], ls),
                ),
                coords = dict(
                    time = time,
                ),
                )
                ds.to_netcdf(path+'mars_analysis/PV_strength/%s.nc' % exp_name)

            else:
                ds = xr.open_dataset(path+'mars_analysis/PV_strength/%s.nc' % exp_name,
                                     decode_times=False)
                PV_max_n = ds.PV_n
                PV_max_s = ds.PV_s
                ls = ds.ls
                time = ds.time


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



    fig.savefig(figpath \
                + 'PV_maxstrength.%s' %ext, dpi=300,
                bbox_inches='tight')


def plot_PV_maps(exps=['curr-ecc','0-ecc','dust'], hem = 'NH',\
    average=30,level=300,ext='png',savedata=False,PVmax=False):
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
            nrow = 5
            ncol = 1
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
            nrow = 3
            ncol = 2
        
        if hem == 'NH':
            proj = ccrs.NorthPolarStereo()
        elif hem == 'SH':
            proj = ccrs.SouthPolarStereo()

        fig, axs = plt.subplots(nrows=nrow,ncols=ncol, figsize = (4*ncol,2.5*nrow),
                            subplot_kw = {'projection': proj})
        l = 0
        for j, ax in enumerate(fig.axes):
            ax.set_frame_on(False)
            try:
                exp_name = exp_names[j]
            except:
                continue
            print(exp_name)
            
            if savedata:
                try:
                    ds = xr.open_dataset(path+exp_name+'/atmos_isentropic.nc', decode_times=False)
                except:
                    continue
                ds = ds[["PV","mars_solar_long"]].sel(level=level,method="nearest")
                
                if exp != '$\gamma = 0.000$':
                    if hem == 'NH':
                        tind = (np.abs(ds.mars_solar_long.values-270)).argmin(axis=0)
                        #i = np.take_along_axis(ds, np.abs(ds.mars_solar_long.values-270).argmin(axis=0), axis=0)
                    elif hem == 'SH':
                        tind = np.abs(ds.mars_solar_long.values-90).argmin(axis=0)
                tslice = slice(tind-average,tind+average)
                if hem == 'NH':
                    dsn = ds.where(ds.lat>50,drop=True).isel(
                        time=tslice)
                                        
                    dsn = dsn.PV
                    dsn = xr_add_cyclic_point(dsn).mean(dim="time")
                    dsn.to_netcdf(path+'mars_analysis/PV_maps/%s_NH.nc' % exp_name)
                elif hem == 'SH':
                    dsn = ds.where(ds.lat<-50,drop=True).isel(time=tslice)
                                        
                    dsn = dsn.PV
                    dsn = xr_add_cyclic_point(dsn).mean(dim="time")            
                    dsn.to_netcdf(path+'mars_analysis/PV_maps/%s_SH.nc' % exp_name)

            else:
                dsn = xr.open_dataset(path+'mars_analysis/PV_maps/%s_%s.nc' % (exp_name,hem),
                                     decode_times=False)
                dsn = dsn.to_array().squeeze()
                

            if hem == 'NH':
                make_stereo_plot(ax, [np.max(dsn.lat), 80, 70, 60, np.min(dsn.lat)],
                          [-180, -120, -60, 0, 60, 120, 180],
                          circle, alpha = 0.3, linestyle = '--',)
                dsn = dsn
            else:
                make_stereo_plot(ax, [np.max(dsn.lat),-60,-70,-80, np.min(dsn.lat)],
                          [-180, -120, -60, 0, 60, 120, 180],
                          circle, alpha = 0.3, linestyle = '--',)
                dsn = -dsn
            
            lims = [10,40]
            if exp == 'long-dust' and hem == 'NH':
                lims = [10, 50]
            boundaries, cmap, norm = new_cmap(lims, extend='both',override=True, i=10,cols='viridis')
            ax.contourf(dsn.lon,dsn.lat,dsn*10**4,transform=ccrs.PlateCarree(),cmap=cmap,
                            norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])
            if exp == '$\gamma = 0.093$' or exp == '$\gamma = 0.000$':
                ax.set_title(titles[j]+'$^\circ$')
            elif exp == 'MY28':
                ax.set_title('')
            else:
                ax.set_title(titles[j])

            if PVmax:
                q_max = []
                a = dsn.load()
                for l in range(len(a.lon)):
                    q = a.isel(lon = l)
                    q0, _ = calc_jet_lat(q, a.lat)
                    q_max.append(q0)

                c0 = ax.plot(a.lon, q_max,transform=ccrs.PlateCarree(),
                     color='blue', linewidth=1)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4,wspace=0.05)
        cb = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),extend = 'both',
            ticks=boundaries[slice(None,None,2)],ax = axs,pad=0.11,aspect=50)
        cb.set_label(label='PV (MPVU)',fontsize='large')
        
        fig.suptitle('%s %s' % (exp, hem),y=1.05)
        fig.savefig(figpath \
                + 'PV_map_%s_%s.%s' % (exps[i],hem,ext), dpi=300,
                bbox_inches='tight')

#%%

if __name__ == "__main__":
    eps = np.arange(10,55,5)
    gamma = [0.093,0.00]
    savedata = True
    PVmax = True
    plot_PV_maps(exps=['high_res_dust'],average=30,level=300,
                 PVmax=PVmax,ext='pdf',hem='SH',savedata=savedata)
# %%
