# %%
import xarray as xr
import numpy as np
import sys, os
import math

sys.path.append('../')

from atmospy import get_exps, stereo_plot, lait, xr_add_cyclic_point, \
        calc_jet_lat, moving_average, new_cmap, make_stereo_plot, nf
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
            nrow = 3
            ncol = 2
        
        if hem == 'NH':
            proj = ccrs.NorthPolarStereo()
        elif hem == 'SH':
            proj = ccrs.SouthPolarStereo()

        fig, axs = plt.subplots(nrows=nrow,ncols=ncol, figsize = (4*ncol,2.5*nrow),
                            subplot_kw = {'projection': proj})
        l = 0
        for j in range(len(exp_names)):
            try:
                exp_name = exp_names[j]
            except:
                continue
            print(exp_name)
            
            if savedata:
                if os.path.isfile(path+'mars_analysis/PV_maps/%s_%s.nc' % (exp_name,hem)):
                    continue

                try:
                    ds = xr.open_dataset(path+exp_name+'/atmos_isentropic.nc', decode_times=False)
                except:
                    continue
                ds = ds[["PV","mars_solar_long","ucomp"]].sel(level=level,method="nearest")
                
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
                                        
                elif hem == 'SH':
                    dsn = ds.where(ds.lat<-50,drop=True).isel(time=tslice)
                                                  
                PV = dsn.PV
                u = dsn.ucomp
                PV = xr_add_cyclic_point(PV).mean(dim="time")
                u = xr_add_cyclic_point(u).mean(dim="time")

                dsn = xr.Dataset(data_vars=dict(
                        PV    = (["lat","lon"], PV.data),
                        ucomp = (["lat","lon"],  u.data),
                        ),
                        coords = dict(
                        lat = PV.lat.data,
                        lon = PV.lon.data,
                    ),
                    )
                dsn.to_netcdf(path+'mars_analysis/PV_maps/%s_%s.nc' % (exp_name,hem))

        if exp == '$\gamma = 0.093$' or exp == 'Dust Scale':

            d = xr.open_dataset(path+'mars_analysis/PV_maps/tracer_soc_mars_mola_topo_lh_eps_' + \
                            '25_gamma_0.093_cdod_clim_scenario_7.4e-05_%s.nc' % hem,
                            decode_times=False)
        elif exp == '$\gamma = 0.000$':
            d = xr.open_dataset(path+'mars_analysis/PV_maps/tracer_soc_mars_mola_topo_lh_eps_' + \
                            '25_gamma_0.000_cdod_clim_scenario_7.4e-05_%s.nc' % hem,
                            decode_times=False)
        elif exp == 'Dust Vertical Res':
            d = xr.open_dataset(path+'mars_analysis/PV_maps/tracer_vert_soc_mars_mola_topo_lh_eps_' + \
                            '25_gamma_0.093_cdod_clim_scenario_7.4e-05_%s.nc' % hem,
                            decode_times=False)
        elif exp == 'Longitudinal Dust':
            d = xr.open_dataset(path+'mars_analysis/PV_maps/tracer_soc_mars_mola_topo_lh_eps_' + \
                            '25_gamma_0.093_clim_latlon_7.4e-05_%s.nc' % hem,
                            decode_times=False)
        for j, ax in enumerate(fig.axes):
            ax.set_frame_on(False)
            try:
                exp_name = exp_names[j]
            except:
                continue
            dsn = xr.open_dataset(path+'mars_analysis/PV_maps/%s_%s.nc' % (exp_name,hem),
                                     decode_times=False)
                
            
            PV = dsn.PV - d.PV
            u  = dsn.ucomp  - d.ucomp

            lims = [-30,30]
            levels = [-20,-10,0,10,20]
            if exp == 'long-dust' and hem == 'NH':
                lims = [-8, 8]
            boundaries, cmap, norm = new_cmap(lims, extend='both', i=20,cols='RdBu_r')

            if exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05' \
                or exp_name == 'tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.000_cdod_clim_scenario_7.4e-05' \
                    or exp_name == 'tracer_vert_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05':
                PV = dsn.PV
                u  = dsn.ucomp

                lims = [0,45]
                levels = [0,50,100,120]
                if exp == 'long-dust' and hem == 'NH':
                    lims = [10, 50]
                boundaries, cmap, norm = new_cmap(lims, extend='both',override=True, i=10,cols='viridis')
                cb = fig.colorbar(
                    cm.ScalarMappable(norm=norm, cmap=cmap),extend = 'both',
                    ticks=boundaries[slice(None,None,2)],ax = ax,aspect=20,#location='left',
                    pad=0.15)
                cb.set_label(label='PV (MPVU)',fontsize='large')

            if hem == 'NH':
                make_stereo_plot(ax, [np.max(dsn.lat), 80, 70, 60, np.min(dsn.lat)],
                          [-180, -120, -60, 0, 60, 120, 180],
                          circle, alpha = 0.3, linestyle = '--',)

            else:
                make_stereo_plot(ax, [np.max(dsn.lat),-60,-70,-80, np.min(dsn.lat)],
                          [-180, -120, -60, 0, 60, 120, 180],
                          circle, alpha = 0.3, linestyle = '--',)
                PV = -PV
            
            ax.contourf(dsn.lon,dsn.lat,PV*10**4,transform=ccrs.PlateCarree(),cmap=cmap,
                            norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])
            c0 = ax.contour(dsn.lon,dsn.lat,u,transform=ccrs.PlateCarree(),colors='k',
                            levels=levels,linewidths=0.8)
            c0.levels = [nf(val) for val in c0.levels]
            ax.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
            if exp == '$\gamma = 0.093$' or exp == '$\gamma = 0.000$':
                ax.set_title(titles[j]+'$^\circ$')
            elif exp == 'MY28':
                ax.set_title('')
            else:
                ax.set_title(titles[j])

            if PVmax:
                q_max = []
                if hem == 'NH':
                    a = dsn.PV.load()
                else:
                    a = -dsn.PV.load()
                for l in range(len(a.lon)):
                    q = a.isel(lon = l)
                    q0, _ = calc_jet_lat(q, a.lat)
                    q_max.append(q0)

                c0 = ax.plot(a.lon, q_max,transform=ccrs.PlateCarree(),
                     color='xkcd:orchid', linewidth=2)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.35,wspace=0.05)
        cb = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),extend = 'both',
            ticks=boundaries[slice(None,None,2)],ax = axs,pad=0.06,aspect=50,location='left')
        cb.set_label(label='PV (MPVU)',fontsize='large')
        
        fig.suptitle('%s %s' % (exp, hem),y=1.05)
        fig.savefig(figpath \
                + 'PV_map_anomaly_%s_%s.%s' % (exps[i],hem,ext), dpi=300,
                bbox_inches='tight')

#%%

if __name__ == "__main__":
    eps = np.arange(10,55,5)
    gamma = [0.093,0.00]
    savedata = True
    PVmax = True
    plot_PV_maps(exps=['dust'],average=30,level=300,
                 PVmax=PVmax,ext='pdf',hem='SH',savedata=savedata)
# %%
