# %%
import xarray as xr
import numpy as np
import sys, os

sys.path.append('/user/home/xz19136/Py_Scripts/atmospy/')

import analysis_functions as funcs

import string

from test_tracer_plot import (open_files, plot_map_contourf)

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import (cm, colors, cycler)
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

if plt.rcParams["text.usetex"]:
    fmt = r'%r \%'
else:
    fmt = '%.2f'


path = '/user/work/xz19136/Isca_data/'
theta, center, radius, verts, circle = funcs.stereo_plot()
theta0 = 200.
kappa = 1/4.0

def get_contours(lat,lon,z, **kwargs):
    '''
    Inputs
    ------
    lat
    lon
    z
    proj : North or South polar stereographic plot
    close=True

    Outputs
    -------
    Returns all contours from plot
    '''
    proj   = kwargs.pop(  'proj', 'PC')
    levels = kwargs.pop('levels', np.linspace(np.min(z),np.max(z),50))
    
    close  = kwargs.pop( 'close', True)
    if proj == 'PC':
        projection = ccrs.PlateCarree()
    elif proj == 'NP':
        projection = ccrs.NorthPolarStereo()
    elif proj == 'SP':
        projection = ccrs.SouthPolarStereo()
    fig, axs = plt.subplots(nrows=1,ncols=1,
                            subplot_kw = {'projection': projection})
    #not fig.get_visible())

    #_, _, _, _, circle = funcs.stereo_plot()
    #if proj == ccrs.NorthPolarStereo():
    #    funcs.make_stereo_plot(axs, [90, 75, 50, 25,0],
    #                      [-180, -120, -60, 0, 60, 120, 180],
    #                      circle, alpha = 0.3, linestyle = '--',)
    #else:
    #    funcs.make_stereo_plot(axs, [0, -25, -50, -75,-90],
    #                      [-180, -120, -60, 0, 60, 120, 180],
    #                      circle, alpha = 0.3, linestyle = '--',)
    #    z = -z
    #levels = kwargs.pop('levels',np.linspace(np.min(z),np.max(z),50))
    if proj == 'NP':
        funcs.make_stereo_plot(axs, [np.max(lat), 80, 70, 60,],
                          [-180, -120, -60, 0, 60, 120, 180],
                          circle, alpha = 0.3, linestyle = '--',)

    CS = axs.contour(lon,lat,z,transform=ccrs.PlateCarree(),
                    levels=levels)
    x = []
    for i in range(len(CS.allsegs[1:-1])):
        
        y = CS.allsegs[i+1]
        #if len(y) < 3:
        x.append(y)
    if close:
        fig.set_visible(False)
        plt.close()
    else:
        plt.show()
    return x, fig, axs

def plot_contour_keff_map(exp_name, isentropic=False, mean=None, tind=101, n=29, \
                        level=0.2, proj='NP', PVmax=False):
    '''
    Plot contours of tracer and effective diffusivity at that equivalent
    latitude, at a given time on a given level'''

    ds, d = open_files(exp_name, isentropic)
    if mean is None:
        tind, n = funcs.get_nth_sol(tind, n)
        tind = tind+n
        d = d.isel(time=tind)
        ds = ds.isel(time=tind)
        ls = d.mars_solar_long
    else:
        tind, n = funcs.get_timeslice(tind, mean)
        tslice = slice(tind-n, tind+n)
        ls = d.mars_solar_long.isel(time=tind)
        d = d.isel(time=tslice)
        ds = ds.isel(time=tslice)
        
    
    if not isentropic:
        d = d.sel(pfull = level, method="nearest")
        if PVmax:
            PV = funcs.lait(d.PV,d.theta,theta0,kappa=kappa)
    else:
        d = d.sel(level = level, method="nearest")
        if PVmax:
            PV = funcs.lait(d.PV,d.level,theta0,kappa=kappa)

    d = d.test_tracer+2

    ds = ds.sel(level = level, method="nearest")
    
    if isentropic:
        level = d.level.values
    ds = xr.ufuncs.log(ds.nkeff)

    if mean is not None:
        d  =  d.mean(dim="time")
        ds = ds.mean(dim="time")
        if PVmax:
            PV = PV.mean(dim="time")

    ds = ds.expand_dims({'lon':d.lon})
    

    #_, fig, axs = get_contours(d.lat, d.lon, d, proj=proj,close=True, levels = np.linspace(-1,2,20))
    
    if proj == 'PC':
        projection = ccrs.PlateCarree()
    elif proj == 'NP':
        projection = ccrs.NorthPolarStereo()
        d  =  d.where( d.lat > 60, drop = True)
        if PVmax:
            PV = PV.where(PV.lat > 60, drop = True)
        ds = ds.where(ds.new > 60, drop = True)
        hem = 'NH'
    elif proj == 'SP':
        projection = ccrs.SouthPolarStereo()
        d  =  d.where( d.lat <-60, drop = True)
        if PVmax:
            PV = PV.where(PV.lat <-60, drop = True)
        ds = ds.where(ds.new <-60, drop = True)
        hem = 'SH'
    
    levels = np.linspace(np.min(d), np.max(d), 10)

    
    fig, axs = plt.subplots(nrows=1,ncols=1,
                            subplot_kw = {'projection': projection})
    if proj == 'NP':
        funcs.make_stereo_plot(axs, [np.max(d.lat), 80, 70, np.min(d.lat)],# 60,],
                          [-180, -120, -60, 0, 60, 120, 180],
                          circle, alpha = 0.3, linestyle = '--',)
    elif proj == 'SP':
        funcs.make_stereo_plot(axs, [np.max(d.lat), -70, -80, np.min(d.lat)],
                          [-180, -120, -60, 0, 60, 120, 180],
                          circle, alpha = 0.3, linestyle = '--',)
    
    CS = axs.contour(d.lon,d.lat,d,transform=ccrs.PlateCarree(),
                    levels = 10, colors='k',)
                    #levels=levels, colors='k')
    CS.levels = [funcs.nf(val) for val in CS.levels]
    axs.clabel(CS, CS.levels, inline=1, fmt = fmt, fontsize ='small')

    ds = ds.interp({'new':d.lat})
    boundaries, cmap, norm = funcs.new_cmap([0,4], extend='max',override=True, i =10)
    axs.contourf(ds.lon,ds.new,ds.transpose(),transform=ccrs.PlateCarree(),cmap=cmap,
                    norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])

    
    if PVmax:
        q_max = []
        a = PV.load()
        for l in range(len(a.lon)):
            q = a.isel(lon = l)
            if hem == 'NH':
                try:
                    q0, _ = funcs.calc_jet_lat(q, a.lat)
                except:
                    q0 = 90
                if q0 == 90 or q0 == -90 or q0 == 0:
                    q0 = np.max(np.abs(a.lat))
            if hem == 'SH':
                try:
                    q0, _ = funcs.calc_jet_lat(-q, a.lat)
                except:
                    q0 = -90
                if q0 == 90 or q0 == -90 or q0 == 0:
                    q0 = np.max(np.abs(a.lat))
                    q0 = -q0
            q_max.append(q0)

        c0 = axs.plot(a.lon, q_max,transform=ccrs.PlateCarree(),
                 color='blue', linewidth=1)
    
    cb = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),extend = 'max',
            ticks=boundaries[slice(None,None,2)],
            ax = axs)
    cb.set_label(label='$k_{eff}$',fontsize='large')

    fig.suptitle('%s,\nLs %i$^\circ$, %s' % (exp_name, ls, hem))

    if mean is not None:
        n = '%i-%i' % ((tind-n)%30,(tind+n)%30)

        fig.savefig('/user/home/xz19136/Figures/mars_analysis/maps/contours/' + \
            '%s_%s_%.1f_%i_%ssol.png' % (exp_name, hem, level, tind, n),
            bbox_inches='tight',dpi=300)
    else:
        fig.savefig('/user/home/xz19136/Figures/mars_analysis/maps/contours/' + \
            '%s_%s_%.1f_%03d.png' % (exp_name, hem, level, tind),
            bbox_inches='tight',dpi=300)

def plot_contour_keff_all_maps(exps, isentropic=False, mean=None, tind=101, n=29, \
                        level=0.2, proj='NP', PVmax=False):

    if proj == 'PC':
        projection = ccrs.PlateCarree()
    elif proj == 'NP':
        projection = ccrs.NorthPolarStereo()
        #d  =  d.where( d.lat > 60, drop = True)
        #ds = ds.where(ds.new > 60, drop = True)
        hem = 'NH'
    elif proj == 'SP':
        projection = ccrs.SouthPolarStereo()
        #d  =  d.where( d.lat <-60, drop = True)
        #ds = ds.where(ds.new <-60, drop = True)
        hem = 'SH'
    
    #levels = np.linspace(np.min(d), np.max(d), 10)

    
    exp_names = []
    if exps == 'parameter':
        eps = np.arange(10,55,5)
        gamma = [0.093,0.00]
        titles = []
        for i in gamma:
            for j in eps:
                titles.append('$\epsilon = %i$,\n$\gamma = %.3f$' %(j,i))
                exp_names.append('tracer_soc_mars_mola_topo_lh_eps_' + \
                    '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' %(j,i))
        nrows = len(gamma)
        ncols = len(eps)
        

    elif exps == 'attribution':
        titles = ['Control', 'LH', 'D', 'LH+D', 'T', 'LH+T', 'D+T', 'LH+D+T']
        for t in ['', '_mola_topo']:
            for d in ['', '_cdod_clim_scenario_7.4e-05']:
                for l in ['', '_lh']:
                    exp_names.append('tracer_soc_mars%s%s_eps_25_gamma_0.093%s' % (t, l, d))
        nrows = 2
        ncols = 4

    elif exps == 'dust':
        titles = ['1/2', '1', '2', '4']
        for ds in [3.7e-5, 7.4e-5,1.48e-4,2.96e-4]:
            exp_names.append('tracer_soc_mars_mola_topo_lh_eps_' + \
                    '25_gamma_0.093_cdod_clim_scenario_%s' % str(ds))
        nrows = 1
        ncols = 4

    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(3*ncols,3*nrows),
                            subplot_kw = {'projection': projection})

    boundaries, cmap, norm = funcs.new_cmap([0,6], extend='max',override=True, i =10)

    if mean is None:
        tind, n = funcs.get_nth_sol(tind, n)
        tind = tind+n
    else:
        tind, n = funcs.get_timeslice(tind, mean)
        tslice = slice(tind-n, tind+n)

    for i, ax in enumerate(fig.axes):
        if proj == 'NP':
            funcs.make_stereo_plot(ax, [87, 80, 70, 60,],
                          [-180, -120, -60, 0, 60, 120, 180],
                          circle, alpha = 0.3, linestyle = '--',)
        elif proj == 'SP':
            funcs.make_stereo_plot(ax, [-60, -70, -80, -87],
                          [-180, -120, -60, 0, 60, 120, 180],
                          circle, alpha = 0.3, linestyle = '--',)

        ax.text(0.05, 0.95, string.ascii_lowercase[i], transform=ax.transAxes, 
            size='large')
        ax.set_title(titles[i])

        try:
            ds, d = open_files(exp_names[i], isentropic)
        except:
            continue
        if mean is None:
            d = d.isel(time=tind)
            ds = ds.isel(time=tind)
            if (i == 0 and exps == 'parameter') or exps == 'attribution' or exps == 'dust':
                ls = d.mars_solar_long
        else:
            if (i == 0 and exps == 'parameter') or exps == 'attribution' or exps == 'dust':
                ls = d.mars_solar_long.isel(time=tind)
            d = d.isel(time=tslice)
            ds = ds.isel(time=tslice)



        d = d.sel(pfull = level, method="nearest")
        if PVmax:
            PV = d.PV

        d = d.test_tracer+2

        ds = ds.sel(level = level, method="nearest")


        ds = xr.ufuncs.log(ds.nkeff)

        if mean is not None:
            d  =  d.mean(dim="time")
            if PVmax:
                PV = PV.mean(dim="time")
            ds = ds.mean(dim="time")

        ds = ds.expand_dims({'lon':d.lon})


        #_, fig, axs = get_contours(d.lat, d.lon, d, proj=proj,close=True, levels = np.linspace(-1,2,20))


        if proj == 'NP':
            d  =  d.where( d.lat > 60, drop = True)
            ds = ds.where(ds.new > 60, drop = True)
            if PVmax:
                PV = PV.where(PV.lat > 60, drop = True)
        elif proj == 'SP':
            d  =  d.where( d.lat <-60, drop = True)
            ds = ds.where(ds.new <-60, drop = True)
            if PVmax:
                PV = PV.where(PV.lat <-60, drop = True)

        levels = np.linspace(np.min(d), np.max(d), 10)   

        CS = ax.contour(d.lon,d.lat,d,transform=ccrs.PlateCarree(),
                        levels = 5, colors='k',)
                        #levels=levels, colors='k')
        CS.levels = [funcs.nf(val) for val in CS.levels]
        ax.clabel(CS, CS.levels, inline=1, fmt = fmt, fontsize ='small')

        ds = ds.interp({'new':d.lat})


        ax.contourf(ds.lon,ds.new,ds.transpose(),transform=ccrs.PlateCarree(),cmap=cmap,
                        norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])

        if PVmax:
            q_max = []
            a = PV.load()
            for l in range(len(a.lon)):
                q = a.isel(lon = l)
                q0, _ = funcs.calc_jet_lat(q, a.lat)
                q_max.append(q0)

            c0 = ax.plot(a.lon, q_max,transform=ccrs.PlateCarree(),
                     color='blue', linewidth=1)

    cb = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),extend = 'max',
            ticks=boundaries[slice(None,None,2)],ax = axs,)
    cb.set_label(label='$k_{eff}$',fontsize='large')

    fig.suptitle('Ls %i$^\circ$, %s' % (ls, hem))

    if mean is not None:
        n = '%i-%i' % ((tind-n)%30,(tind+n)%30)

        fig.savefig('/user/home/xz19136/Figures/mars_analysis/maps/contours/' + \
            '%s_%s_%.1f_%i_%ssol.png' % (exps, hem, level, tind, n),
            bbox_inches='tight')
    else:
        fig.savefig('/user/home/xz19136/Figures/mars_analysis/maps/contours/' + \
            '%s_%s_%.1f_%i_%isol.png' % (exps, hem, level, tind, n),
            bbox_inches='tight')


def plot_PV_map(exp_name, isentropic=True, mean=None, tind=101, n=29, \
                        level=300, proj='NP'):
    '''
    Plot contours of tracer and effective diffusivity at that equivalent
    latitude, at a given time on a given level'''
    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%i'
    _, d = open_files(exp_name, isentropic)
    if mean is None:
        tind, n = funcs.get_nth_sol(tind, n)
        tind = tind+n
        d = d.isel(time=tind)
        ls = d.mars_solar_long
    else:
        #tind, n = funcs.get_timeslice(tind, mean)

        tslice = slice(tind-int(mean/2), tind+int(mean/2))
        ls = d.mars_solar_long.isel(time=tind)
        d = d.isel(time=tslice)
        
    
    if not isentropic:
        d = d.sel(pfull = level, method="nearest")
        PV = funcs.lait(d.PV,d.theta,theta0,kappa=kappa)*10**4
    else:
        d = d.sel(level = level, method="nearest")
        PV = funcs.lait(d.PV,d.level,theta0,kappa=kappa)*10**4

        level = d.level.values

    if mean is not None:
        d  =  d.mean(dim="time")
        PV = PV.mean(dim="time")   

    #_, fig, axs = get_contours(d.lat, d.lon, d, proj=proj,close=True, levels = np.linspace(-1,2,20))
    
    if proj == 'PC':
        projection = ccrs.PlateCarree()
    elif proj == 'NP':
        projection = ccrs.NorthPolarStereo()
        d  =  d.where( d.lat > 60, drop = True)
        PV = PV.where(PV.lat > 60, drop = True)
        hem = 'NH'
    elif proj == 'SP':
        projection = ccrs.SouthPolarStereo()
        d  =  d.where( d.lat <-60, drop = True)
        PV = PV.where(PV.lat <-60, drop = True)
        hem = 'SH'
    
    #levels = np.linspace(np.min(d), np.max(d), 10)

    
    fig, axs = plt.subplots(nrows=1,ncols=1,
                            subplot_kw = {'projection': projection})
    if proj == 'NP':
        funcs.make_stereo_plot(axs, [np.max(d.lat), 80, 70, np.min(d.lat)],# 60,],
                          [-180, -120, -60, 0, 60, 120, 180],
                          circle, alpha = 0.3, linestyle = '--',)
    elif proj == 'SP':
        funcs.make_stereo_plot(axs, [np.max(d.lat), -70, -80, np.min(d.lat)],
                          [-180, -120, -60, 0, 60, 120, 180],
                          circle, alpha = 0.3, linestyle = '--',)
        PV = -PV
    
    CS = axs.contour(d.lon,d.lat,d.ucomp,transform=ccrs.PlateCarree(),
                    levels = [0,50,100], colors='k',)
                    #levels=levels, colors='k')
    CS.levels = [funcs.nf(val) for val in CS.levels]
    axs.clabel(CS, CS.levels, inline=1, fmt = fmt, fontsize ='small')

    boundaries, cmap, norm = funcs.new_cmap([0,7], extend='max',override=True, i =10)
    axs.contourf(d.lon,d.lat,PV,transform=ccrs.PlateCarree(),cmap=cmap,
                    norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])

    
    cb = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),extend = 'max',
            ticks=boundaries[slice(None,None,2)],
            ax = axs)
    cb.set_label(label='PV (MPVU)',fontsize='large')

    fig.suptitle('Potential Vorticity,\nLs %i$^\circ$, %s' % (ls, hem))

    if mean is not None:
        n = '%i-%i' % ((tind-n)%30,(tind+n)%30)

        fig.savefig('/user/home/xz19136/Figures/mars_analysis/maps/PV/' + \
            '%s_%s_%.1f_%i_%ssol.png' % (exp_name, hem, level, tind, n),
            bbox_inches='tight',dpi=300)
    else:
        fig.savefig('/user/home/xz19136/Figures/mars_analysis/maps/PV/' + \
            '%s_%s_%.1f_%03d.png' % (exp_name, hem, level, tind),
            bbox_inches='tight',dpi=300)


def plot_two_diagram(isentropic=True, \
                        level=300, proj='NP'):
    '''
    Plot contours of tracer and effective diffusivity at that equivalent
    latitude, at a given time on a given level'''

    _, d = open_files('tracer_soc_mars_mola_topo_lh_eps_' + \
                    '25_gamma_0.093_cdod_clim_scenario_3.7e-05', isentropic)
    #_, d2 = open_files('tracer_soc_mars_mola_topo_lh_eps_' + \
    #                '25_gamma_0.093_cdod_clim_scenario_7.4e-05', isentropic)
            
    tinds = [450, 479]
    
    if not isentropic:
        d = d.sel(pfull = level, method="nearest")
    else:
        d = d.sel(level = level, method="nearest")


    d = d.test_tracer+2
    
    if isentropic:
        level = d.level.values
    
    

    
    if proj == 'PC':
        projection = ccrs.PlateCarree()
    elif proj == 'NP':
        projection = ccrs.NorthPolarStereo()
        d  =  d.where( d.lat > 60, drop = True)
        hem = 'NH'
        levels = [3.9]
    elif proj == 'SP':
        projection = ccrs.SouthPolarStereo()
        d  =  d.where( d.lat <-60, drop = True)
        hem = 'SH'
        levels = [0.1,0.6]

    
    
    fig, axs = plt.subplots(nrows=1,ncols=2,
                            subplot_kw = {'projection': projection})
    for i, ax in enumerate(fig.axes):
        if proj == 'NP':
            funcs.make_stereo_plot(ax, [np.max(d.lat), 80, 70, np.min(d.lat)],# 60,],
                              [-180, -120, -60, 0, 60, 120, 180],
                              circle, alpha = 0.3, linestyle = '--',)
        elif proj == 'SP':
            funcs.make_stereo_plot(ax, [np.max(d.lat), -70, -80, np.min(d.lat)],
                              [-180, -120, -60, 0, 60, 120, 180],
                              circle, alpha = 0.3, linestyle = '--',)
        d1 = d.isel(time=tinds[i])
    
        CS = ax.contour(d.lon,d.lat,d1,transform=ccrs.PlateCarree(),
                    levels = levels[i], colors='k',)
                    #levels=levels, colors='k')
        CS.levels = [funcs.nf(val) for val in CS.levels]
        ax.clabel(CS, CS.levels, inline=1, fmt = fmt, fontsize ='small')

    


    fig.suptitle('Showing contour length')

    fig.savefig('/user/home/xz19136/Figures/mars_analysis/maps/contours/' + \
        'keff_calc_diagram.png',
        bbox_inches='tight',dpi=300)

if __name__ == "__main__":

    exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                    '25_gamma_0.093_cdod_clim_scenario_7.4e-05'
    isentropic = False
    exps = []
    gamma = [0.00,0.093]
    eps = np.arange(10,55,5)
    gamma = [0.093]#093]
    
    for i in gamma:
        for j in eps:
            exps.append('tracer_soc_mars_mola_topo_lh_eps_' + \
                    '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' %(j,i))
    
    #exps = []
    for d in [3.7e-5,7.4e-5,1.48e-4,2.96e-4]:
        exps.append('tracer_soc_mars_mola_topo_lh_eps_25_' + \
                    'gamma_0.093_cdod_clim_scenario_' + str(d))

    tind = 102
    proj='NP'
    level = 300
    n = 24
    mean = 10
    PVmax = True
    isentropic=True

    plot_two_diagram(proj='SP')


    #plot_contour_keff_all_maps('attribution', tind=tind,n=n,level=1,proj=proj)
    #for exp_name in exps:
    #    plot_PV_map(exp_name,mean=mean,tind=tind,proj=proj,level=level,)
    #for exp_name in exps:
    #    for n in np.arange(0,30):

            #plot_contour_keff_map(exp_name,isentropic=isentropic,mean=mean,
            #            tind=tind,n=n,level=level,proj=proj,PVmax=PVmax)


    

    
# %%
