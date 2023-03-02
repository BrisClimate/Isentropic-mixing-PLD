# %%
import xarray as xr
import numpy as np
import sys, os

sys.path.append('/user/home/xz19136/Py_Scripts/atmospy/')

import analysis_functions as funcs

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


def open_files(exp_name, isentropic=False, tname='test_tracer'):
    
    if not isentropic:
        isent = ''
    else:
        isent = '_isentropic'
    
    if tname is not 'test_tracer':
        tname = 'test_tracer'
    ds = xr.open_dataset(
        	path + exp_name + '/keff%s_%s.nc' % (isent, tname), decode_times = False,)

    d = xr.open_dataset(
        	path + exp_name + '/atmos%s.nc' % isent, decode_times = False,)
    
    return ds, d

def get_PV_lats(di, hem='nh'):
    '''
    Lait-scale PV and then return the latitude of maximum PV on given 
    pressure levels'''
    laitPV = funcs.lait(di.PV,di.theta,theta0,kappa=kappa)

    l = []
    for a in range(len(di.pfull)):
        try:
            x = laitPV.mean(dim=["lon"]).isel(pfull=a)

            x = x.where(x != np.nan, drop = True)
            if hem == 'nh':
                phi_PV, _ = funcs.calc_PV_max(x, x.lat)
            else:
                phi_PV, _ = funcs.calc_PV_max(-x, x.lat)                    
            l.append(phi_PV)
        except:
            l.append(np.nan)

    return l

def plot_smapshots(eps, gamma, isentropic=False, level=300, hem='nh', tname='test_tracer'):
    if hem == 'nh':
        proj = ccrs.NorthPolarStereo()
    else:
        proj = ccrs.SouthPolarStereo()
    
    exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps, gamma)

    _, d = open_files(exp_name, isentropic, tname=tname)

    try:
        #ds = ds.sel(level = level, method="nearest")
        d  =  d.sel(level = level, method="nearest")
    except:
        #ds = ds.sel(pfull = level, method="nearest")
        d  =  d.sel(pfull = level, method="nearest")

    if hem == 'nh':
        latm = d.lat.max().values
        if tname == 'test_tracer':
            lims = [1,2]
        else:
            lims = [1,8]
    else:
        latm = d.lat.min().values
        if tname == 'test_tracer':
            lims = [-2,-1]
        else:
            lims = [-8,-1]

    if tname == 'test_tracer':
        ds = d.test_tracer
    else:
        theta = d.level
        laitPV = funcs.lait(d.PV,theta,theta0,kappa=kappa)
        ds = laitPV*10**4

    fig, axs = plt.subplots(nrows=2,ncols=4, figsize = (14,8),
                            subplot_kw = {'projection':proj})


    boundaries, cmap, norm = funcs.new_cmap(lims, extend='both')
    

    for i, ax in enumerate(fig.axes):
        ax.text(0.05, 0.95, string.ascii_lowercase[i], transform=ax.transAxes, 
            size='large')

        ls = d.mars_solar_long.isel(time=90+3*i).values

        c1 = plot_map_contourf(ds.isel(time=90+3*i), ax, cmap, norm, boundaries,
            hem=hem, latm = latm,
            title='L$_s = %i^\circ$' % ls)

    fig.show()
    cb = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),extend = 'both',
        ticks=boundaries[slice(None,None,2)],shrink=0.8,
        ax = axs)
    if tname == 'test_tracer':
        cb.set_label(label='test tracer mixing ratio',fontsize='large')
    else:
        cb.set_label(label='PV (MPVU)',fontsize='large')

    fig.suptitle('$\epsilon=%i^\circ, \gamma=%.3f$' % (eps, gamma))

    fig.savefig('/user/home/xz19136/Figures/mars_analysis/maps/timestep_' + \
        'eps_%i_gamma_%.3f_%s_%s_%.1f.png' % (eps, gamma, tname, hem, level),
        bbox_inches='tight')

def plot_maps_all_exp(exp='attr',isentropic=False, tind=101, level=0.2, \
                      hem='nh', tname='test_tracer', mean=10):
    if hem == 'nh':
        proj = ccrs.NorthPolarStereo()
    else:
        proj = ccrs.SouthPolarStereo()
        
    if exp == 'attr':

        fig, axs = plt.subplots(nrows=2, ncols=4, figsize = (12,8),
                            subplot_kw = {'projection':proj})
    else:
        eps = [10,15,20,25,30,35,40,45,50]
        gamma = [0.093,0]
    
    

        fig, axs = plt.subplots(nrows=2,ncols=len(eps), figsize = (len(eps)*3,8),
                            subplot_kw = {'projection':proj})


    if hem == 'nh':
        latm = 87
        if tname == 'test_tracer':
            lims = [0.5,1.5]
        else:
            lims = [1,8]
    else:
        latm = -87
        if tname == 'test_tracer':
            lims = [-2,-1]
        else:
            lims = [-8,-1]



    boundaries, cmap, norm = funcs.new_cmap(lims, extend='both')
    for i, ax in enumerate(fig.axes):
        ax.text(0.05, 0.95, string.ascii_lowercase[i], transform=ax.transAxes, 
            size='large')
    
    exps = []
    if exp == 'attr':
        titles = ['Control', 'LH', 'D', 'LH+D', 'T', 'LH+T', 'D+T', 'LH+D+T']
        for t in ['', '_mola_topo']:
            for d in ['', '_cdod_clim_scenario_7.4e-05']:
                for l in ['', '_lh']:
                    exps.append('tracer_soc_mars%s%s_eps_25_gamma_0.093%s' % (t, l, d))

    else:
        titles = []
        for j in range(len(eps)):
            for i in range(len(gamma)):
                exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                    '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps[j], gamma[i])

                titles.append('$\epsilon=%i$' % (eps[j]))

    for i in range(len(exps)):
        exp_name = exps[i]
        ax = fig.axes[i]

        try:
            _, d = open_files(exp_name, isentropic, tname=tname)
            print(exp_name)
        except:
            continue

        try:
            d  =  d.sel(level = level, method="nearest")
        except:
            d  =  d.sel(pfull = level, method="nearest")

        latm = np.max(np.abs(d.lat.values))
        
        tind, m = funcs.get_timeslice(tind, mean)

        if exp == 'attr':
            if i == 0:
                ls = d.mars_solar_long.isel(time=tind).values
                ytext = 'No topography'
            elif i == 4:
                ytext = 'MOLA topography'
            else:
                ytext = None
        else:
            if i == 0:
                ls = d.mars_solar_long.isel(time=tind).values
                ytext = '$\gamma = %.3f$' % gamma[0]
            elif i == 4:
                ytext = '$\gamma = %.3f$' % gamma[1]
            else:
                ytext = None
    
        title = titles[i]
        
        

        if m != 0:
            tslice = slice(tind-m, tind+m)
            ds = d.isel(time=tslice)
            ds = ds.mean(dim="time")
        else:
            ds = d.isel(time=tind)

        if tname == 'test_tracer':
            x = ds.test_tracer
        else:
            try:
                theta = ds.level
                laitPV = funcs.lait(ds.PV,theta,theta0,kappa=kappa)
                x = laitPV*10**4
            except:
                laitPV = funcs.lait(ds.PV,ds.theta, theta0,kappa=kappa)
                x = laitPV*10**4

        c1 = plot_map_contourf(x, ax, cmap, norm, boundaries,
                hem=hem, latm = latm, ytext=ytext, title=title)

        

        if tname != 'test_tracer':
            q_max = []
            if hem == 'nh':
                x =  x.where( x.lat >= 52, drop = True)
            else:
                x = -x.where(x.lat <= -52, drop = True)

            for l in range(len(x.lon)):
                q = x.isel(lon = l)
                q = q.where(q != np.nan, drop = True)
                try:
                    q0, _ = funcs.calc_jet_lat(q, q.lat)
                except:
                    q0 = latm
                q_max.append(q0)

            c0 = ax.plot(x.lon, q_max,transform=ccrs.PlateCarree(),
                 color='blue', linewidth=1)

    fig.show()
    cb = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),extend = 'both',
            ticks=boundaries[slice(None,None,2)],
            ax = axs)
    cb.set_label(label='%s mixing ratio' % tname,fontsize='large')

    fig.savefig('/user/home/xz19136/Figures/mars_analysis/maps/all_exp_' + \
            '%s_%s_%.1f_%i_%s.png' % (tname, hem, level, tind, exp),
            bbox_inches='tight')

def plot_map_contourf(ds, ax, cmap, norm, boundaries, title=None, ytext=None, hem='nh', latm = 85):
    if title is not None:
        ax.set_title(title)
    if ytext is not None:
        ax.text(
            -0.46, 0.5, ytext,
            ha='right',
            va='center',
            transform=ax.transAxes,
            rotation='vertical',
            fontsize='large',
        )
    if hem == 'nh':
        ds = ds.where(ds.lat >= 52, drop = True)
        funcs.make_stereo_plot(ax, [latm, 80, 70, 60,],
                          [-180, -120, -60, 0, 60, 120, 180],
                          circle, alpha = 0.3, linestyle = '--',)
    else:
        ds = ds.where(ds.lat <= -52, drop = True)
        funcs.make_stereo_plot(ax, [-60, -70, -80, latm,],
                          [-180, -120, -60, 0, 60, 120, 180],
                          circle, alpha = 0.3, linestyle = '--',)
    
    c0 = ax.contourf(ds.lon,ds.lat,ds,cmap=cmap,transform=ccrs.PlateCarree(),
        norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])
    return #c0


def plot_keff_point_evolution(tname='test_tracer', isentropic=False, \
            tind=101, lat = 70, level = 1.0,):
    '''
    Plot effective diffusivity evolution at a given point,
    in order to understand strength of the transport barrier and mixing
    within the vortex.
    '''

    eps = [25,]
    gamma = [0.093,0]
    lat = 60
    level = 2.

    colors = plt.cm.viridis(np.linspace(0,1,int(len(eps))))
    matplotlib.rcParams['axes.prop_cycle'] = (
            cycler('color', colors) * \
            cycler('linestyle', ['-'])
        )

    

    for j in range(len(eps)):
        for i in range(len(gamma)):
            exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps[j], gamma[i])

            fig, ax = plt.subplots(nrows=1,ncols=1, figsize = (5,3),)
            try:
                ds, _ = open_files(exp_name, isentropic, tname=tname)
                print(exp_name)
            except:
                continue
                
            print(ds.nkeff.min().values)
            ls = ds.mars_solar_long
            print(ls[0].values, ls[-1].values)
            ds = xr.ufuncs.log(ds.nkeff)
            dis = ds.sel(new = lat, method = 'nearest').sel(level = level, method = 'nearest')
            ax.plot(dis.time, dis,)
            xlocs = [i for i in ax.get_xticks()]
            ls = ls.interp(time=xlocs,kwargs={"fill_value":"extrapolate"})#, d.time)
    
            ax.set_xticklabels(['%i' % i for i in ls])
            ax.set_xlabel('$\mathrm{L_s}$')
            ax.set_ylabel('$k_{eff}$')

            ymin, ymax = ax.get_ylim()

            for k in range(len(dis.time)):
                if k % 30 == 0:
                    ax.plot([dis.time[k], dis.time[k]], [ymin, ymax], alpha=0.3, linewidth = 1)



            fig.savefig('/user/home/xz19136/Figures/mars_analysis/zonal_lineplots/' \
                + '%iN_%.1fhPa_eps%i_gamma%.3f.png' % (lat, level, eps[j], gamma[i]),
                bbox_inches='tight')


def plot_line(dis, eps, gamma, ax):
    '''
    Plot keff vs equivalent latitude.
    '''

    if gamma == 0.093:
        marker = 'v'
        label = '$\epsilon=%i$' % (eps)
    else:
        marker = 's'
        label = None
    ax.plot(dis.new, dis, label = label)

def plot_phimin(phi, keff, gamma, ax, col):
    '''
    Add a marker to show the keff min latitude
    '''
    if gamma == 0.093:
        marker = 'v'
    else:
        marker = 's'
    ax.plot([phi, phi], [-0.5,keff], linestyle='--',color = col,alpha=0.7)

def plot_keff_line_plot(tname='test_tracer', isentropic=False, \
            mean=None, tind=101, method='vert_int', \
            level = None):
    '''
    Plot effective diffusivity zonal profile on a given level in a given hemisphere,
    in order to understand strength of the transport barrier and mixing
    within the vortex.
    '''

    eps = [10,15,20,25,30,35,40,45,50]
    gamma = [0.093,0.]

    colors = plt.cm.viridis(np.linspace(0,1,int(len(eps))))
    matplotlib.rcParams['axes.prop_cycle'] = (
            cycler('color', colors) * \
            cycler('linestyle', ['-'])
        )

    fig, axs = plt.subplots(nrows=2,ncols=1, figsize = (5,7),)

    for j in range(len(eps)):
        for i in range(len(gamma)):
            exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps[j], gamma[i])
    
            try:
                ds, d = open_files(exp_name, isentropic, tname=tname)
                print(exp_name)
            except:
                continue

            if i == 0:
                ls = ds.mars_solar_long.isel(time=tind).values
            

            tind, m = funcs.get_timeslice(tind, mean)

            if m != 0:
                tslice = slice(tind-m, tind+m)
                
                dis = ds.nkeff.isel(time=tslice)
                dis = dis.mean(dim="time")
            else:
                dis = ds.nkeff.isel(time=tind)           
            dis = xr.ufuncs.log(dis)
#            dis = dis.to_masked_array()
            
            if method == 'vert_int':
                if not isentropic:
                    weights = dis.level
                    dis = dis.weighted(weights)
                dis = dis.mean(dim="level")

            elif method == '2_0.1_int' and isentropic == False:
                dis = dis.where(dis.level <=   2, drop = True)
                dis = dis.where(dis.level >= 0.2, drop = True)

                if not isentropic:
                    weights = dis.level
                    dis = dis.weighted(weights)
                #if eps[j] == 10 and gamma[i] == 0.000:
                #    for level in dis.level:
                #        print(-dis.sel(level=level))
                
                dis = dis.mean(dim="level")
                
            
            elif method == '200_400_int' and isentropic == True:
                dis = dis.where(dis.level <= 400, drop = True)
                dis = dis.where(dis.level >= 200, drop = True)
                dis = dis.mean("level")
                
            
            elif method == 'level' and level is not None:
                dis = dis.sel(level = level, method="nearest")
                
                    
            else:
                raise ValueError('Inconsistent choice of levels and integration')

            if i == 0:
                marker = 'v'
                label = '$\epsilon=%i$' % (eps[j])
            else:
                marker = 's'
                label = None
            axs[i].plot(dis.new, dis, label = label)
            

    axs[0].set_ylabel('Effective diffusivity')
    axs[1].set_ylabel('Effective diffusivity')
    axs[1].set_xlabel('Equivalent latitude ($^\circ$N)')
    
    axs[0].set_xlim([d.lat[0], d.lat[-1]])
    axs[1].set_xlim([d.lat[0], d.lat[-1]])
    
    axs[0].set_xticklabels([])
    #if PVmax:
    #    axs[0].plot(eps, ppv[:, 0], label = '$\gamma=%.3f$, $\phi_{PV}$' % gamma[0], color='g', linestyle= '-', marker='v')
    #    axs[0].plot(eps, ppv[:, 1], label = '$\gamma=%.3f$, $\phi_{PV}$' % gamma[1], color='g', linestyle='--', marker='s')
    #    
    #    ax2 = axs[1].twinx()
    #    ax2.plot(eps, xpv[:, 0], label = 'PV max', color='g', linestyle= '-', marker='v')
    #    ax2.plot(eps, xpv[:, 1], label =     None, color='g', linestyle='--', marker='s')
    #    ax2.set_ylabel('Maximum PV (MPVU)')
    #    #ax2.plot(eps, mxu[:, 0], label = 'u max', color='r', linestyle= '-', marker='v')
    #    #ax2.plot(eps, mxu[:, 1], label =            None, color='r', linestyle='--', marker='s')
    axs[0].legend(loc='center left', bbox_to_anchor=(1.05,-0.05,),
                 borderaxespad=0, fontsize='large')
    #axs[1].legend()
    #ax2.legend()

    if mean is not None:
        m = ' (%i sol average)' % mean
        mean = '_%isol' % mean
    else:
        m = ''
        mean = ''
    
    axs[0].set_title('Ls = $%i^\circ$%s, %s' % (ls, m, tname))
    
    fig.savefig('/user/home/xz19136/Figures/mars_analysis/zonal_lineplots/all_exp_' + \
            '%s_%03d%s_%s.png' % (tname, tind, mean, method),
            bbox_inches='tight')

            

def plot_barrier_strength(tname='test_tracer', isentropic=False, \
            mean=None, tind=101, hem='nh', method='vert_int', \
            level = None, PVmax = False, phi_max=True):
    '''
    Plot effective diffusivity minima and maxima in a given hemisphere,
    in order to understand strength of the transport barrier and mixing
    within the vortex.
    '''

    eps = [10,15,20,25,30,35,40,45,50]
    gamma = [0.093,0.]

    colors = plt.cm.viridis(np.linspace(0,1,int(len(eps))))
    matplotlib.rcParams['axes.prop_cycle'] = (
            cycler('color', colors) * \
            cycler('linestyle', ['-'])
        )

    fig,  axs  = plt.subplots(nrows=2,ncols=1, figsize = (5,7),)
    fig1, axs1 = plt.subplots(nrows=2,ncols=1, figsize = (5,7),)
    if phi_max:
        fig2, axs2 = plt.subplots(nrows=2,ncols=1, figsize = (5,7),)
    
    for i in [0,1]:
        ytext = '$\gamma = %.3f$' % gamma[i]
        for ax in [axs1[i]]:
            ax.text(
                    -0.15, 0.5, ytext,
                    ha='right',
                    va='center',
                    transform=ax.transAxes,
                    rotation='vertical',
                    fontsize='large',
            )
    
    lat = np.full((len(eps),len(gamma)), np.nan)
    mnm = np.full((len(eps),len(gamma)), np.nan)
    if phi_max:
        xeq = np.full((len(eps),len(gamma)), np.nan)
        leq = np.full((len(eps),len(gamma)), np.nan)
        xpo = np.full((len(eps),len(gamma)), np.nan)
        lpo = np.full((len(eps),len(gamma)), np.nan)
    
    if PVmax:
        xpv = np.full((len(eps),len(gamma)), np.nan)
        ppv = np.full((len(eps),len(gamma)), np.nan)

    for j in range(len(eps)):
        for i in range(len(gamma)):
            exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps[j], gamma[i])

            
            ds, d = open_files(exp_name, isentropic, tname=tname)
            print(exp_name)
            if PVmax:
                d = d.PV
            ds = ds.interp({'new':d.lat},method='linear')
            

            if i == 0:
                ls = ds.mars_solar_long.isel(time=tind).values
            
            tind, m = funcs.get_timeslice(tind, mean)

            if m != 0:
                tslice = slice(tind-m, tind+m)
                if PVmax:
                    di  =  d.isel(time=tslice)
                    di  =  di.mean(dim="time")
                dis = ds.nkeff.isel(time=tslice)
                dis = dis.mean(dim="time")
            else:
                if PVmax:
                    di  =  d.isel(time=tind)
                dis = ds.nkeff.isel(time=tind)

            dis = xr.ufuncs.log(dis)

            if hem == 'nh':
                if PVmax:
                    di  =  di.where( di.lat >= 30, drop = True)
                dis = dis.where(dis.new >= 30, drop = True)
            else:
                if PVmax:
                    di  =  di.where( di.lat <= -30, drop = True)
                dis = dis.where(dis.new <= -30, drop = True)
        
            if method == 'vert_int':
                if not isentropic:
                    weights = dis.level
                    dis = dis.weighted(weights)
                dis = dis.mean(dim="level")
                if PVmax:
                    if not isentropic:
                        weights = di.pfull
                        di = di.weighted(weights)
                    di = di.mean(dim="pfull")

            elif method == 'int' and isentropic == False:
                dis = dis.where(dis.level <= level[0], drop = True)
                dis = dis.where(dis.level >= level[1], drop = True)
                if not isentropic:
                    weights = dis.level
                    dis = dis.weighted(weights)
                if PVmax:
                    di  =  di.where( di.pfull <= level[0], drop = True)
                    di  =  di.where( di.pfull >= level[1], drop = True)
                    if not isentropic:
                        weights = di.pfull
                        di = di.weighted(weights)
                    di = di.mean(dim="pfull")
                #if eps[j] == 10 and gamma[i] == 0.000:
                #    for level in dis.level:
                #        print(-dis.sel(level=level))

                dis = dis.mean(dim="level")


            elif method == '200_400_int' and isentropic == True:
                dis = dis.where(dis.level <= 400, drop = True)
                dis = dis.where(dis.level >= 200, drop = True)
                dis = dis.mean("level")
                if PVmax:
                    di  =  di.where( di.level <= 400, drop = True)
                    di  =  di.where( di.level >= 200, drop = True)

                    di  =  di.mean(dim="level")

            elif method == 'level' and level is not None:
                dis = dis.sel(level = level, method="nearest")
                if PVmax:

                    if not isentropic:
                        di  =  di.sel(pfull = level, method="nearest")
                    else:
                        di  =  di.sel(level = level, method="nearest")

            else:
                raise ValueError('Inconsistent choice of levels and integration')
            di = di.mean(dim="lon")
            plot_line(dis,eps[j],gamma[i],axs1[i])

            x = dis.where(dis != np.nan, drop = True)
            di = di.where( di != np.nan, drop = True)
            try:
                phi_min, keff_min = funcs.calc_PV_max(-x, x.new)
            except:
                if hem == 'nh':
                    phi_min = x.new[-1]
                    keff_min = -x[-1]
                else:
                    phi_min = x.new[0]
                    keff_min = -x[0]
            keff_min = - keff_min
            if PVmax:
                if hem == 'nh':
                    phi_pv, pv_max = funcs.calc_PV_max(di, di.lat)
                else:
                    try:
                        phi_pv, pv_max = funcs.calc_PV_max(-di, di.lat)     
                    except:
                        pv_max = -di.min().values
                        phi_pv = di.lat[0]

            if phi_max:
                xi = x.where(np.abs(x.new) <= np.abs(phi_min), drop = True)
                try:
                    leq[j,i], xeq[j,i] = funcs.calc_PV_max(xi, xi.new)
                except:
                    if hem == 'nh':
                        leq[j,i] = xi.new[0]
                        xeq[j,i] = xi[0]
                    else:
                        leq[j,i] = xi.new[-1]
                        xeq[j,i] = xi[-1]

                #xi = x.where(np.abs(x.new) >= np.abs(phi_min), drop = True)
                #try:
                #    lpo[j,i], xpo[j,i] = funcs.calc_PV_max(xi, xi.new)
                #except:
                if hem == 'nh':
                    lpo[j,i] = xi.new[0]
                    xpo[j,i] = xi[0]
                else:
                    lpo[j,i] = xi.new[-1]
                    xpo[j,i] = xi[-1]

            

            lat[j, i] = phi_min
            mnm[j, i] = keff_min
            if PVmax:
                xpv[j, i] = pv_max*10**4
                ppv[j, i] = phi_pv

            plot_phimin(phi_min,keff_min,gamma[i],axs1[i],colors[j])
            #plot_phimin(lpo[j,i],xpo[j,i],gamma[i],axs1[i],colors[j])
            #plot_phimin(leq[j,i],xeq[j,i],gamma[i],axs1[i],colors[j])

    axs[0].plot(eps, lat[:, 0], label = '$\gamma=%.3f$, $k_{eff}$' % gamma[0],
                markersize= 4, marker='v', color = 'xkcd:darkgreen', linestyle = '-')
    axs[0].plot(eps, lat[:, 1], label = '$\gamma=%.3f$, $k_{eff}$' % gamma[1],
                markersize= 4, marker='s', color = 'xkcd:darkgreen', linestyle = '--')

    if PVmax:
        #ax1 = axs[0].twinx()
        axs[0].plot(eps, ppv[:, 0], label = '$\gamma=%.3f$, $\phi_{PV}$' % gamma[0],
                markersize= 4, marker='v', color = 'xkcd:orchid', linestyle = '-')
        axs[0].plot(eps, ppv[:, 1], label = '$\gamma=%.3f$, $\phi_{PV}$' % gamma[1],
                markersize= 4, marker='s', color = 'xkcd:orchid', linestyle = '--')

    axs[1].plot(eps, mnm[:, 0], label = '$\gamma=%.3f$, $k_{eff}$' % gamma[0],
                markersize= 4, marker='v', color = 'xkcd:darkgreen', linestyle = '-')
    axs[1].plot(eps, mnm[:, 1], label = '$\gamma=%.3f$, $k_{eff}$' % gamma[1],
                markersize= 4, marker='s', color = 'xkcd:darkgreen', linestyle = '--')

    #axs[1].plot(eps, mxm[:, 0], label = '$k_{eff}$ max', color='r', linestyle= '-', marker='v')
    #axs[1].plot(eps, mxm[:, 1], label =            None, color='r', linestyle='--', marker='s')

    axs[1].set_ylabel('$k_{eff}$ min')
    #axs[1].set_yticklabels(ax2.get_yticks(), color = 'b')
    axs[0].set_ylabel('Latitude ($^\circ$N)')
    axs[1].set_xlabel('Obliquity $\epsilon$ ($^\circ$)')
    axs[0].set_xticklabels([])

    
    if PVmax:
        ax2 = axs[1].twinx()
        ax2.plot(eps, xpv[:, 0], label = '$\gamma=%.3f$, PV' % gamma[0],
                marker='v', markersize=4, color = 'xkcd:orchid', linestyle = '-')
        ax2.plot(eps, xpv[:, 1], label = '$\gamma=%.3f$, PV' % gamma[1],
                marker='s', markersize=4, color = 'xkcd:orchid', linestyle = '--')
        ax2.set_ylabel('Maximum PV (MPVU)')
        #ax2.plot(eps, mxu[:, 0], label = 'u max', color='r', linestyle= '-', marker='v')
        #ax2.plot(eps, mxu[:, 1], label =            None, color='r', linestyle='--', marker='s')

    axs[0].legend()
    axs[1].legend()
    #ax1.legend()
    ax2.legend(loc=6)

    if phi_max:
        axs2[0].set_ylabel('Equatorward $k_{eff}$ gradient')
        axs2[1].set_ylabel(   'Poleward $k_{eff}$ gradient')

        axs2[1].set_xlabel('Obliquity $\epsilon$ ($^\circ$)')
        axs2[0].set_xticklabels([])

        grad_e = (mnm - xeq)/(lat - leq)
        grad_p = (mnm - xpo)/(lat - lpo)
        axs2[0].plot(eps, grad_e[:, 0], label = '$\gamma=%.3f$, $k_{eff}$' % gamma[0],
                markersize= 4, marker='v', color = 'xkcd:crimson', linestyle = '-')
        axs2[0].plot(eps, grad_e[:, 1], label = '$\gamma=%.3f$, $k_{eff}$' % gamma[1],
                markersize= 4, marker='s', color = 'xkcd:crimson', linestyle = '--')

        axs2[1].plot(eps, grad_p[:, 0], label = '$\gamma=%.3f$, $k_{eff}$' % gamma[0],
                markersize= 4, marker='v', color = 'xkcd:crimson', linestyle = '-')
        axs2[1].plot(eps, grad_p[:, 1], label = '$\gamma=%.3f$, $k_{eff}$' % gamma[1],
                markersize= 4, marker='s', color = 'xkcd:crimson', linestyle = '--')

    if mean is not None:
        m = int(mean/2)
        mean = ' (%i sol average)' % mean
        sols = '_%i-%isai' % ((tind-m)%30,(tind+m)%30)
    else:
        mean = ''
        sols = '_%isai' % (tind % 30)


    if PVmax:
        pv = '_pv'
    else:
        pv = ''

    if method == 'level':
        method=method+str(level)
    if method == 'int':
        method=method+str(level[0])+'-'+str(level[1])

    fig.suptitle('Ls = $%i^\circ$%s, %s' % (ls, mean, tname))
    
    fig.savefig('/user/home/xz19136/Figures/mars_analysis/barrier_strength/all_exp_' + \
            '%s_%s_%03d%s_%s%s.png' % (tname, hem, tind, sols, method, pv),
            bbox_inches='tight')


    fig2.suptitle('Ls = $%i^\circ$%s, %s' % (ls, mean, tname))
    
    fig2.savefig('/user/home/xz19136/Figures/mars_analysis/barrier_strength/grad_all_exp_' + \
            '%s_%s_%03d%s_%s%s.png' % (tname, hem, tind, sols, method, pv),
            bbox_inches='tight')

    axs1[0].set_ylabel('Effective diffusivity')
    axs1[1].set_ylabel('Effective diffusivity')
    axs1[1].set_xlabel('Equivalent latitude ($^\circ$N)')
    
    axs1[0].set_xlim([di.lat[0], di.lat[-1]])
    axs1[1].set_xlim([di.lat[0], di.lat[-1]])
    
    axs1[0].set_xticklabels([])
    #if PVmax:
    #    axs[0].plot(eps, ppv[:, 0], label = '$\gamma=%.3f$, $\phi_{PV}$' % gamma[0], color='g', linestyle= '-', marker='v')
    #    axs[0].plot(eps, ppv[:, 1], label = '$\gamma=%.3f$, $\phi_{PV}$' % gamma[1], color='g', linestyle='--', marker='s')
    #    
    #    ax2 = axs[1].twinx()
    #    ax2.plot(eps, xpv[:, 0], label = 'PV max', color='g', linestyle= '-', marker='v')
    #    ax2.plot(eps, xpv[:, 1], label =     None, color='g', linestyle='--', marker='s')
    #    ax2.set_ylabel('Maximum PV (MPVU)')
    #    #ax2.plot(eps, mxu[:, 0], label = 'u max', color='r', linestyle= '-', marker='v')
    #    #ax2.plot(eps, mxu[:, 1], label =            None, color='r', linestyle='--', marker='s')
    axs1[0].legend(loc='center left', bbox_to_anchor=(1.05,-0.05,),
                 borderaxespad=0, fontsize='large')
    #axs[1].legend()
    #ax2.legend()
    
    axs1[0].set_title('Ls = $%i^\circ$%s, %s' % (ls, mean, tname))
    
    fig1.savefig('/user/home/xz19136/Figures/mars_analysis/zonal_lineplots/all_exp_' + \
            '%s_%03d%s_%s.png' % (tname, tind, sols, method),
            bbox_inches='tight')
    

    return lat, mnm, xpv, ppv


def plot_keff_cross_all_exp(isentropic=False, tname=None, hem='nh', PVmax=True, mean=None, tind=101):
    eps = [10,15,20,25,30,35,40,45,50]
    gamma = [0.093,0]

    fig, axs = plt.subplots(nrows=2,ncols=len(eps), figsize = (len(eps)*3,7),)
    
    lims = [0,4]

    boundaries, cmap, norm = funcs.new_cmap(lims, extend='max', i = 10, override=True, cols='YlGn')

    for i, ax in enumerate(fig.axes):
        ax.text(0.05, 1.05, string.ascii_lowercase[i], transform=ax.transAxes, 
            size='large')
        ax.set_yscale('log')
        ax.set_ylim([5.5,0.01])

    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ticks=boundaries[slice(None,None,2)],
        ax = axs, pad = 0.01,
        label='%s normalized effective diffusivity' % (tname), extend='max')

    for j in range(len(eps)):
        for i in range(len(gamma)):
            exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps[j], gamma[i])

            ax = axs[i,j]

            try:
                ds, d = open_files(exp_name, isentropic, tname=tname)
                print(exp_name)
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

            tind, m = funcs.get_timeslice(tind, mean)            

            if hem == 'nh':
                d  =  d.where( d.lat >= 0, drop = True)
                ds = ds.where(ds.new >= 0, drop = True)
                ds = ds.interp({'new':d.lat})
            else:
                d  =  d.where( d.lat <= 0, drop = True)
                ds = ds.where(ds.new <= 0, drop = True)
                ds = ds.interp({'new':d.lat})

            if m != 0:
                tslice = slice(tind-m, tind+m)
                
                di  =  d.isel(time=tslice)
                di  =  di.mean(dim="time")
                dis = ds.nkeff.isel(time=tslice)
                dis = dis.mean(dim="time")
            else:
                di  =  d.isel(time=tind)
                dis = ds.isel(time=tind)

            if not isentropic:
                
                dis = dis.where(dis.level <= 5.5, drop = True)
                c1=ax.contourf(dis.new, dis.level, np.log(dis),
                        cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])

                ax.contour(di.lat, di.pfull, di.theta.mean(dim=["lon"]).transpose(),
                        levels = [200,300,400,500,600,700,800,900], colors='k', linestyles='--',linewidths=0.5)

                c0 = ax.contour(di.lat, di.pfull, di.ucomp.mean(dim=["lon"]).transpose(),
                        levels=[-50,0,50,100,150], colors='black',linewidths=1)
                c0.levels = [funcs.nf(val) for val in c0.levels]
                ax.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
                if PVmax:
                    l = get_PV_lats(di, hem=hem)
                    ax.plot(l, di.pfull, linestyle='-', color='xkcd:orchid', linewidth=3)
    
    if mean is not None:
        mean = ' (%i sol average)' % mean
        sols = '_%i-%isai' % ((tind-m)%30,(tind+m)%30)
    else:
        mean = ''
        sols = '_%isai' % (tind % 30)
    
    fig.suptitle('Zonal mean cross-section of effective diffusivity, Ls = $%i^\circ$%s, %s' % (ls, mean, tname))
    fig.savefig('/user/home/xz19136/Figures/mars_analysis/xsections/all_exp_xsect_' + \
            '%s_%s_%03d%s.png' % (tname, hem, tind, sols),
            bbox_inches='tight')

def plot_keff_cross_section(eps, gamma, tname, tind=101, hem='nh', isentropic=False):
    exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps, gamma)

    ds, d = open_files(exp_name, isentropic, tname=tname)
    ds = ds.isel(time=tind)
    d  =  d.isel(time=tind)
    lims = [0,4]

    boundaries, cmap, norm = funcs.new_cmap(lims, extend='max', i = 10, override=True, cols='YlGn')

    
    ls = ds.mars_solar_long.values
    if hem == 'nh':
        ds = ds.where(ds.new >= 45, drop = True)
        d  =  d.where( d.lat >= 45, drop = True)
    else:
        ds = ds.where(ds.new <= -45, drop = True)
        d  =  d.where( d.lat <= -45, drop = True)

    if not isentropic:

        ds = ds.where(ds.level <= 5.5, drop = True)
    else:
        ds = ds.where(ds.level >= 100, drop = True)
    #ds = ds.where(ds.level > 0.1, drop = True)
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    c1 = axs.contourf(ds.new, ds.level, np.log(ds.nkeff),
        cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])#, levels=np.arange(0,100,10))
    if not isentropic:
        c0 = axs.contour(d.lat, d.pfull, d.ucomp.mean(dim=["lon"]).transpose(),
                levels=[-50,0,50,100,150], colors='black',linewidths=1)
        c0.levels = [funcs.nf(val) for val in c0.levels]
        axs.clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize ='small')
        l = get_PV_lats(d, hem=hem)
        
        axs.contour(d.lat, d.pfull, d.theta.mean(dim=["lon"]).transpose(),
                levels = [200,300,400,500,600,700,800,900], colors='k', linestyles='--')
        axs.plot(l, d.pfull, linestyle='-', color='xkcd:orchid', linewidth=3)
        axs.set_yscale('log')
        axs.set_ylim([5.5,0.01])
        isentropic=''
    else:
        isentropic='_isentropic'
    if gamma == 0.093:
        ls = 'Ls = %i,' % ls
    else:
        ls = hem + ' winter,'
    fig.suptitle('$\epsilon = %i, \gamma = %.3f$, %s %s' % (eps, gamma, ls, tname))
    axs.set_xlabel('equivalent latitude ($^\circ$N)')
    axs.set_ylabel('pressure (hPa)')
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ticks=boundaries[slice(None,None,2)],
        ax = axs, pad = 0.01,
        label='normalized effective diffusivity', extend='max')
    fig.savefig('/user/home/xz19136/Figures/mars_analysis/xsections/animation/keff_xsect_' + \
        'eps_%i_gamma_%.3f_%s_%s_%03d%s.png' % (eps, gamma, tname, hem, tind, isentropic),
        bbox_inches='tight')
    

def plot_tracer_evolution(eps, gamma):
    
    exp_name = 'tracer_soc_mars_mola_topo_lh_eps_%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps, gamma)

    if os.path.isfile(path+exp_name+'/atmos.nc'):
        d = xr.open_dataset(
                path + exp_name + '/atmos.nc', decode_times = False,)
        tmean = d.test_tracer.mean(dim="lon").sel(pfull = 0.2,method="nearest")
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        c0 = axs.contourf(tmean.time, tmean.lat, tmean)
        fig.suptitle('$\epsilon = %i$, $\gamma = %.3f$' % (eps, gamma))

            
def plot_evolution_on_levels(eps, gamma, tname, isentropic=False, level=0.2, PVmax=False, smooth=False):
    exp_name = 'tracer_soc_mars_mola_topo_lh_eps_%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps, gamma)

    ds, _ = open_files(exp_name, isentropic, tname)
    nkeff  = ds.nkeff.sel(level=level,method="nearest")
    
    #tmean  = d.test_tracer.sel(pfull=j,method="nearest")

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    #c0 = axs[0].contourf(tmean.time, tmean.lat, tmean.mean(dim="lon"))
    
    lims = [-2,2]

    boundaries, cmap, norm = funcs.new_cmap(lims, extend='both', i =10)
    if not smooth:
        c1 = axs.contourf(nkeff.time, nkeff.new, np.log(nkeff.transpose()),
            norm=norm,cmap=cmap,levels=[boundaries[0]-100]+boundaries+[boundaries[-1]+100])
    else:
        time = funcs.moving_average(nkeff.time, 45)
        keff = funcs.moving_average_2d(nkeff.transpose(), 45)
        
        c1 = axs.contourf(time, nkeff.new, np.log(keff),#.transpose()),
            norm=norm,cmap=cmap,levels=[boundaries[0]-100]+boundaries+[boundaries[-1]+100])
    xlocs = [i for i in axs.get_xticks()]
    ls = ds.mars_solar_long.interp(time=xlocs,kwargs={"fill_value":"extrapolate"})#, d.time)
    
    axs.set_xticklabels(['%i' % i for i in ls])
    axs.set_ylabel('equivalent latitude ($^\circ$N)')
    axs.set_xlabel('L$_s (^\circ)$')
    if not isentropic:
        lev = '%.1f hPa' % level
    else:
        lev = '%i K' % level
    cb = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),extend = 'both',
        ticks=boundaries[slice(None,None,2)], ax = axs,
        orientation='vertical', label='normalized effective diffusivity')
    #fig.colorbar(c1, orientation='vertical', label='normalized effective diffusivity', ax=axs, extend="both")
    axs.set_title('$\epsilon=%i^\circ, \gamma=%.3f$, %s, %s' % (eps, gamma, lev, tname))

    if PVmax == True:
        _, d = open_files(exp_name, isentropic, tname)
        if isentropic:
            d = d.sel(level=level,method="nearest").mean(dim="lon")
            laitPV = funcs.lait(d.PV,d.level,theta0,kappa=kappa)
        else:
            d = d.sel(pfull=level,method="nearest").mean(dim="lon")
            laitPV = funcs.lait(d.PV,d.theta,theta0,kappa=kappa)
        l = []
        if not smooth:
            time = d.time
        else:
            
            lnorth = laitPV.where(d.lat>0, drop=True).where(d.mars_solar_long >= 180, drop = True)
            lsouth = laitPV.where(d.lat<0, drop=True).where(d.mars_solar_long <= 180, drop = True)
            latn = lnorth.lat
            lats = lsouth.lat
            timen   = funcs.moving_average(lnorth.time, 45)
            times   = funcs.moving_average(lsouth.time, 45)
            lnorth = funcs.moving_average_2d(lnorth.transpose(), 45)
            lsouth = funcs.moving_average_2d(lsouth.transpose(), 45)

        for a in range(len(timen)):
            try:
                try:
                    x = lnorth.isel(time=a)
                except:
                    x = lnorth[:, a]
                #x = x.where(x != np.nan, drop = True)
                phi_PV, _ = funcs.calc_PV_max(x, latn)
                l.append(phi_PV)
            except:
                l.append(np.nan)
        axs.plot(timen, l, linestyle='-', color='k')

        l = []
        for a in range(len(times)):
            try:
                try:
                    x = lsouth.isel(time=a)
                except:
                    x = lsouth[:, a]

                #x = x.where(x != np.nan, drop = True)
                phi_PV, _ = funcs.calc_PV_max(-x, lats)
                l.append(phi_PV)
            except:
                l.append(np.nan)
        axs.plot(times, l, linestyle='-', color='k')


if __name__ == "__main__":
    #e = []
    #g = []
    #for gamma in [0.093, 0.00]:
    #    for eps in [10,15,20,25,30,35,40,45,50]:#,30,35,40,45,50]:
    #        isentropic = False
    #        exp_name = 'tracer_soc_mars_mola_topo_lh_eps_%i_gamma_%.3f_cdod_clim_scenario_7.4e-05'
    #        ds, d = open_files(exp_name, isentropic, tname=tname)
    for tind in np.arange(90,121):
        plot_keff_cross_section(25,0.093,'test_tracer',tind=tind)
        #for tind in [60,90]:
        #    for t in ['test_tracer']:#, 'PV']:
        #        plot_keff_cross_all_exp(tname=t,mean=10,tind=115,hem='nh')
                #plot_maps_all_exp(isentropic=False,tind = 101,tname=t, mean=10,hem='nh')
                #for l in [200,250,300,350,400,500]:
    #                try:
    #                    plot_evolution_on_levels(eps, gamma, t, isentropic=True, level=300, PVmax=True,smooth=True)
    #                except:
    #                    continue
                #plot_keff_line_plot(tname=t, tind = 101, isentropic=False, mean=10, method = 'level', level=0.5)
                #lat, mnm, xpv, ppv = \
                #    plot_barrier_strength(
                #        tname=t,tind = 115, mean=10, hem = 'nh', method = 'int', PVmax=True, level=[1.,0.1]
                #        )
                #lat, mnm, xpv, ppv = \
                #    plot_barrier_strength(
                #        tname=t,tind = 111, mean=10, hem = 'nh', method = 'int', PVmax=True, level=[1.,0.1]
                #        )
                #lat, mnm, xpv, ppv = \
                #    plot_barrier_strength(
                #        tname=t,tind = 451, mean=10, hem = 'sh', method = 'int', PVmax=True, level=[1.,0.1],
                #        )
                #lat, mnm, xpv, ppv = \
                #    plot_barrier_strength(
                #        tname=t,tind = 201, mean=10, hem = 'nh', method = 'int', PVmax=True, level=[1.,0.1],
                #        )
                #plot_keff_cross_all_exp(tname=t,mean=10,tind=115,hem='nh')
                #plot_keff_cross_all_exp(tname=t,mean=10,tind=101,hem='nh')
                #plot_keff_cross_all_exp(tname=t,mean=10,tind=491,hem='sh')
                #plot_keff_cross_all_exp(tname=t,mean=10,tind=501,hem='sh')
                #plot_keff_cross_all_exp(tname=t,mean=10,tind=201,hem='nh')
                #plot_keff_cross_all_exp(tname=t,mean=10,tind=61,hem='nh')
                #plot_keff_cross_all_exp(tname=t,mean=10,tind=51,hem='nh')
                #plot_keff_cross_all_exp(tname=t,mean=10,tind=91,hem='nh')
                #plot_keff_cross_all_exp(tname=t,mean=10,tind=81,hem='nh')
                #plot_keff_cross_all_exp(tname=t,mean=10,tind=71,hem='nh')

                #plot_keff_point_evolution(tname=t, isentropic=False, tind=101,)
                #plot_keff_line_plot(tname=t, tind = 101, isentropic=True, mean=10, method = '200_400_int', PVmax=False)
                
                #lat, mnm, xpv, ppv = plot_barrier_strength(tname=t,tind = 101, mean=10, hem = 'nh', method = '2_0.1_int', PVmax=True)
                
                #plot_barrier_strength(tname=t, tind = 451, mean=10, hem = 'sh', method = '2_0.1_int', PVmax=True)                
                #plot_maps_all_exp(isentropic=True,tind = 101,tname=t, mean30=True,hem='nh')
                    #plot_cross_section(eps,gamma,t,hem='nh',isentropic=False,PVmax=True,keff=True)
                    #plot_tracer_evolution(eps,gamma)
    #                plot_smapshots(eps, gamma, isentropic=True,tname=t,level=300)
                #except:
                #    continue
# %%
