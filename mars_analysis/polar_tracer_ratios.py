# %%
import xarray as xr
import numpy as np
import sys, os

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

figpath = '/user/home/xz19136/Figures/mars_analysis/conc_ratios/'
path = '/user/work/xz19136/Isca_data/'
theta, center, radius, verts, circle = atmospy.stereo_plot()
theta0 = 200.
kappa = 1/4.0

if plt.rcParams["text.usetex"]:
    fmt = r'%r \%'
else:
    fmt = '%r'

def plot_polar_conc_ratio_dust(tname='test_tracer', isentropic=False, ylim=80, \
            n=np.arange(1,30), tind=101, hem='nh', method='int', level=[1,0.1]):
    
    scal = [3.7e-5,7.4e-5,1.48e-4,2.96e-4]
    ticks = [1/2, 1, 2, 4]

    colors = plt.cm.viridis(np.linspace(0,1,int(len(n))))
    matplotlib.rcParams['axes.prop_cycle'] = (
            cycler('color', colors) * \
            cycler('linestyle', ['-'])
        )

    fig,  axs  = plt.subplots(nrows=1,ncols=1, figsize = (5,3),)
    conc = np.full((len(n),30), np.nan)
    rats = np.full((len(n),30), np.nan)
    
    
    es = []
    for j in range(len(scal)):
    
        exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
            '25_gamma_0.093_cdod_clim_scenario_' + str(scal[j])

        
        _, d = atmospy.open_files(path, exp_name, isentropic, tname=tname)
        print(exp_name)
                   
        
        
        tind = atmospy.get_init_sol(tind)

        if j == 0:
            ls = d.mars_solar_long.isel(time=tind).values

        tslice = [tind + nsol for nsol in [0]+n]
        di  =  d.isel(time=tslice)
        di  = di.test_tracer+3

        if hem == 'nh':
            di  =  di.where( di.lat >= ylim, drop = True)
        else:
            di  =  di.where( di.lat <=-ylim, drop = True)
        

        if method == 'int' and isentropic == False:
            
            di  =  di.where( di.pfull <= level[0], drop = True)
            di  =  di.where( di.pfull >= level[1], drop = True)
            if not isentropic:
                weights = di.pfull
                di = di.weighted(weights)
            di = di.mean(dim="pfull")


        elif method == 'int' and isentropic == True:
            
            di  =  di.where( di.level <= level[1], drop = True)
            di  =  di.where( di.level >= level[0], drop = True)

                
            di  =  di.mean(dim="level")

        elif method == 'level' and level is not None:
            
            if not isentropic:
                di  =  di.sel(pfull = level, method="nearest").squeeze()
            else:
                di  =  di.sel(level = level, method="nearest").squeeze()

        else:
            raise ValueError('Inconsistent choice of levels and integration')

        di = di.mean(dim="lon")                
        weights = np.cos(np.deg2rad(di.lat))
        di_weighted = di.weighted(weights)
        di = di_weighted.mean(dim="lat")

        di['dust_scale'] = j
        es.append(di)
    conc = xr.concat(es, dim="dust_scale")    

    conc0 = conc.isel(time=0)#.expand_dims({'time':conc.time}).transpose('eps','gamma','time')
    conc = conc.transpose('dust_scale','time')

    
    rats = conc / conc0
    #cols = ['xkcd:darkgreen','xkcd:orchid','xkcd:aquamarine']
    #for nsol in range(len(n)):
    #    rs = rats.isel(time = nsol)
    #    axs[0].plot(rs.eps, rs.sel(gamma = 0.093), label = '$\gamma=%.3f$, %i sols' % (0.093, n[nsol]),
    #            linestyle = '-', )# color = cols[nsol])
    #    axs[1].plot(rs.eps, rs.sel(gamma = 0.000), label = '$\gamma=%.3f$, %i sols' % (0.000, n[nsol]),
    #            linestyle = '--',)# color = cols[nsol])


    cmap = plt.get_cmap("viridis", len(n))
    norm = matplotlib.colors.BoundaryNorm(np.arange(len(n)+1)+0.5,len(n))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # this line may be ommitted for matplotlib >= 3.1

    for nsol in range(len(n)):
        rs = rats.isel(time = nsol)
        axs.plot(rs.dust_scale, rs, label = '$%i sols' % (n[nsol]),
                linestyle = '-', c = cmap(nsol))
    #for i, yi in enumerate(y.T):
    #   ax.plot(x, yi, c=cmap(i))
    axs.set_xticks(rats.dust_scale.values)
    axs.set_xticklabels(ticks)
    cb = fig.colorbar(sm, ticks=n[slice(None,None,2)], ax = axs)
    cb.set_label('Sols after initialization')
    fig.text(0.5,0.0,'Dust scaling',ha='center',va='center')
    #axs[0].set_ylabel('Polar tracer concentration ratio: $x$ sols / initial')
    #axs[1].set_ylabel('Polar tracer concentration ratio: $x$ sols / initial')
    fig.text(-0.03,0.5,'Polar tracer concentration ratio: $x$ sols / initial',
            ha='center',va='center',rotation='vertical')

    if method == 'level':
        method=method+str(level)
    if method == 'int':
        method=method+str(level[0])+'-'+str(level[1])
    
    #axs[0].legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
    #             borderaxespad=0, fontsize='large')
    #axs[1].legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
    #             borderaxespad=0, fontsize='large')
    fig.suptitle('%s initialized at Ls = $%i^\circ$' % (tname, ls))

    fig.savefig(figpath+'dust_' + \
            '%s_%03d_%s_%s.png' % (tname, tind, hem, method),
            bbox_inches='tight', dpi=300)


def plot_polar_conc_ratio(tname='test_tracer', isentropic=False, \
            n=np.arange(1,30), tind=101, hem='nh', method='int', level=[1,0.1], ylim=80):
    eps = [10,15,20,25,30,35,40,45,50]
    gamma = [0.093,0.]

    colors = plt.cm.viridis(np.linspace(0,1,int(len(n))))
    matplotlib.rcParams['axes.prop_cycle'] = (
            cycler('color', colors) * \
            cycler('linestyle', ['-'])
        )

    fig,  axs  = plt.subplots(nrows=2,ncols=1, figsize = (5,6),)
    conc = np.full((len(eps),len(gamma),30), np.nan)
    rats = np.full((len(eps),len(gamma),30), np.nan)
    
    gms = []
    for i in range(len(gamma)):
        es = []
        for j in range(len(eps)):
        
            exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps[j], gamma[i])

            
            _, d = atmospy.open_files(path, exp_name, isentropic, tname=tname)
            print(exp_name)
                       
            
            
            tind = atmospy.get_init_sol(tind)

            if i == 0:
                ls = d.mars_solar_long.isel(time=tind).values

            tslice = [tind + nsol for nsol in [0]+n]
            di  =  d.isel(time=tslice)
            di  = di.test_tracer+3

            if hem == 'nh':
                di  =  di.where( di.lat >= ylim, drop = True)
            else:
                di  =  di.where( di.lat <=-ylim, drop = True)
            

            if method == 'vert_int':
                
                if not isentropic:
                    weights = di.pfull
                    di = di.weighted(weights)
                di = di.mean(dim="pfull")

            elif method == 'int' and isentropic == False:
                
                di  =  di.where( di.pfull <= level[0], drop = True)
                di  =  di.where( di.pfull >= level[1], drop = True)
                if not isentropic:
                    weights = di.pfull
                    di = di.weighted(weights)
                di = di.mean(dim="pfull")


            elif method == 'int' and isentropic == True:
                
                di  =  di.where( di.level <= level[1], drop = True)
                di  =  di.where( di.level >= level[0], drop = True)

                    
                di  =  di.mean(dim="level")

            elif method == 'level' and level is not None:
                
                if not isentropic:
                    di  =  di.sel(pfull = level, method="nearest")
                else:
                    di  =  di.sel(level = level, method="nearest")

            else:
                raise ValueError('Inconsistent choice of levels and integration')

            di = di.mean(dim="lon")                
            weights = np.cos(np.deg2rad(di.lat))
            di_weighted = di.weighted(weights)
            di = di_weighted.mean(dim="lat")

            di['eps'] = eps[j]
            es.append(di)
        es = xr.concat(es, dim="eps")
        es["gamma"] = gamma[i]
        gms.append(es)
    
    conc = xr.concat(gms, dim="gamma")

    

    conc0 = conc.isel(time=0)#.expand_dims({'time':conc.time}).transpose('eps','gamma','time')
    conc = conc.transpose('eps','gamma','time')

    
    rats = conc / conc0
    #cols = ['xkcd:darkgreen','xkcd:orchid','xkcd:aquamarine']
    #for nsol in range(len(n)):
    #    rs = rats.isel(time = nsol)
    #    axs[0].plot(rs.eps, rs.sel(gamma = 0.093), label = '$\gamma=%.3f$, %i sols' % (0.093, n[nsol]),
    #            linestyle = '-', )# color = cols[nsol])
    #    axs[1].plot(rs.eps, rs.sel(gamma = 0.000), label = '$\gamma=%.3f$, %i sols' % (0.000, n[nsol]),
    #            linestyle = '--',)# color = cols[nsol])


    cmap = plt.get_cmap("viridis", len(n))
    norm = matplotlib.colors.BoundaryNorm(np.arange(len(n)+1)+0.5,len(n))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # this line may be ommitted for matplotlib >= 3.1

    for nsol in range(len(n)):
        rs = rats.isel(time = nsol)
        axs[0].plot(rs.eps, rs.sel(gamma = 0.093), label = '$\gamma=%.3f$, %i sols' % (0.093, n[nsol]),
                linestyle = '-', c = cmap(nsol))
        axs[1].plot(rs.eps, rs.sel(gamma = 0.000), label = '$\gamma=%.3f$, %i sols' % (0.000, n[nsol]),
                linestyle = '--', c = cmap(nsol))
    #for i, yi in enumerate(y.T):
    #   ax.plot(x, yi, c=cmap(i))
    cb = fig.colorbar(sm, ticks=n[slice(None,None,2)], ax = axs)
    cb.set_label('Sols after initialization')

    fig.text(0.5,0.06,'Obliquity $\epsilon$ ($^\circ$)',ha='center',va='center')
    #axs[0].set_ylabel('Polar tracer concentration ratio: $x$ sols / initial')
    #axs[1].set_ylabel('Polar tracer concentration ratio: $x$ sols / initial')
    fig.text(0.0,0.5,'Polar tracer concentration ratio: $x$ sols / initial',
            ha='center',va='center',rotation='vertical')

    if method == 'level':
        method=method+str(level)
    if method == 'int':
        method=method+str(level[0])+'-'+str(level[1])
    
    #axs[0].legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
    #             borderaxespad=0, fontsize='large')
    #axs[1].legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
    #             borderaxespad=0, fontsize='large')
    fig.suptitle('%s initialized at Ls = $%i^\circ$' % (tname, ls))

    fig.savefig(figpath+'all_exp_' + \
            '%s_%03d_%s_%s.png' % (tname, tind, hem, method),
            bbox_inches='tight', dpi=300)


def plot_multiple_times_dust(tname='test_tracer', isentropic=True, \
            n=np.arange(1,30), tinds=[60,90,120], hems=['nh','sh'], method='level', level=300, ylim=80):
    
    scal = [3.7e-5,7.4e-5,1.48e-4,2.96e-4]
    ticks = [1/2, 1, 2, 4]

    colors = plt.cm.viridis(np.linspace(0,1,int(len(n))))
    matplotlib.rcParams['axes.prop_cycle'] = (
            cycler('color', colors) * \
            cycler('linestyle', ['-'])
        )

    conc = np.full((len(n),30), np.nan)
    rats = np.full((len(n),30), np.nan)
    fig,  axs  = plt.subplots(nrows=len(tinds),ncols=len(hems), figsize = (5*len(hems),3*len(tinds)), dpi = 300)

    for i, ax in enumerate(fig.axes):
        if hems[i%2] == 'nh':
            ax.set_ylim([0.4,1.0])
        else:
            ax.set_ylim([1.0,3.0])
        
        
        #if i % 2 == 1:
        #    ax.set_yticklabels([])
    
    
    
    for t in range(len(tinds)):
        for h in range(len(hems)):
            es = []
            for j in range(len(scal)):
        
    
                exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                '25_gamma_0.093_cdod_clim_scenario_' + str(scal[j])


                _, d = atmospy.open_files(path, exp_name, isentropic, tname=tname)
                print(exp_name)

                hem = hems[h]

                tind = atmospy.get_init_sol(tinds[t])
                tind = tind % 360
                if hem == 'sh':
                    tind = tind + 360


                ls = d.mars_solar_long.isel(time=tind).values

                tslice = [tind + nsol for nsol in [0]+n]
                di  =  d.isel(time=tslice)
                di  = di.test_tracer+3

                if hem == 'nh':
                    di  =  di.where( di.lat >= ylim, drop = True)
                else:
                    di  =  di.where( di.lat <=-ylim, drop = True)


                if method == 'int' and isentropic == False:

                    di  =  di.where( di.pfull <= level[0], drop = True)
                    di  =  di.where( di.pfull >= level[1], drop = True)
                    if not isentropic:
                        weights = di.pfull
                        di = di.weighted(weights)
                    di = di.mean(dim="pfull")


                elif method == 'int' and isentropic == True:

                    di  =  di.where( di.level <= level[1], drop = True)
                    di  =  di.where( di.level >= level[0], drop = True)


                    di  =  di.mean(dim="level")

                elif method == 'level' and level is not None:

                    if not isentropic:
                        di  =  di.sel(pfull = level, method="nearest")
                    else:
                        di  =  di.sel(level = level, method="nearest")

                else:
                    raise ValueError('Inconsistent choice of levels and integration')

                di = di.mean(dim="lon")                
                weights = np.cos(np.deg2rad(di.lat))
                di_weighted = di.weighted(weights)
                di = di_weighted.mean(dim="lat")

                di['dust_scale'] = j
                es.append(di)
            conc = xr.concat(es, dim="dust_scale")    

            conc0 = conc.isel(time=0)#.expand_dims({'time':conc.time}).transpose('eps','gamma','time')
            conc = conc.transpose('dust_scale','time')

            rats = conc / conc0
            #cols = ['xkcd:darkgreen','xkcd:orchid','xkcd:aquamarine']
            #for nsol in range(len(n)):
            #    rs = rats.isel(time = nsol)
            #    axs[0].plot(rs.eps, rs.sel(gamma = 0.093), label = '$\gamma=%.3f$, %i sols' % (0.093, n[nsol]),
            #            linestyle = '-', )# color = cols[nsol])
            #    axs[1].plot(rs.eps, rs.sel(gamma = 0.000), label = '$\gamma=%.3f$, %i sols' % (0.000, n[nsol]),
            #            linestyle = '--',)# color = cols[nsol])


            cmap = plt.get_cmap("viridis", len(n))
            norm = matplotlib.colors.BoundaryNorm(np.arange(len(n)+1)+0.5,len(n))
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])  # this line may be ommitted for matplotlib >= 3.1

            for nsol in range(len(n)):
                rs = rats.isel(time = nsol)
                axs[t,h].plot(rs.dust_scale, rs, label = '$%i sols' % (n[nsol]),
                    linestyle = '-', c = cmap(nsol))

            axs[t,h].set_xticks(rats.dust_scale.values)
            axs[t,h].set_xticklabels(ticks)
    #for i, yi in enumerate(y.T):
    #   ax.plot(x, yi, c=cmap(i))
    
        
            axs[t,h].set_title('%s winter, initialized at Ls = $%i^\circ$' % (hem, ls))
        #for i, yi in enumerate(y.T):
        #   ax.plot(x, yi, c=cmap(i))
        
    for i, ax in enumerate(fig.axes):
        if i < 2*len(tinds) - 2:
            ax.set_xticklabels([])

    fig.text(0.5,0.07,'Dust scaling',ha='center',va='center')
        #axs[0].set_ylabel('Polar tracer concentration ratio: $x$ sols / initial')
        #axs[1].set_ylabel('Polar tracer concentration ratio: $x$ sols / initial')
    fig.text(0.06,0.5,'Polar tracer concentration ratio: $x$ sols / initial',
                ha='center',va='center',rotation='vertical')

    cb = fig.colorbar(sm, ticks=n[slice(None,None,2)], ax = axs)
    cb.set_label('Sols after initialization')
    if method == 'level':
        method=method+str(level)
    if method == 'int':
        method=method+str(level[0])+'-'+str(level[1])
    
    #axs[0].legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
    #             borderaxespad=0, fontsize='large')
    #axs[1].legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
    #             borderaxespad=0, fontsize='large')

    fig.savefig(figpath+'dust_' + \
            '%s_%s_%s_%03d-%03d.png' % (tname, hem, method, tinds[0], tinds[-1]),
            bbox_inches='tight', dpi=300)

def plot_multiple_times(tname='test_tracer', isentropic=True, \
            n=np.arange(1,30), tinds=[60,90,120], hem='nh', method='level', level=300, ylim=80):
    eps = [10,15,20,25,30,35,40,45,50]
    gamma = [0.093,0.]

    colors = plt.cm.viridis(np.linspace(0,1,int(len(n))))
    matplotlib.rcParams['axes.prop_cycle'] = (
            cycler('color', colors) * \
            cycler('linestyle', ['-'])
        )

    fig,  axs  = plt.subplots(nrows=len(tinds),ncols=2, figsize = (10,3*len(tinds)), dpi = 300)
    for i, ax in enumerate(fig.axes):
        if hem == 'nh':
            ax.set_ylim([0.4,1.0])
        else:
            ax.set_ylim([1.0,4.0])
        ax.set_xlim([9,51])
        if i < 2*len(tinds) - 2:
            ax.set_xticklabels([])
        if i % 2 == 1:
            ax.set_yticklabels([])

    conc = np.full((len(eps),len(gamma),30), np.nan)
    rats = np.full((len(eps),len(gamma),30), np.nan)
    
    for t in range(len(tinds)):
        gms = []
        for i in range(len(gamma)):
            es = []
            for j in range(len(eps)):
            
        
                exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                    '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps[j], gamma[i])


                _, d = atmospy.open_files(path, exp_name, isentropic, tname=tname)
                print(exp_name)



                tind = atmospy.get_init_sol(tinds[t])

                if i == 0:
                    ls = d.mars_solar_long.isel(time=tind).values

                tslice = [tind + nsol for nsol in [0]+n]
                di  =  d.isel(time=tslice)
                di  = di.test_tracer+3

                if hem == 'nh':
                    di  =  di.where( di.lat >= ylim, drop = True)
                else:
                    di  =  di.where( di.lat <=-ylim, drop = True)


                if method == 'vert_int':

                    if not isentropic:
                        weights = di.pfull
                        di = di.weighted(weights)
                    di = di.mean(dim="pfull")

                elif method == 'int' and isentropic == False:

                    di  =  di.where( di.pfull <= level[0], drop = True)
                    di  =  di.where( di.pfull >= level[1], drop = True)
                    if not isentropic:
                        weights = di.pfull
                        di = di.weighted(weights)
                    di = di.mean(dim="pfull")


                elif method == 'int' and isentropic == True:

                    di  =  di.where( di.level <= level[1], drop = True)
                    di  =  di.where( di.level >= level[0], drop = True)


                    di  =  di.mean(dim="level")

                elif method == 'level' and level is not None:

                    if not isentropic:
                        di  =  di.sel(pfull = level, method="nearest")
                    else:
                        di  =  di.sel(level = level, method="nearest")

                else:
                    raise ValueError('Inconsistent choice of levels and integration')

                di = di.mean(dim="lon")                
                weights = np.cos(np.deg2rad(di.lat))
                di_weighted = di.weighted(weights)
                di = di_weighted.mean(dim="lat")

                di['eps'] = eps[j]
                es.append(di)
            es = xr.concat(es, dim="eps")
            es["gamma"] = gamma[i]
            gms.append(es)
    
        conc = xr.concat(gms, dim="gamma")



        conc0 = conc.isel(time=0)#.expand_dims({'time':conc.time}).transpose('eps','gamma','time')
        conc = conc.transpose('eps','gamma','time')


        rats = conc / conc0
        #cols = ['xkcd:darkgreen','xkcd:orchid','xkcd:aquamarine']
        #for nsol in range(len(n)):
        #    rs = rats.isel(time = nsol)
        #    axs[0].plot(rs.eps, rs.sel(gamma = 0.093), label = '$\gamma=%.3f$, %i sols' % (0.093, n[nsol]),
        #            linestyle = '-', )# color = cols[nsol])
        #    axs[1].plot(rs.eps, rs.sel(gamma = 0.000), label = '$\gamma=%.3f$, %i sols' % (0.000, n[nsol]),
        #            linestyle = '--',)# color = cols[nsol])


        cmap = plt.get_cmap("viridis", len(n))
        norm = matplotlib.colors.BoundaryNorm(np.arange(len(n)+1)+0.5,len(n))
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # this line may be ommitted for matplotlib >= 3.1

        for nsol in range(len(n)):
            rs = rats.isel(time = nsol)
            axs[t, 0].plot(rs.eps, rs.sel(gamma = 0.093), label = '$\gamma=%.3f$, %i sols' % (0.093, n[nsol]),
                    linestyle = '-', c = cmap(nsol))
            axs[t, 1].plot(rs.eps, rs.sel(gamma = 0.000), label = '$\gamma=%.3f$, %i sols' % (0.000, n[nsol]),
                    linestyle = '--', c = cmap(nsol))

        
        axs[t,0].set_title('$\gamma=%.3f$, initialized at Ls = $%i^\circ$' % (gamma[0], ls))
        axs[t,1].set_title('$\gamma=%.3f$, initialized at Ls = $%i^\circ$' % (gamma[1], ls))
        #for i, yi in enumerate(y.T):
        #   ax.plot(x, yi, c=cmap(i))
        

    fig.text(0.5,0.06,'Obliquity $\epsilon$ ($^\circ$)',ha='center',va='center')
        #axs[0].set_ylabel('Polar tracer concentration ratio: $x$ sols / initial')
        #axs[1].set_ylabel('Polar tracer concentration ratio: $x$ sols / initial')
    fig.text(0.06,0.5,'Polar tracer concentration ratio: $x$ sols / initial',
                ha='center',va='center',rotation='vertical')

    cb = fig.colorbar(sm, ticks=n[slice(None,None,2)], ax = axs)
    cb.set_label('Sols after initialization')
    if method == 'level':
        method=method+str(level)
    if method == 'int':
        method=method+str(level[0])+'-'+str(level[1])
    
    #axs[0].legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
    #             borderaxespad=0, fontsize='large')
    #axs[1].legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
    #             borderaxespad=0, fontsize='large')

    fig.savefig(figpath+ 'all_exp_' + \
            '%s_%s_%s_%03d-%03d.png' % (tname, hem, method, tinds[0], tinds[-1]),
            bbox_inches='tight', dpi=300)
    

def plot_multiple_times_all(tname='test_tracer', isentropic=True, \
            tinds=[60,90,120], method='level', level=300, ylim=80, ext='png'):
    eps = [10,15,20,25,30,35,40,45,50]
    gamma = [0.093,0.]

    colors = plt.cm.viridis(np.linspace(0,1,int(len(tinds))))
    matplotlib.rcParams['axes.prop_cycle'] = (
            cycler('linestyle', ['-','--'])
        )
    exps = ['curr-ecc','0-ecc','dust',]#'attribution']
    fig,  axs  = plt.subplots(nrows=len(exps)-1,ncols=2, figsize = (10,10), dpi = 300)
    

    for i in range(len(exps)):
        exp = exps[i]
        filepath = path + 'mars_analysis/tracer_concs/%s_tracer_conc_' % exp
        exp_names, titles, _, _ = atmospy.get_exps(exp)
        
        if exp == 'curr-ecc':
            exp = '$\gamma = 0.093$'
            lnstl = '-'
            col = 'xkcd:crimson'
        elif exp == '0-ecc':
            exp = '$\gamma = 0.000$'
            lnstl = '--'
            col = 'xkcd:teal'
        elif exp == 'dust':
            exp = 'Dust Scale'
            lnstl = '-'
            col = 'xkcd:darkgreen'
        elif exp == 'attribution':
            exp = 'Attribution'
            lnstl = '-'
            col = 'k'

        cmap = plt.get_cmap("viridis", len(tinds))
        norm = matplotlib.colors.BoundaryNorm(np.arange(len(tinds)+1)+0.5,len(tinds))
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # this line may be ommitted for matplotlib >= 3.1

        nh = []
        sh = []
        for t in range(len(tinds)):
            en = []
            es = []
            tind = atmospy.get_init_sol(tinds[t])
            for j in range(len(exp_names)):

                exp_name = exp_names[j]
            
                d = xr.open_dataset(path+exp_name+'/atmos_isentropic.nc', decode_times=False)
            
                di  =  d.isel(time=[tind,tind+29,tind+360,tind+389])
                ls = di.mars_solar_long
                
                di  = di.test_tracer
                

                if method == 'int' and isentropic == False:
                    di  =  di.where( di.pfull <= level[0], drop = True)
                    di  =  di.where( di.pfull >= level[1], drop = True)
                    if not isentropic:
                        weights = di.pfull
                        di = di.weighted(weights)
                    di = di.mean(dim="pfull")

                elif method == 'int' and isentropic == True:
                    di  =  di.where( di.level <= level[1], drop = True)
                    di  =  di.where( di.level >= level[0], drop = True)
                    di  =  di.mean(dim="level")

                elif method == 'level' and level is not None:
                    if not isentropic:
                        di  =  di.sel(pfull = level, method="nearest")
                    else:
                        di  =  di.sel(level = level, method="nearest")

                else:
                    raise ValueError('Inconsistent choice of levels and integration')

                di = di.mean(dim="lon",skipna=True)

                dn = di.where(di.lat >= ylim, drop = True) + 3
                dn_weighted = dn.weighted(np.cos(np.deg2rad(dn.lat)))
                dn = dn_weighted.mean(dim="lat").isel(time=[0,1])
                
                ds = - di.where(di.lat <=-ylim, drop = True) + 3
                ds_weighted = ds.weighted(np.cos(np.deg2rad(ds.lat)))
                ds = ds_weighted.mean(dim="lat").isel(time=[-2,-1])
                #print(ds.values)
                
                
                dn['exp'] = j
                
                ds['exp'] = j
                
                en.append(dn)
                es.append(ds)
            
            #if exp == 'Attribution':
            #    print(en[-1].values)
            en = xr.concat(en, dim="exp")
            #if exp == 'Attribution':
            #    print(en.values)
            en['hem']=0
            es = xr.concat(es, dim="exp")
            es['hem']=1
            
            hems = [en, es]

            dat = xr.concat(hems, dim="hem")
            dat['mars_solar_long'] = ls
            dat.to_netcdf(filepath + '%i.nc' % (tind))
    
            for k in [0,1]:
                conc = hems[k]
                
                rats = conc.isel(time=-1) / conc.isel(time= 0)
                
                if exp == '$\gamma = 0.093$' or exp == '$\gamma = 0.000$':
                    ax = axs[0, k]
                elif exp == 'Dust Scale':
                    ax = axs[1, k]
                else:
                    ax = axs[2, k]
            
                rs = rats#.isel(time = -1)
                
                if k == 0:
                    nh.append(rs)
                    labl = 'Init L$_s$: %i$^\circ$' % (d.mars_solar_long.isel(time=tind).values)
                else:
                    sh.append(rs)
                    labl = 'Init L$_s$: %i$^\circ$' % (d.mars_solar_long.isel(time=tind+360).values)
                
                if exp == '$\gamma = 0.000$':
                    labl = None
                ax.plot(rs.exp, rs*100, label = labl,linewidth=0.8,
                    linestyle = lnstl, c = cmap(t),alpha=0.6, zorder=0)
                
            #
            #    ls.append()
        
        for k in [0,1]:
            if exp == '$\gamma = 0.093$' or exp == '$\gamma = 0.000$':
                ax = axs[0, k]
            elif exp == 'Dust Scale':
                ax = axs[1, k]
            else:
                ax = axs[2, k]
            
            if k == 0:
                rs = nh
            else:
                rs = sh
            if exp == '$\gamma = 0.000$':
                labl = None
            else:
                labl = 'Winter average'

            rs = xr.concat(rs,dim="time")
            rs = rs.mean(dim="time")
            ax.plot(rs.exp, rs*100, label = labl, linewidth=1.7,
                    linestyle = lnstl, c = 'k',alpha=1,zorder=1)
            #for b in range(len(rs[0])):
            #    for a in range(len(rs)):
            #        r += rs[a][b]

    axs[0,0].set_title('NH')
    axs[0,1].set_title('SH')
    m1 = []
    m2 = []
    for i, ax in enumerate(fig.axes):
        ax.text(-0.01, 1.03, string.ascii_lowercase[i]+')', transform=ax.transAxes)
        m1.append(ax.get_ylim()[0])
        m2.append(ax.get_ylim()[1])
    
    for i, ax in enumerate(fig.axes):
        ax.set_ylim([np.min(m1),np.max(m2)])
        ax.legend(loc = 'lower left')

    for ax in [axs[0,0], axs[0,1]]:
        ax.set_xticks(np.arange(len(eps)))
        ax.set_xticklabels(eps)
        ax.set_xlabel('Obliquity $\epsilon$ ($^\circ$)')
        #for i, yi in enumerate(y.T):
        #   ax.plot(x, yi, c=cmap(i))
        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.plot([],[],c='k',linestyle='-' ,label='$\gamma=0.093$')
        ax2.plot([],[],c='k',linestyle='--',label='$\gamma=0.000$')
        ax2.legend()
    for ax in [axs[1,0], axs[1,1]]:
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(['1/2','1','2','4'])
        ax.set_xlabel('Dust Scale ($\\tau$)')
    if len(exps)==4:
        for ax in [axs[2,0], axs[2,1]]:
            ax.set_xticks(np.arange(len(titles)))
            ax.set_xticklabels(titles)
            ax.set_xlabel('Attribution')

        #axs[0].set_ylabel('Polar tracer concentration ratio: $x$ sols / initial')
        #axs[1].set_ylabel('Polar tracer concentration ratio: $x$ sols / initial')
    fig.text(0.06,0.5,'Percentage of tracer at pole after 30 sols ($c_{{pole}_{30}}/c_{{pole}_0}$, %)',
                ha='center',va='center',rotation='vertical',fontsize='large')

    

    #cb = fig.colorbar(sm, ax = axs, ticks = np.arange(len(tinds)+1)+1/2, boundaries = np.arange(len(tinds)+1))
    #cb.ax.set_yticklabels(ls)
    #cb.set_label('Initialization L$_s$')


    if method == 'level':
        method=method+str(level)
    if method == 'int':
        method=method+str(level[0])+'-'+str(level[1])
    
    #axs[0].legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
    #             borderaxespad=0, fontsize='large')
    #axs[1].legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
    #             borderaxespad=0, fontsize='large')

    fig.savefig(figpath + \
            '%s_%s_%s_%03d-%03d_%i.%s' % (tname, hem, method, tinds[0], tinds[-1],ylim,ext),
            bbox_inches='tight', dpi=300)
    


def plot_multiple_times_all_from_data(tname = 'test_tracer', isentropic=True, \
            tinds=[60,90,120], method='level', level=300, ylim=80, ext='png'):
    eps = [10,15,20,25,30,35,40,45,50]
    gamma = [0.093,0.]

    colors = plt.cm.viridis(np.linspace(0,1,int(len(tinds))))
    matplotlib.rcParams['axes.prop_cycle'] = (
            cycler('linestyle', ['-','--'])
        )
    exps = ['curr-ecc','0-ecc','dust',]#'attribution']
    fig,  axs  = plt.subplots(nrows=len(exps)-1,ncols=2, figsize = (10,10), dpi = 300)
    

    for i in range(len(exps)):
        exp = exps[i]
        filepath = path + 'mars_analysis/tracer_concs/%s_tracer_conc_' % (exp)
        exp_names, titles, _, _ = atmospy.get_exps(exp)
        
        if exp == 'curr-ecc':
            exp = '$\gamma = 0.093$'
            lnstl = '-'
            col = 'xkcd:crimson'
        elif exp == '0-ecc':
            exp = '$\gamma = 0.000$'
            lnstl = '--'
            col = 'xkcd:teal'
        elif exp == 'dust':
            exp = 'Dust Scale'
            lnstl = '-'
            col = 'xkcd:darkgreen'
        elif exp == 'attribution':
            exp = 'Attribution'
            lnstl = '-'
            col = 'k'

        cmap = plt.get_cmap("viridis", len(tinds))
        norm = matplotlib.colors.BoundaryNorm(np.arange(len(tinds)+1)+0.5,len(tinds))
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # this line may be ommitted for matplotlib >= 3.1

        ls = []
        nh = []
        sh = []
        for t in range(len(tinds)):
            tind = atmospy.get_init_sol(tinds[t])
            
            hems = xr.open_dataset(filepath+'%i.nc' % tind, decode_times=False)

    
            for k in [0,1]:
                conc = hems.test_tracer.isel(hem=k)
                
                rats = conc.isel(time=2*k+1) / conc.isel(time=2*k)
                
                if exp == '$\gamma = 0.093$' or exp == '$\gamma = 0.000$':
                    ax = axs[0, k]
                elif exp == 'Dust Scale':
                    ax = axs[1, k]
                else:
                    ax = axs[2, k]
            
                rs = rats#.isel(time = -1)
                
                if k == 0:
                    nh.append(rs)
                    labl = 'Init L$_s$: %i$^\circ$' % (hems.mars_solar_long.isel(time=0).values)
                else:
                    sh.append(rs)
                    labl = 'Init L$_s$: %i$^\circ$' % (hems.mars_solar_long.isel(time=2).values)
                
                if exp == '$\gamma = 0.000$':
                    labl = None
                ax.plot(rs.exp, rs*100, label = labl,linewidth=0.8,
                    linestyle = lnstl, c = cmap(t),alpha=0.6, zorder=0)
                
            #
            #    ls.append()
        
        for k in [0,1]:
            if exp == '$\gamma = 0.093$' or exp == '$\gamma = 0.000$':
                ax = axs[0, k]
            elif exp == 'Dust Scale':
                ax = axs[1, k]
            else:
                ax = axs[2, k]
            
            if k == 0:
                rs = nh
            else:
                rs = sh
            if exp == '$\gamma = 0.000$':
                labl = None
            else:
                labl = 'Winter average'

            rs = xr.concat(rs,dim="time")
            rs = rs.mean(dim="time")
            ax.plot(rs.exp, rs*100, label = labl, linewidth=1.7,
                    linestyle = lnstl, c = 'k',alpha=1,zorder=1)
            #for b in range(len(rs[0])):
            #    for a in range(len(rs)):
            #        r += rs[a][b]

    axs[0,0].set_title('NH')
    axs[0,1].set_title('SH')
    m1 = []
    m2 = []
    for i, ax in enumerate(fig.axes):
        ax.text(-0.01, 1.03, string.ascii_lowercase[i]+')', transform=ax.transAxes)
        m1.append(ax.get_ylim()[0])
        m2.append(ax.get_ylim()[1])
    
    for i, ax in enumerate(fig.axes):
        ax.set_ylim([np.min(m1),np.max(m2)])
        ax.legend(loc = 'lower left')

    for ax in [axs[0,0], axs[0,1]]:
        ax.set_xticks(np.arange(len(eps)))
        ax.set_xticklabels(eps)
        ax.set_xlabel('Obliquity $\epsilon$ ($^\circ$)')
        #for i, yi in enumerate(y.T):
        #   ax.plot(x, yi, c=cmap(i))
        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.plot([],[],c='k',linestyle='-' ,label='$\gamma=0.093$')
        ax2.plot([],[],c='k',linestyle='--',label='$\gamma=0.000$')
        ax2.legend()
    for ax in [axs[1,0], axs[1,1]]:
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(['1/2','1','2','4'])
        ax.set_xlabel('Dust Scale ($\\tau$)')
    if len(exps)==4:
        for ax in [axs[2,0], axs[2,1]]:
            ax.set_xticks(np.arange(len(titles)))
            ax.set_xticklabels(titles)
            ax.set_xlabel('Attribution')

        #axs[0].set_ylabel('Polar tracer concentration ratio: $x$ sols / initial')
        #axs[1].set_ylabel('Polar tracer concentration ratio: $x$ sols / initial')
    fig.text(0.06,0.5,'Percentage of tracer at pole after 30 sols ($c_{{pole}_{30}}/c_{{pole}_0}$, %)',
                ha='center',va='center',rotation='vertical',fontsize='large')

    

    #cb = fig.colorbar(sm, ax = axs, ticks = np.arange(len(tinds)+1)+1/2, boundaries = np.arange(len(tinds)+1))
    #cb.ax.set_yticklabels(ls)
    #cb.set_label('Initialization L$_s$')


    if method == 'level':
        method=method+str(level)
    if method == 'int':
        method=method+str(level[0])+'-'+str(level[1])
    
    #axs[0].legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
    #             borderaxespad=0, fontsize='large')
    #axs[1].legend(loc='center left', bbox_to_anchor=(1.05,0.5,),
    #             borderaxespad=0, fontsize='large')

    fig.savefig(figpath + \
            '%s_%s_%s_%03d-%03d_%i.%s' % (tname, hem, method, tinds[0], tinds[-1],ylim,ext),
            bbox_inches='tight', dpi=300)
    

#%%
if __name__ == "__main__":
    tind = 60
    isentropic=True
    method="level"
    level = 300
    hem = "nh"
    ylim = 75

    tinds = [60,90,120,150]
    savedata = False
    if savedata:
        plot_multiple_times_all(
        level=300,
        ylim=ylim, tinds = tinds,ext='pdf',
        )
    else:
        plot_multiple_times_all_from_data(
        level=300,
        ylim=ylim, tinds = tinds,ext='pdf',
        )

# %%
