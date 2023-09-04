# %%
import xarray as xr
import numpy as np
import sys, os
import math

sys.path.append('../')

import atmospy

from plot_HC_anomaly import get_HC_edge
from plot_PV_anomaly import get_PV_lats_isentropic
import string

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import (cm, colors, cycler)
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


path = '/user/work/xz19136/Isca_data/'
theta, center, radius, verts, circle = atmospy.stereo_plot()
theta0 = 200.
kappa = 1/4.0

figpath = '/user/home/xz19136/Figures/mars_analysis/'

if plt.rcParams["text.usetex"]:
    fmt = r'%r \%'
else:
    fmt = '%r'

def get_lats_all(exps=['curr-ecc','0-ecc','dust'],level=300):
    '''
    Plot effective diffusivity evolution at a given point,
    in order to understand strength of the transport barrier and mixing
    within the vortex.
    '''
    lats = []
    
    for i in range(len(exps)):
        exp = exps[i]
        exp_names, titles, _, _ = atmospy.get_exps(exp)
        a_s = np.zeros((len(exp_names),2,4))
        t = np.zeros((len(exp_names),2))
        
        
        for j in range(len(exp_names)):

            exp_name = exp_names[j]
            print(exp_name)
            
            
            ds = xr.open_dataset(path+exp_name+'/psi.nc', decode_times=False)
            if exp != '0-ecc':
                tindn = (np.abs(ds.mars_solar_long.values-260)).argmin(axis=0)
                        #i = np.take_along_axis(ds, np.abs(ds.mars_solar_long.values-270).argmin(axis=0), axis=0)
                tinds = np.abs(ds.mars_solar_long.values-90).argmin(axis=0)
            tslicen = slice(tindn-15,tindn+15)
            tslices = slice(tinds-15,tinds+15)
            
            psi0n = get_HC_edge(ds.where(ds.lat>0,drop=True).isel(time=tslicen))
            psi0s = get_HC_edge(ds.where(ds.lat<0,drop=True).isel(time=tslices).sortby('lat',ascending=False))
            
            a_s[j,0,0] = np.nanmean(psi0n)
            a_s[j,1,0] = np.nanmean(psi0s)

            ds = xr.open_dataset(path+exp_name+'/EDJ.nc', decode_times=False)
            
            phi_n = ds.phi_n.squeeze()
            phi_s = ds.phi_s.squeeze()
            
            a_s[j,0,1] = phi_n.isel(time=tslicen).mean(dim="time").values
            a_s[j,1,1] = phi_s.isel(time=tslices).mean(dim="time").values

            ds = xr.open_dataset(path+exp_name+'/atmos_isentropic.nc', decode_times=False)

            ds = ds[["PV","mars_solar_long"]]
            if level != 'int':
                ds = ds.sel(level=level,method="nearest")
            else:
                ds = ds.sel(level=300,method="nearest")
            ds = ds.mean(dim="lon")

            phiPV_n, _ = get_PV_lats_isentropic(ds.where(ds.lat > 0, drop=True).isel(time=tslicen),hem='nh')
            phiPV_s, _ = get_PV_lats_isentropic(ds.where(ds.lat < 0, drop=True).isel(time=tslices),hem='sh')

            a_s[j,0,2] = np.nanmean(phiPV_n)
            a_s[j,1,2] = np.nanmean(phiPV_s)

            ds = xr.open_dataset(path+exp_name+'/atmos.nc', decode_times=False)
            ds = ds.sel(pfull=0.5,method="nearest").mean(dim="lon")

            t_n = ds.temp.where(ds.lat> 75,drop=True).isel(time=tslicen).mean(dim="time")
            t_s = ds.temp.where(ds.lat<-75,drop=True).isel(time=tslices).mean(dim="time")

            tn_weighted = t_n.weighted(np.cos(np.deg2rad(t_n.lat)))
            t_n = tn_weighted.mean(dim="lat").values
            ts_weighted = t_s.weighted(np.cos(np.deg2rad(t_s.lat)))
            t_s = ts_weighted.mean(dim="lat").values

            t[j,0] = t_n
            t[j,1] = t_s            

            if level != 'int':
                ds = xr.open_dataset(path+exp_name+'/keff_isentropic_test_tracer.nc', decode_times=False)
                ds = ds.nkeff.sel(level=level,method="nearest")
            else:
                ds = xr.open_dataset(path+exp_name+'/keff_test_tracer.nc', decode_times=False)
                ds = ds.where(ds.level<=4.5,drop=True).where(ds.level>0.05,drop=True)
                ds = ds.nkeff.weighted(ds.level)
                ds = ds.mean(dim='level')

            keff_n = ds.where(ds.new > 40, drop=True).isel(time=tslicen)
            dn_weighted = keff_n.weighted(np.cos(np.deg2rad(keff_n.new)))
            keff_n = dn_weighted.mean(dim="new")
            keff_s = ds.where(ds.new <-40, drop=True).isel(time=tslices)
            dn_weighted = keff_s.weighted(np.cos(np.deg2rad(keff_s.new)))
            keff_s = dn_weighted.mean(dim="new")

            a_s[j,0,3] = np.nanmean(np.log(keff_n))
            a_s[j,1,3] = np.nanmean(np.log(keff_s))

            dat = xr.DataArray(a_s,
                coords = {'exp_name':exp_names, 'hem':[0,1], 'phi':['HC','u','PV','keff']},
                dims = ['exp_name','hem','phi'],
                )
            dat.to_netcdf(path+'mars_analysis/summary_of_lats/%s_summary_of_lats_%s.nc' % (exp, level))

            dat = xr.DataArray(t,
                coords = {'exp_name':exp_names, 'hem':[0,1]},
                dims = ['exp_name','hem'],
                )
            dat.to_netcdf(path+'mars_analysis/summary_of_lats/%s_summary_of_lats_%s_temp.nc' % (exp, level))
            
        lats.append(a_s)
    return lats

def plot_summary_of_lats(lats, exps = ['curr-ecc','0-ecc','dust']):
    fig, axs = plt.subplots(nrows=len(lats),ncols=2, figsize = (15,3.5*len(lats)),dpi=300)
    
    boundaries, cmap, norm = atmospy.new_cmap(
        [30,90], extend='neither',override=True, i=40)
        
    for i in range(len(exps)):
        lat = lats[i]
        exp_names, titles, _, _ = atmospy.get_exps(exps[i])

        if exps[i] == 'curr-ecc':
            exp = '$\gamma = 0.093$'
            xticklabs = np.arange(10,55,5)
            xlabel = 'Obliquity'
        elif exps[i] == '0-ecc':
            exp = '$\gamma = 0.000$'
            xticklabs = np.arange(10,55,5)
            xlabel = 'Obliquity'
        elif exps[i] == 'dust':
            exp = 'Dust Scale'
            xticklabs = [1/2,1,2,4]
            xlabel = 'Dust Scale'
        elif exps[i] == 'attribution':
            exp = 'Attribution'
        elif exps[i] == 'high_res_dust':
            exp = 'Dust Vertical Res'
            xticklabs = [1/2,1,2,4]
            xlabel = 'Dust Scale'

        a0 = axs[i,0].pcolor(lat[:,0,:-1].transpose(), cmap = cmap, norm = norm,)
        axs[i,1].pcolor(np.abs(lat[:,1,:-1].transpose()), cmap = cmap, norm = norm,)

        axs[i,0].text(
                    -0.16, 0.5, exp,
                    ha='right',
                    va='center',
                    transform=axs[i,0].transAxes,
                    rotation='vertical',
                    fontsize='large',
            )
        for j in [0,1]:
            ax = axs[i,j]
            xticks = [i+0.5 for i in range(len(exp_names))]
            yticks = [0.5,1.5,2.5]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticks(xticks,minor=True)
            ax.set_yticks(yticks,minor=True)
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.set_xticklabels(xticklabs,minor=True)
            ax.set_yticklabels(['$\phi_{umax}$','$\phi_{HC}$','$\phi_{PV}$'],minor=True)
            ax.set_xlabel(xlabel)
    axs[0,0].set_title('NH')
    axs[0,1].set_title('SH')

    cb = fig.colorbar(a0, ax = axs,
        #cmap = cmap, norm = norm,
        label = 'Latitude ($^\circ$ N/S)',
        ticks = boundaries[slice(None,None,4)], boundaries = boundaries)

    fig.savefig(figpath + 'summary_of_lats.png', dpi=300,
                bbox_inches='tight')

def plot_summary_of_lats_line(lats, exps = ['curr-ecc','0-ecc','dust'],
                              level=300,ext='png'):
    nrows = len(lats)-1
    if exps.count('MY28'):
        nrows -= 1
    if exps.count('vert_dust_only'):
        nrows -= 1
    if (exps.count('long-dust') or exps.count('long-dust_only')) and exps.count('dust'):
        nrows -= 1
    fig, axs = plt.subplots(nrows=nrows,ncols=2, figsize = (10,3.4*nrows),dpi=300)
    
    colors = plt.cm.viridis(np.linspace(0,1,4))
    matplotlib.rcParams['axes.prop_cycle'] = (
            cycler('color', colors) * \
            cycler('linestyle', ['-','--'])
        )

    for i, ax in enumerate(fig.axes):
        ax.text(-0.01, 1.03, string.ascii_lowercase[i+2]+')', transform=ax.transAxes)
    
    cmap = plt.get_cmap("viridis", 4)
    norm = matplotlib.colors.BoundaryNorm(np.arange(5)+0.5,4)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # this line may be ommitted for matplotlib >= 3.1

    a   = []
    std = []

    for i in range(len(exps)):
        lat = lats[i]
        
        exp_names, _, _, _ = atmospy.get_exps(exps[i])
        marker = None
        if exps[i] == 'curr-ecc':
            xticklabs = np.arange(10,55,5)
            xlabel = 'Obliquity $\epsilon$ ($^\circ$)'
            #axs[i,0].text(
        #            -0.16, 0.5, exp,
        #            ha='right',
        #            va='center',
        #            transform=axs[i,0].transAxes,
        #            rotation='vertical',
        #            fontsize='large',
        #    )
            lnstl = '-'
            #titles = '$\gamma=0.093$'
            labs = ['$\phi_{\\rm HC}$','$\phi_{u_{\\rm max}}$','$\phi_{\\rm PV}$','$\kappa_{{\\rm eff}_{300}}$']
            k = i
        elif exps[i] == '0-ecc':
            xticklabs = np.arange(10,55,5)
            xlabel = 'Obliquity $\epsilon$ ($^\circ$)'
            k = i-1
            lnstl = '--'
            #titles = '$\gamma=0.000$'
            labs=[None,None,None,None]
        elif exps[i] == 'dust':
            xticklabs = np.array([1/2,1,2,4,8])
            xlabel = 'Dust Scale ($\lambda}$)'
            k = i-1
            lnstl = '-'
            labs = ['$\phi_{\\rm HC}$','$\phi_{u_{\\rm max}}$','$\phi_{\\rm PV}$','$\kappa_{{\\rm eff}_{300}}$']
        elif exps[i] == 'vert_dust_only':
            xticklabs = np.array([1/2,1,2,4,8])
            k = i-3
            lnstl = '--'
            labs = [None, None, None, None,None]
        elif exps[i] == 'long-dust_only':
            xticklabs = np.array([1/2,1,2,4,8])
            k = i-2
            lnstl = '-.'
            labs = [None, None, None, None,None]
        elif exps[i] == 'MY28':
            k = i-3
            marker = 'o'
        elif exps[i] == 'attribution':
            exp = 'Attribution'
            labs = ['$\phi_{\\rm HC}$','$\phi_{u_{\\rm max}}$','$\phi_{\\rm PV}$','$\kappa_{{\\rm eff}_{300}}$']

        if exps[i] != 'MY28':
            axs[k,0].plot(range(len(exp_names)),lat[:,0,0],
                        c = cmap(1),linestyle=lnstl,label=labs[0],marker=marker)
            axs[k,0].plot(range(len(exp_names)),lat[:,0,1],
                        c = cmap(2),linestyle=lnstl,label=labs[1],marker=marker)
            axs[k,0].plot(range(len(exp_names)),lat[:,0,2],
                        c = cmap(3),linestyle=lnstl,label=labs[2],marker=marker)

            axs[k,1].plot(range(len(exp_names)),lat[:,1,0],
                        c = cmap(1),linestyle=lnstl,label=labs[0],marker=marker)
            axs[k,1].plot(range(len(exp_names)),lat[:,1,1],
                        c = cmap(2),linestyle=lnstl,label=labs[1],marker=marker)
            axs[k,1].plot(range(len(exp_names)),lat[:,1,2],
                        c = cmap(3),linestyle=lnstl,label=labs[2],marker=marker)

        
            diff = np.diff(np.abs(lat),axis=0)
            xticks = xticklabs.reshape(-1,1).repeat(diff.shape[1], axis=1)
            xticks = xticks.reshape(xticks.shape[0],-1,1).repeat(diff.shape[2], axis=2)
            a.append(np.mean(diff/np.diff(xticks,axis=0),axis=0))
            std.append(np.std(diff/np.diff(xticks,axis=0), axis=0))
        
        else:
            axs[k,0].plot(1,lat[0,0],
                        c = cmap(1),linestyle=lnstl,marker=marker)
            axs[k,0].plot(1,lat[0,1],
                        c = cmap(2),linestyle=lnstl,marker=marker)
            axs[k,0].plot(1,lat[0,2],
                        c = cmap(3),linestyle=lnstl,marker=marker)

            axs[k,1].plot(1,lat[1,0],
                        c = cmap(1),linestyle=lnstl,label=labs[0],marker=marker)
            axs[k,1].plot(1,lat[1,1],
                        c = cmap(2),linestyle=lnstl,label=labs[1],marker=marker)
            axs[k,1].plot(1,lat[1,2],
                        c = cmap(3),linestyle=lnstl,label=labs[2],marker=marker)
            a.append(None)
            std.append(None)
        

        for j in [0,1]:
            ax = axs[k,j]
            xticks = [l for l in range(len(exp_names))]
            #ax.set_xticks([])
            if exps[i] != 'MY28':
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabs)
                ax.set_xlabel(xlabel)
                ax1 = ax.twinx()
            
                ax1.plot(range(len(exp_names)),lat[:,j,3],
                    c = cmap(0),linestyle=lnstl,label=labs[3])
            else:
                ax1 = ax.twinx()
                ax1.plot(1,lat[j,3],
                    c = cmap(0),linestyle=lnstl,label=labs[3],marker='o')
            #ax1.set_xticks([])
            ax1.set_ylim([0,3.5])
            #ax1.set_yticks([0.5,1.0,1.5,2.0])
            ax1.spines['right'].set_color(cmap(0))
            ax1.tick_params(axis='y',colors=cmap(0))
            if j == 1:
                ax1.set_ylabel('Normalized effective diffusivity', color=cmap(0))
                ax1.tick_params(axis='y', labelcolor=cmap(0))
                ax.set_yticklabels([])
            else:
                ax1.set_yticklabels([])
            
            ax.plot([],[],c = cmap(0),linestyle=lnstl,label=labs[3])
            #if exps[i] != '0-ecc':
            #    ax1.legend()

    axs[0,0].set_title('NH')
    axs[0,1].set_title('SH')
    axs[0,0].set_ylim([ 40, 90])
    axs[0,1].set_ylim([-40,-90])
    axs[1,0].set_ylim([ 40, 90])
    axs[1,1].set_ylim([-40,-90])

    #axs[0,0].legend(loc='upper left')
    #axs[0,1].legend(loc='upper left')
    #axs[1,0].legend(loc='upper left')
    axs[0,1].legend(loc='lower right')

    axs[0,0].set_ylabel('latitude ($^\circ$N/S)')
    #axs[0,0].legend()
    axs[1,0].set_ylabel('latitude ($^\circ$N/S)')

    ax2 = axs[0,1].twinx()
    ax2.set_yticks([])
    ax2.plot([],[],c = cmap(0),linestyle='-',label='$\gamma=0.093$')
    ax2.plot([],[],c = cmap(0),linestyle='--',label='$\gamma=0.000$')
    ax2.legend(loc='center left',fontsize='small')

    ax2 = axs[1,1].twinx()
    ax2.set_yticks([])
    ax2.plot([],[],c = cmap(0),linestyle='-',label='Dust Scale')
    ax2.plot([],[],c = cmap(0),linestyle='--',label='50 vert levels')
    ax2.plot([],[],c = cmap(0),linestyle='-.',label='Longitudinal Dust')
    ax2.legend(loc='upper left',fontsize='small')


    fig.tight_layout()
    if level != 'int':
        fig.savefig(figpath + 'line_summary_of_lats_%i_log.%s' % (level, ext), dpi=300,
                bbox_inches='tight')
    else:
        fig.savefig(figpath + 'line_summary_of_lats_%s_log.%s' % (level, ext), dpi=300,
                bbox_inches='tight')

    return a, std


def plot_rate_of_change(a, std, exps = ['curr-ecc','0-ecc','dust'],level=300,ext='png'):
    ncols = len(exps)
    if exps.count('curr-ecc') and exps.count('0-ecc'):
        ncols -= 1
    if (exps.count('high_res_dust') or exps.count('vert_dust_only')) and exps.count('dust'):
        ncols -= 1
    if (exps.count('long-dust') or exps.count('long-dust_only')) and exps.count('dust'):
        ncols -= 1
    if exps.count('MY28'):
        ncols -= 1
    
    fig, axs = plt.subplots(nrows=1,ncols=ncols,
                            figsize = (5*ncols,4.2),dpi=300)
    for i, ax in enumerate(fig.axes):
        ax.text(-0.01, 1.03, string.ascii_lowercase[i]+')', transform=ax.transAxes,fontsize='small')
    
    axs[0].annotate('', xy = (-0.25,1), xycoords='axes fraction', xytext=(-0.25,0.34),
        arrowprops=dict(arrowstyle='->'),
    )

    axs[0].text(
        -0.3, 0.7, 'poleward',
        ha='right',
        va='center',
        transform=axs[0].transAxes,
        rotation='vertical',
        fontsize='small',
    )

    axs[0].annotate('', xy = (-0.25,0), xycoords='axes fraction', xytext=(-0.25,0.24),
        arrowprops=dict(arrowstyle='->'),
    )

    axs[0].text(
        -0.3, 0.13, 'equatorward',
        ha='right',
        va='center',
        transform=axs[0].transAxes,
        rotation='vertical',
        fontsize='small',
    )
    axs[1].annotate('', xy = (1.2,1), xycoords='axes fraction', xytext=(1.2,0),
        arrowprops=dict(arrowstyle='->'),
    )

    axs[1].text(
        1.27, 0.5, 'increasing $\kappa_{{\\rm eff}_{300}}$',
        ha='right',
        va='center',
        transform=axs[1].transAxes,
        rotation='vertical',
        #fontsize='large',
    )
    offs = [-0.13,0.13,-0.26,0.0,0.26,0.13]
    k = 0
    l = 0
    denom = ['$^\circ$obliquity','$\lambda$']
    markers = ['o','s','d','^','v','x',]
    labs = ['$\phi_{\\rm HC}$','$\phi_{u_{\\rm max}}$','$\phi_{\\rm PV}$','$\kappa_{{\\rm eff}_{300}}$']
    for i in range(len(a)):
        if exps[i] == 'curr-ecc':
            xlabel = '$\gamma=0.093$'
        elif exps[i] == '0-ecc':
            xlabel = '$\gamma=0.000$'
        elif exps[i] == 'dust':
            xlabel = 'Dust Scale ($\lambda}$)'
            k +=1
        elif exps[i] == 'high_res_dust' or exps[i] == 'vert_dust_only':
            xlabel = '50 vert levels'
        elif exps[i] == 'long-dust_only':
            xlabel = 'longitudinal dust'
        elif exps[i] == 'MY28':
            continue
        elif exps[i] == 'attribution':
            xlabel = 'Attribution'
            k+=1

        mark = markers[i]
        l+=1
        ax = axs[k]
        #a[i][1,-1] = -a[i][1,-1]     # because other metrics are moving poleward/equatorward
        ax.errorbar(np.arange(a[i].shape[1]-1)+offs[i]-0.04, a[i][0,:-1],yerr=std[i][0,:-1],
            color='xkcd:aquamarine', capsize=2,zorder=1,
            marker=mark, ms=6,alpha=0.7,linestyle='',)
        ax.scatter([],[],color='xkcd:aquamarine',
            marker=mark, s=35,label='%s NH' % xlabel,alpha=0.7,)
        #ax.errorbar(np.arange(a[i].shape[1]-1)+offs[i], a[i][0,:-1],yerr=std[i][0,:-1],
        #    color='xkcd:aquamarine',alpha=0.5,linestyle='')

        ax.errorbar(np.arange(a[i].shape[1]-1)+offs[i]+0.04, a[i][1,:-1],yerr=std[i][1,:-1],
            color='xkcd:darkblue', capsize=2,zorder=0,
            marker=mark, ms=6,alpha=0.7,linestyle='',)
        ax.scatter([],[],color='xkcd:darkblue',
            marker=mark, s=30,label='%s SH' % xlabel,alpha=0.7,)
        
    l=0
    loc = ['upper center', 'lower center']
    for i, ax in enumerate(fig.axes):
        ax.set_xticks(np.arange(a[i].shape[1]))
        ax.set_xticklabels(labs,fontsize='large')
        ax.tick_params(axis='y',labelsize='small')
        ax.set_ylabel('$\Delta(^\circ$N/S) / $\Delta$(%s)' % denom[i])
        ax.hlines(0,0,a[i].shape[1]-1.75,colors='gray',linestyles='--',alpha=0.5,zorder=0)
        ax.vlines(a[i].shape[1]-1.5,ax.get_ylim()[0],ax.get_ylim()[1],
            colors='k',linestyles='-',alpha=0.5,linewidths=0.8)
        ax.legend(fontsize='xx-small',loc=loc[i],)# bbox_to_anchor=(1.05,0.75,),
                 #borderaxespad=0)

        ax1 = ax.twinx()
        ax1.tick_params(axis='y',labelsize='small')
        if i > 0 and (exps.count('long-dust_only') or exps.count('vert_dust_only')):
            k = [i+1,i+2]
            if exps.count('vert_dust_only') and exps.count('long-dust_only'):
                k.append(i+3)
        elif i > 0:
            k = [i+1]
        else:
            k = [i,i+1]

        for j in k:
            mark = markers[j]
            
            ax1.errorbar(a[j].shape[1]-1+offs[j]-0.04, a[j][0,-1],yerr=std[j][0,-1],
                color='xkcd:aquamarine', capsize=2,zorder=2,
                marker=mark, ms=6,label='%s NH' % xlabel,alpha=0.7,linestyle='',)
            
            
            ax1.errorbar(a[j].shape[1]-1+offs[j]+0.04, a[j][1,-1],yerr=std[j][1,-1],
                color='xkcd:darkblue', capsize=2.,zorder=1,
                marker=mark, ms=6,label='%s SH' % xlabel,alpha=0.7,linestyle='',)

        #ax1.set_ylim([ax1.get_ylim()[0],ax1.get_ylim()[1]*1.3])
        
        ax1.set_ylabel('$\Delta(\kappa_{{\\rm eff}_{300}})$ / $\Delta$(%s)' % denom[i])
    fig.tight_layout()
    if level != 'int':
        fig.savefig(figpath + 'lat_rates_of_change_%i_log.%s' % (level, ext), dpi=300,
                bbox_inches='tight')
    else:
        fig.savefig(figpath + 'lat_rates_of_change_%s_log.%s' % (level, ext), dpi=300,
                bbox_inches='tight')

#%%
if __name__ == "__main__":

    eps = np.arange(10,55,5)
    gamma = [0.093,0.00]
    exps = ['curr-ecc','0-ecc','dust','long-dust_only','vert_dust_only']#,'long-dust_only']
    level = 300
    savedata = True
    
    if savedata:
        lats = get_lats_all(exps=exps,level=level)
        a, std = plot_summary_of_lats_line(lats, exps = exps,level=level,ext='pdf')
        plot_rate_of_change(a,std,exps=exps,level=level,ext='pdf')
    else:
        lats = []
        for exp in exps:
            if level != 'int':
                d = xr.open_dataset(path+'mars_analysis/summary_of_lats/%s_summary_of_lats_%i.nc' % (exp,level))
            else:
                d = xr.open_dataset(path+'mars_analysis/summary_of_lats/%s_summary_of_lats_%s.nc' % (exp,level))
            d = d.to_array().squeeze()
            lats.append(d)
        a, std = plot_summary_of_lats_line(lats, exps = exps,level=level,ext='pdf')
        plot_rate_of_change(a,std,exps=exps,level=level,ext='pdf')

# %%
