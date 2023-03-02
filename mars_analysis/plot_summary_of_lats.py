# %%
import xarray as xr
import numpy as np
import sys, os
import math

sys.path.append('../')

import atmospy

from plot_HC import get_HC_edge
from plot_PV import get_PV_lats_isentropic
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
        
        
        for j in range(len(exp_names)):

            exp_name = exp_names[j]
            print(exp_name)
            
            
            ds = xr.open_dataset(path+exp_name+'/psi.nc', decode_times=False)
            if exp != '0-ecc':
                cond1 = ds.mars_solar_long>180
                cond2 = ds.mars_solar_long<180

            psi0n = get_HC_edge(ds.where(ds.lat>0,drop=True).where(cond1,drop=True))
            psi0s = get_HC_edge(ds.where(ds.lat<0,drop=True).where(cond2,drop=True).sortby('lat',ascending=False))
            
            a_s[j,0,0] = np.nanmean(psi0n)
            a_s[j,1,0] = np.nanmean(psi0s)

            ds = xr.open_dataset(path+exp_name+'/EDJ.nc', decode_times=False)
            
            phi_n = ds.phi_n.squeeze()
            phi_s = ds.phi_s.squeeze()
            
            a_s[j,0,1] = phi_n.where(cond1,drop=True).mean(dim="time").values
            a_s[j,1,1] = phi_s.where(cond2,drop=True).mean(dim="time").values

            ds = xr.open_dataset(path+exp_name+'/atmos_isentropic.nc', decode_times=False)

            ds = ds[["PV","mars_solar_long"]].sel(level=level,method="nearest")
            ds = ds.mean(dim="lon")

            phiPV_n, _ = get_PV_lats_isentropic(ds.where(ds.lat > 0, drop=True).where(cond1,drop=True),hem='nh')
            phiPV_s, _ = get_PV_lats_isentropic(ds.where(ds.lat < 0, drop=True).where(cond2,drop=True),hem='sh')

            a_s[j,0,2] = np.nanmean(phiPV_n)
            a_s[j,1,2] = np.nanmean(phiPV_s)

            ds = xr.open_dataset(path+exp_name+'/keff_isentropic_test_tracer.nc', decode_times=False)

            ds = ds.nkeff.sel(level=level,method="nearest")

            keff_n = ds.where(ds.new > 40, drop=True).where(cond1,drop=True)
            dn_weighted = keff_n.weighted(np.cos(np.deg2rad(keff_n.new)))
            keff_n = dn_weighted.mean(dim="new")
            keff_s = ds.where(ds.new <-40, drop=True).where(cond2,drop=True)
            dn_weighted = keff_s.weighted(np.cos(np.deg2rad(keff_s.new)))
            keff_s = dn_weighted.mean(dim="new")

            a_s[j,0,3] = np.nanmean(np.log(keff_n))
            a_s[j,1,3] = np.nanmean(np.log(keff_s))

            dat = xr.DataArray(a_s,
                coords = {'exp_name':exp_names, 'hem':[0,1], 'phi':['HC','u','PV','keff']},
                dims = ['exp_name','hem','phi'],
                )
            dat.to_netcdf(path+'mars_analysis/%s_summary_of_lats.nc' % exp)
            
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

def plot_summary_of_lats_line(lats, exps = ['curr-ecc','0-ecc','dust'],level=300,ext='png'):
    fig, axs = plt.subplots(nrows=len(lats)-1,ncols=2, figsize = (10,2.5*len(lats)-1),dpi=300)
    
    colors = plt.cm.viridis(np.linspace(0,1,4))
    matplotlib.rcParams['axes.prop_cycle'] = (
            cycler('color', colors) * \
            cycler('linestyle', ['-','--'])
        )

    for i, ax in enumerate(fig.axes):
        ax.text(-0.01, 1.03, string.ascii_lowercase[i+2]+')', transform=ax.transAxes)
    
    cmap = plt.get_cmap("viridis", len(lats)+1)
    norm = matplotlib.colors.BoundaryNorm(np.arange(5)+0.5,4)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # this line may be ommitted for matplotlib >= 3.1

    a   = []
    std = []

    for i in range(len(exps)):
        lat = lats[i]
        exp_names, _, _, _ = atmospy.get_exps(exps[i])

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
            xticklabs = np.array([1/2,1,2,4])
            xlabel = 'Dust Scale ($\\tau}$)'
            k = i-1
            lnstl = '-'
            labs = ['$\phi_{\\rm HC}$','$\phi_{u_{\\rm max}}$','$\phi_{\\rm PV}$','$\kappa_{{\\rm eff}_{300}}$']
        elif exps[i] == 'attribution':
            exp = 'Attribution'
            labs = ['$\phi_{\\rm HC}$','$\phi_{u_{\\rm max}}$','$\phi_{\\rm PV}$','$\kappa_{{\\rm eff}_{300}}$']

        axs[k,0].plot(range(len(exp_names)),lat[:,0,0],
                    c = cmap(1),linestyle=lnstl,label=labs[0])
        axs[k,0].plot(range(len(exp_names)),lat[:,0,1],
                    c = cmap(2),linestyle=lnstl,label=labs[1])
        axs[k,0].plot(range(len(exp_names)),lat[:,0,2],
                    c = cmap(3),linestyle=lnstl,label=labs[2])

        axs[k,1].plot(range(len(exp_names)),lat[:,1,0],
                    c = cmap(1),linestyle=lnstl,label=labs[0])
        axs[k,1].plot(range(len(exp_names)),lat[:,1,1],
                    c = cmap(2),linestyle=lnstl,label=labs[1])
        axs[k,1].plot(range(len(exp_names)),lat[:,1,2],
                    c = cmap(3),linestyle=lnstl,label=labs[2])

        diff = np.diff(np.abs(lat),axis=0)
        xticks = xticklabs.reshape(-1,1).repeat(diff.shape[1], axis=1)
        xticks = xticks.reshape(xticks.shape[0],-1,1).repeat(diff.shape[2], axis=2)
        
        a.append(np.mean(diff/np.diff(xticks,axis=0),axis=0))
        std.append(np.std(diff/np.diff(xticks,axis=0), axis=0))

        for j in [0,1]:
            ax = axs[k,j]
            xticks = [l for l in range(len(exp_names))]
            #ax.set_xticks([])
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabs)
            ax.set_xlabel(xlabel)
            ax1 = ax.twinx()
            
            ax1.plot(range(len(exp_names)),lat[:,j,3],
                    c = cmap(0),linestyle=lnstl,label=labs[3])
            #ax1.set_xticks([])
            ax1.set_ylim([0,1.9])
            #ax1.set_yticks([0.5,1.0,1.5,2.0])
            ax1.spines['right'].set_color(cmap(0))
            ax1.tick_params(axis='y',colors=cmap(0))
            if j == 1:
                ax1.set_ylabel('Effective diffusivity', color=cmap(0))
                ax1.tick_params(axis='y', labelcolor=cmap(0))
                ax.set_yticklabels([])
            else:
                ax1.set_yticklabels([])
            
            ax.plot([],[],c = cmap(0),linestyle=lnstl,label=labs[3])
            #if exps[i] != '0-ecc':
            #    ax1.legend()

    axs[0,0].set_title('NH')
    axs[0,1].set_title('SH')
    axs[0,0].set_ylim([ 35, 90])
    axs[0,1].set_ylim([-35,-90])
    axs[1,0].set_ylim([ 35, 90])
    axs[1,1].set_ylim([-35,-90])

    #axs[0,0].legend(loc='upper left')
    #axs[0,1].legend(loc='upper left')
    #axs[1,0].legend(loc='upper left')
    axs[1,0].legend(loc='center left')

    axs[0,0].set_ylabel('latitude ($^\circ$N/S)')
    #axs[0,0].legend()
    axs[1,0].set_ylabel('latitude ($^\circ$N/S)')

    fig.tight_layout()
    fig.savefig(figpath + 'line_summary_of_lats_%i_log.%s' % (level, ext), dpi=300,
                bbox_inches='tight')

    return a, std


def plot_rate_of_change(a, std, exps = ['curr-ecc','0-ecc','dust'],level=300,ext='png'):
    fig, axs = plt.subplots(nrows=1,ncols=len(exps)-1,
                            figsize = (2.75*(len(exps)),3),dpi=300)
    for i, ax in enumerate(fig.axes):
        ax.text(-0.01, 1.03, string.ascii_lowercase[i]+')', transform=ax.transAxes,fontsize='small')
    
    axs[0].annotate('', xy = (-0.37,1), xycoords='axes fraction', xytext=(-0.37,0.45),
        arrowprops=dict(arrowstyle='->'),
    )

    axs[0].text(
        -0.4, 0.7, 'poleward',
        ha='right',
        va='center',
        transform=axs[0].transAxes,
        rotation='vertical',
        fontsize='small',
    )

    axs[0].annotate('', xy = (-0.37,0), xycoords='axes fraction', xytext=(-0.37,0.38),
        arrowprops=dict(arrowstyle='->'),
    )

    axs[0].text(
        -0.4, 0.2, 'equatorward',
        ha='right',
        va='center',
        transform=axs[0].transAxes,
        rotation='vertical',
        fontsize='small',
    )
    axs[1].annotate('', xy = (1.32,1), xycoords='axes fraction', xytext=(1.32,0),
        arrowprops=dict(arrowstyle='->'),
    )

    axs[1].text(
        1.4, 0.5, 'increasing $\kappa_{{\\rm eff}_{300}}$',
        ha='right',
        va='center',
        transform=axs[1].transAxes,
        rotation='vertical',
        #fontsize='large',
    )
    offs = [-0.13,0.13,0]
    k = 0
    l = 0
    denom = ['$^\circ$obliquity','$\\tau$']
    markers = ['o','s','d','^','v','x',]
    labs = ['$\phi_{\\rm HC}$','$\phi_{u_{\\rm max}}$','$\phi_{\\rm PV}$','$\kappa_{{\\rm eff}_{300}}$']
    for i in range(len(a)):

        if exps[i] == 'curr-ecc':
            xlabel = '$\gamma=0.093$'
            
        elif exps[i] == '0-ecc':
            xlabel = '$\gamma=0.000$'
            
        elif exps[i] == 'dust':
            xlabel = 'Dust Scale ($\\tau}$)'
            k +=1
            

        elif exps[i] == 'attribution':
            xlabel = 'Attribution'
            k+=1

        mark = markers[i]
        l+=1
        ax = axs[k]
        #a[i][1,-1] = -a[i][1,-1]     # because other metrics are moving poleward/equatorward
        ax.errorbar(np.arange(a[i].shape[1]-1)+offs[i]+0.05, a[i][0,:-1],yerr=std[i][0,:-1],
            color='xkcd:aquamarine', capsize=2,zorder=1,
            marker=mark, ms=6,alpha=0.7,linestyle='',)
        ax.scatter([],[],color='xkcd:aquamarine',
            marker=mark, s=35,label='%s NH' % xlabel,alpha=0.7,)
        #ax.errorbar(np.arange(a[i].shape[1]-1)+offs[i], a[i][0,:-1],yerr=std[i][0,:-1],
        #    color='xkcd:aquamarine',alpha=0.5,linestyle='')

        ax.errorbar(np.arange(a[i].shape[1]-1)+offs[i]-0.05, a[i][1,:-1],yerr=std[i][1,:-1],
            color='xkcd:darkblue', capsize=2,zorder=0,
            marker=mark, ms=6,alpha=0.7,linestyle='',)
        ax.scatter([],[],color='xkcd:darkblue',
            marker=mark, s=30,label='%s SH' % xlabel,alpha=0.7,)
        
    l=0
    for i, ax in enumerate(fig.axes):
        ax.set_xticks(np.arange(a[i].shape[1]))
        ax.set_xticklabels(labs,fontsize='large')
        ax.tick_params(axis='y',labelsize='small')
        ax.set_ylabel('$\Delta(^\circ$N/S) / $\Delta$(%s)' % denom[i])
        ax.hlines(0,0,a[i].shape[1]-1.75,colors='gray',linestyles='--',alpha=0.5,zorder=0)
        ax.vlines(a[i].shape[1]-1.5,ax.get_ylim()[0],ax.get_ylim()[1],
            colors='k',linestyles='-',alpha=0.5,linewidths=0.8)
        ax.legend(fontsize='xx-small',loc='upper center',)# bbox_to_anchor=(1.05,0.75,),
                 #borderaxespad=0)

        ax1 = ax.twinx()
        ax1.tick_params(axis='y',labelsize='small')
        if i > 0:
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
    fig.savefig(figpath + 'lat_rates_of_change_%i_log.%s' % (level, ext), dpi=300,
                bbox_inches='tight')

#%%
if __name__ == "__main__":

    eps = np.arange(10,55,5)
    gamma = [0.093,0.00]
    exps = ['curr-ecc','0-ecc','dust']
    level = 300
    savedata = False
    
    if savedata:
        lats = get_lats_all(exps=exps,level=level)

        a, std = plot_summary_of_lats_line(lats, exps = exps,level=level,ext='pdf')
        plot_rate_of_change(a,std,level=level,ext='pdf')
    else:
        lats = []
        for exp in exps:
            d = xr.open_dataset(path+'mars_analysis/%s_summary_of_lats.nc' % exp)
            d = d.to_array().squeeze()
            lats.append(d)
        a, std = plot_summary_of_lats_line(lats, exps = exps,level=level,ext='pdf')
        plot_rate_of_change(a,std,level=level,ext='pdf')

# %%
