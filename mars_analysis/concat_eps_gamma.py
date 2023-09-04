# %%
import xarray as xr
import numpy as np
import sys, os

sys.path.append('../')

from atmospy import open_files, stereo_plot, get_timeslice

import string

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import (cm, colors, cycler)
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


path = '/user/work/xz19136/Isca_data/'
theta, center, radius, verts, circle = stereo_plot()
theta0 = 200.
kappa = 1/4.0

if plt.rcParams["text.usetex"]:
    fmt = r'%r \%'
else:
    fmt = '%r'

def concat_parameter(isentropic = False):
    g_keff = []
    for gamma in [0.093, 0.00]:
        e_keff = []
        for eps in [10,15,20,25,30,35,40,45,50]:#,30,35,40,45,50]:
            exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps, gamma)
            print(exp_name)
            ds, d = open_files(path,exp_name, isentropic)
            ds = ds.interp({'new':d.lat.values})
            ds['epsilon'] = eps
            
            e_keff.append(ds)
        
        e_keff = xr.concat(e_keff, dim='epsilon')
        e_keff['gamma'] = gamma
        g_keff.append(e_keff)


    g_keff = xr.concat(g_keff, dim='gamma')
    if isentropic:
        ise = '_isentropic'
    else:
        ise = ''
    g_keff.to_netcdf(path+'mars_analysis/keffs/parameter_keff_test_tracer%s.nc' %ise, mode='w')

def concat_attribution(isentropic = False):
    c = []
    for t in ['', '_mola_topo']:
        b = []
        for dust in ['', '_cdod_clim_scenario_7.4e-05']:
            a = []
            for l in ['', '_lh']:
                exp_name = 'tracer_soc_mars%s%s_eps_25_gamma_0.093%s' % (t, l, dust)
                print(exp_name)
                ds, d = open_files(path, exp_name, isentropic)
                ds = ds.interp({'new':d.lat.values})
                if l == '_lh':
                    ds['lh'] = 1
                else:
                    ds['lh'] = 0

                a.append(ds)
            a = xr.concat(a, dim='lh')
            if dust == '':
                a['dust'] = 0
            else:
                a['dust'] = 1
            b.append(a)
        b = xr.concat(b, dim = "dust")
        if t == '':
            b['topo'] = 0
        else:
            b['topo'] = 1
        c.append(b)
    
    c = xr.concat(c, dim = "topo")

    if isentropic:
        ise = '_isentropic'
    else:
        ise = ''
    c.to_netcdf(path+'mars_analysis/keffs/attribution_keff_test_tracer%s.nc' %ise, mode='w')

def concat_dust(isentropic = False, res=''):
    
    a = []
    for dust in [3.7e-5, 7.4e-5,1.48e-4,2.96e-4,5.92e-4]:
        exp_name = 'tracer_%ssoc_mars_mola_topo_lh_eps_' % res + \
                    '25_gamma_0.093_cdod_clim_scenario_%s' % str(dust)

        print(exp_name)
        ds, d = open_files(path, exp_name, isentropic)
        ds = ds.interp({'new':d.lat.values})
        ds['dust_scale'] = dust
        a.append(ds)
    a = xr.concat(a, dim="dust_scale")
    if isentropic:
        ise = '_isentropic'
    else:
        ise = ''
    a.to_netcdf(path+'mars_analysis/keffs/%sdust_keff_test_tracer%s.nc' %(res,ise), mode='w')

def add_MY28(isentropic = False):
    
    exp_name = 'tracer_MY28_soc_mars_mola_topo_lh_eps_' + \
                    '25_gamma_0.093_cdod_clim_scenario_7.4e-05'

    ds, d = open_files(path, exp_name, isentropic)
    ds = ds.interp({'new':d.lat.values})
    
    if isentropic:
        ise = '_isentropic'
    else:
        ise = ''
    ds.to_netcdf(path+'mars_analysis/keffs/MY28_keff_test_tracer%s.nc' %ise, mode='w')

def add_long(isentropic = False):
    a=[]
    
    exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                    '25_gamma_0.093_cdod_clim_scenario_7.4e-05'

    ds, d = open_files(path, exp_name, isentropic)
    ds = ds.interp({'new':d.lat.values})
    ds['long_dust'] = 0
    a.append(ds)
    
    exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                    '25_gamma_0.093_clim_latlon_7.4e-05'

    ds, d = open_files(path, exp_name, isentropic)
    ds = ds.interp({'new':d.lat.values})
    ds['long_dust'] = 1
    a.append(ds)

    if isentropic:
        ise = '_isentropic'
    else:
        ise = ''
    ds.to_netcdf(path+'mars_analysis/keffs/latlon_keff_test_tracer%s.nc' %ise, mode='w')

def concat_winter_variables(exp, res = '', hem = 'nh'):
    if exp == 'dust':
        d_keff = xr.open_dataset(path + 
                               'mars_analysis/keffs/%sdust_keff_test_tracer.nc' % res,
                               decode_times=False)
        if hem == 'nh':
            tind = 115
            d_keff = d_keff.where(d_keff.new >= 0, drop = True)
        else:
            tind = 450
            d_keff = d_keff.where(d_keff.new <= 0, drop = True)
        tind, m = get_timeslice(tind, 10)
        if m != 0:
            tslice = slice(tind-m, tind+m)

            d_keff = d_keff.nkeff.isel(time=tslice)
            d_keff = d_keff.mean(dim="time")
        else:
            d_keff = d_keff.nkeff.isel(time=tind)

        d_keff = d_keff.where(d_keff.level <= 5.5, drop = True)
        d_keff = d_keff.rename({'level':'pfull'})
        d_keff = d_keff.rename({'new':'lat'})

        a = []
        for dust in [3.7e-5, 7.4e-5,1.48e-4,2.96e-4,5.92e-4]:
            exp_name = 'tracer_%ssoc_mars_mola_topo_lh_eps_' % res + \
                        '25_gamma_0.093_cdod_clim_scenario_%s' % str(dust)

            print(exp_name)
            _, d = open_files(path, exp_name, isentropic)
            d = d[["theta", "PV", "ucomp", "mars_solar_long"]]

            if hem == 'nh':
                d  =  d.where( d.lat >= 0, drop = True)
            else:
                d  =  d.where( d.lat <= 0, drop = True)

            if m != 0:
                tslice = slice(tind-m, tind+m)

                d  =  d.isel(time=tslice)
                d  =  d.mean(dim="time")
            else:
                d  =  d.isel(time=tind)
            d  =  d.where(d.pfull  <= 5.5, drop = True)
            d = d.mean(dim=["lon"])
            
            d['dust_scale'] = dust
            a.append(d)
        a = xr.concat(a, dim="dust_scale")
        a["keff"] = d_keff
        a.to_netcdf(path+
            'mars_analysis/winter_vars/%sdust_%s.nc' %(res,hem),
            mode='w')
        
    if exp == 'parameter':
        d_keff = xr.open_dataset(
            path+'mars_analysis/keffs/parameter_keff_test_tracer.nc',
            decode_times=False)
        if hem == 'nh':
            tind = 115
            d_keff = d_keff.where(d_keff.new >= 0, drop = True)
        else:
            tind = 450
            d_keff = d_keff.where(d_keff.new <= 0, drop = True)
        tind, m = get_timeslice(tind, 10)
        if m != 0:
            tslice = slice(tind-m, tind+m)

            d_keff = d_keff.nkeff.isel(time=tslice)
            d_keff = d_keff.mean(dim="time")
        else:
            d_keff = d_keff.nkeff.isel(time=tind)

        d_keff = d_keff.where(d_keff.level <= 5.5, drop = True)
        d_keff = d_keff.rename({'level':'pfull'})
        d_keff = d_keff.rename({'new':'lat'})

        g_keff = []
        for gamma in [0.093, 0.00]:
            e_keff = []
            for eps in [10,15,20,25,30,35,40,45,50]:#,30,35,40,45,50]:
                exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                    '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps, gamma)
                print(exp_name)
                _, d = open_files(path,exp_name, isentropic)
                d = d[["theta", "PV", "ucomp", "mars_solar_long"]]

                if hem == 'nh':
                    d  =  d.where( d.lat >= 0, drop = True)
                else:
                    d  =  d.where( d.lat <= 0, drop = True)

                if m != 0:
                    tslice = slice(tind-m, tind+m)

                    d  =  d.isel(time=tslice)
                    d  =  d.mean(dim="time")
                else:
                    d  =  d.isel(time=tind)
                d = d.where(d.pfull  <= 5.5, drop = True)
                d = d.mean(dim=["lon"])
                d['epsilon'] = eps

                e_keff.append(d)

            e_keff = xr.concat(e_keff, dim='epsilon')
            e_keff['gamma'] = gamma
            g_keff.append(e_keff)


        g_keff = xr.concat(g_keff, dim='gamma')

        
        g_keff["keff"] = d_keff
        g_keff.to_netcdf(path+
            'mars_analysis/winter_vars/parameter_%s.nc' %(hem),
            mode='w')
        
    if exp == 'long-dust':
        d_keff = xr.open_dataset(path + 
                               'mars_analysis/keffs/latlon_keff_test_tracer.nc',
                               decode_times=False)
        if hem == 'nh':
            tind = 115
            d_keff = d_keff.where(d_keff.new >= 0, drop = True)
        else:
            tind = 450
            d_keff = d_keff.where(d_keff.new <= 0, drop = True)
        tind, m = get_timeslice(tind, 10)
        if m != 0:
            tslice = slice(tind-m, tind+m)

            d_keff = d_keff.nkeff.isel(time=tslice)
            d_keff = d_keff.mean(dim="time")
        else:
            d_keff = d_keff.nkeff.isel(time=tind)

        d_keff = d_keff.where(d_keff.level <= 5.5, drop = True)
        d_keff = d_keff.rename({'level':'pfull'})
        d_keff = d_keff.rename({'new':'lat'})

        a = []
        for dust in ['cdod_clim_scenario','clim_latlon']:
            exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                        '25_gamma_0.093_%s_7.4e-05' % dust

            print(exp_name)
            _, d = open_files(path, exp_name, isentropic)
            d = d[["theta", "PV", "ucomp", "mars_solar_long"]]

            if hem == 'nh':
                d  =  d.where( d.lat >= 0, drop = True)
            else:
                d  =  d.where( d.lat <= 0, drop = True)

            if m != 0:
                tslice = slice(tind-m, tind+m)

                d  =  d.isel(time=tslice)
                d  =  d.mean(dim="time")
            else:
                d  =  d.isel(time=tind)
            d  =  d.where(d.pfull  <= 5.5, drop = True)
            d = d.mean(dim=["lon"])
            if dust == 'cdod_clim_scenario':
                long = 0
            elif dust == 'clim_latlon':
                long = 1
            d['long_dust'] = long
            a.append(d)
        a = xr.concat(a, dim="long_dust")
        a["keff"] = d_keff
        a.to_netcdf(path+
            'mars_analysis/winter_vars/latlon_dust_%s.nc' %(hem),
            mode='w')
    
def concat_isentrope_variables(exp, res = '', level = 300):
    isentropic=True
    if exp == 'dust':
        d_keff = xr.open_dataset(path + 
            'mars_analysis/keffs/%sdust_keff_test_tracer_isentropic.nc' % res,
            decode_times=False)
        
        d_keff = d_keff.nkeff.sel(level=level,method="nearest")
        level = int(d_keff.level.values)
        
        d_keff = d_keff.rename({'new':'lat'})

        a = []
        for dust in [3.7e-5, 7.4e-5,1.48e-4,2.96e-4,5.92e-4]:
            exp_name = 'tracer_%ssoc_mars_mola_topo_lh_eps_' % res + \
                        '25_gamma_0.093_cdod_clim_scenario_%s' % str(dust)

            print(exp_name)
            _, d = open_files(path, exp_name, isentropic)
            d = d[["PV", "ucomp", "mars_solar_long"]]
            d = d.sel(level = level, method="nearest").mean(dim="lon")

            d['dust_scale'] = dust
            a.append(d)
        a = xr.concat(a, dim="dust_scale")
        a["keff"] = d_keff
        a.to_netcdf(path+
            'mars_analysis/isentropic_vars/%sdust_%iK.nc' %(res,level),
            mode='w')
        
    if exp == 'parameter':
        d_keff = xr.open_dataset(
            path+'mars_analysis/keffs/parameter_keff_test_tracer_isentropic.nc',
            decode_times=False)
        d_keff = d_keff.nkeff.sel(level=level,method="nearest")
        level = int(d_keff.level.values)
        
        d_keff = d_keff.rename({'new':'lat'})

        g_keff = []
        for gamma in [0.093, 0.00]:
            e_keff = []
            for eps in [10,15,20,25,30,35,40,45,50]:#,30,35,40,45,50]:
                exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                    '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps, gamma)
                print(exp_name)
                _, d = open_files(path,exp_name, isentropic)
                d = d[["PV", "ucomp", "mars_solar_long"]]

                d = d.sel(level = level, method="nearest").mean(dim="lon")
                d['epsilon'] = eps

                e_keff.append(d)

            e_keff = xr.concat(e_keff, dim='epsilon')
            e_keff['gamma'] = gamma
            g_keff.append(e_keff)


        g_keff = xr.concat(g_keff, dim='gamma')

        
        g_keff["keff"] = d_keff
        g_keff.to_netcdf(path+
            'mars_analysis/isentropic_vars/parameter_%iK.nc' % level,
            mode='w')
    
    if exp == 'long-dust':
        d_keff = xr.open_dataset(path + 
            'mars_analysis/keffs/latlon_keff_test_tracer_isentropic.nc',
            decode_times=False)
        
        d_keff = d_keff.nkeff.sel(level=level,method="nearest")
        level = int(d_keff.level.values)
        
        d_keff = d_keff.rename({'new':'lat'})

        a = []
        for dust in ['cdod_clim_scenario','clim_latlon']:
            exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                        '25_gamma_0.093_%s_7.4e-05' % dust

            print(exp_name)
            _, d = open_files(path, exp_name, isentropic)
            d = d[["PV", "ucomp", "mars_solar_long"]]
            d = d.sel(level = level, method="nearest").mean(dim="lon")
            if dust == 'cdod_clim_scenario':
                long = 0
            elif dust == 'clim_latlon':
                long = 1
            d['long_dust'] = long
            a.append(d)
        a = xr.concat(a, dim="long_dust")
        a["keff"] = d_keff
        a.to_netcdf(path+
            'mars_analysis/isentropic_vars/latlon_dust_%iK.nc' %(level),
            mode='w')


if __name__ == "__main__":
    isentropic = False
    #add_MY28(isentropic=isentropic)
    add_long(isentropic=True)
    add_long(isentropic=False)

    concat_winter_variables('long-dust')
    concat_winter_variables('long-dust', hem='sh')
    concat_isentrope_variables('long-dust')
    #concat_dust(isentropic=isentropic,res='')
    #concat_dust(isentropic=True,res='')
    #concat_dust(isentropic=isentropic,res='vert_')
    #concat_winter_variables('dust',)
    #concat_winter_variables('dust',hem='sh')
    #concat_isentrope_variables('parameter',)
    #concat_parameter(isentropic=isentropic)

    #concat_attribution(isentropic=isentropic)

            #for t in ['test_tracer']:#, 'PV']:
# %%
