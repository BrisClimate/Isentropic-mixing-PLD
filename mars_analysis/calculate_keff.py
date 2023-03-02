# %%

import xarray as xr
import numpy as np
import os, sys

sys.path.append('../')

import Contour2D, latitude_lengths_at, add_latlon_metrics, get_planet_parameters

import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

path     = '/user/work/xz19136/Isca_data/'

def test_PV_plot():
    dset = xr.open_dataset('PV.nc')
    
    r, deg2m, g, omega = get_planet_parameters('earth')
    # add metrics for xgcm.Grid
    x = calculate_keff(dset, deg2m, "PV", r)
    

    # Plot effective diffusiviy again in equivalent latitude space
    fontsize = 13
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    m=ax.contourf(x.new, x.level[8:], np.log(x.nkeff[8:]),
                cmap='jet', levels=np.linspace(0,4.6,24), extend='both')
    fig.colorbar(m, orientation='vertical', label='')
    ax.set_xlabel('equivalent latitude', fontsize=fontsize-2)
    ax.set_ylabel('isentropic level (K)', fontsize=fontsize-2)
    ax.set_title('normalized effective diffusivity', fontsize=fontsize)
    fig.savefig('test_keff.png')

def calculate_keff(dset, deg2m, tracer_name, r):
    # add metrics for xgcm.Grid
    dset, grid = add_latlon_metrics(dset,deg2m=deg2m)
    
    if tracer_name == 'test_tracer':
        tracer = dset.test_tracer
        try:
            grdS = dset.grdStr_2
        except:
            return
    else:
        try:
            tracer = dset.PV
        except:
            tracer = dset.pv
        #grdS = dset.grdSpv
        grdS = dset.grdSpv

    N  = 121           # increase the contour number may get non-monotonic A(q) relation
    increase = True    # Y-index increases with latitude (sometimes not)
    lt = True          # northward of PV contours (larger than) is inside the contour
                       # change this should not change the result of Keff, but may alter
                       # the values at boundaries
    dtype = np.float32 # use float32 to save memory
    undef = -9.99e8    # for maskout topography if present

    # initialize a Contour2D analysis class using grid and tracer
    analysis = Contour2D(grid, tracer,
                         dims={'X':'longitude','Y':'latitude'},
                         dimEq={'Y':'latitude'},
                         increase=increase,
                         lt=lt)
    # evenly-spaced contours
    ctr = analysis.cal_contours(N)

    # Mask for A(q) relation table.
    # This can be done analytically in simple case, but we choose to do it
    # numerically in case there are undefined values (topography) inside the domain.
    mask = xr.where(tracer!=undef, 1, 0).astype(dtype)

    #print(ctr)

    # calculate related quantities for Keff
    # First set of APIs
    # xarray's conditional integration, memory consuming and not preferred, for test only
    table   = analysis.cal_area_eqCoord_table(mask) # A(Yeq) table
    area    = analysis.cal_integral_within_contours(ctr).rename('intArea')
    intgrdS = analysis.cal_integral_within_contours(ctr, integrand=grdS).rename('intgrdS')

    # Second set of APIs
    # xhistogram's box-counting, memory-friendly and preferred, but not here as contour bins vary with level
    #table   = analysis.cal_area_eqCoord_table_hist(mask) # A(Yeq) table
    #area    = analysis.cal_integral_within_contours_hist(ctr).rename('intArea')
    #intgrdS = analysis.cal_integral_within_contours_hist(ctr, integrand=grdS).rename('intgrdS')

    latEq   = table.lookup_coordinates(area).rename('latEq')
    Lmin    = latitude_lengths_at(latEq,r=r).rename('Lmin')
    dintSdA = analysis.cal_gradient_wrt_area(intgrdS, area).rename('dintSdA')
    dqdA    = analysis.cal_gradient_wrt_area(ctr, area).rename('dqdA')
    Leq2    = analysis.cal_sqared_equivalent_length(dintSdA, dqdA).rename('Leq2')
    nkeff   = analysis.cal_normalized_Keff(Leq2, Lmin).rename('nkeff')
    # results in contour space
    ds_contour = xr.merge([ctr, area, intgrdS, latEq, dintSdA, dqdA, Leq2, Lmin, nkeff])

    # interpolate from contour space to equivalent-latitude space
    preLats = np.linspace(np.min(dset.latitude), np.max(dset.latitude), len(dset.latitude)).astype(dtype)
    # results in latEq space
    ds_latEq = analysis.interp_to_dataset(preLats, latEq, ds_contour)

    return ds_latEq

def setup(ds, tracer_name='test_tracer', planet='earth'):

    
    #ds = ds.sel(time=ds.time[-10:]).squeeze()
    r, deg2m, _, _ = get_planet_parameters(planet)
    print(r, deg2m)
    ds_full = []
    if planet == "mars":
        ds = ds.transpose("level","latitude","longitude","time")
        #ds = ds.sel(time = ds.time[-10:])
        for j in range(len(ds.time)):
            dset = ds.isel(time=j)
            ds_latEq = calculate_keff(dset, deg2m, tracer_name, r)
        
            ds_full.append(ds_latEq)
        
        ds_keff = xr.concat(ds_full, dim="time")
    else:
        dset = ds.transpose("level","latitude","longitude")
        ds_latEq = calculate_keff(dset, deg2m, tracer_name, r)
        
        ds_keff = ds_latEq
    
    ds_keff["tracer_name"] = tracer_name
    
    return ds_keff

def process_attr_exps(exp_name):
    
    print(exp_name)
    if os.path.isfile(path+exp_name+'/atmos.nc'):
        ds = xr.open_dataset(
                path+exp_name+'/atmos.nc', 
                decode_times = False,
                )
        ds = ds.sortby("pfull", ascending=False)

        dset = ds.rename(
                    {
                        'pfull':'level',
                        'lat'  :'latitude',
                        'lon'  :'longitude',
                    })
        x = []
        for tname in ['test_tracer']:#, 'PV']:
            if not os.path.isfile(path+exp_name+'/keff_%s.nc' % tname):
                #x.append(setup(dset, tracer_name=tname))
                x = setup(dset, tracer_name=tname, planet='mars')
                x["mars_solar_long"] = ds.mars_solar_long
                x.to_netcdf(path + exp_name + '/keff_%s.nc' % tname, mode='w')
                print('%s keff calculated' % tname)
            else:
                print('%s already exists' % tname)
        #d = xr.concatenate(x, dim = "tracer_name")
        #d.to_netcdf(path + exp_name + '/keff.nc', mode='w')

def iterate_over_all(eps):
    gamma = [0.0,0.093]
    
    for ga in gamma:
        exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' +\
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps, ga)
        print(exp_name)
        if os.path.isfile(path+exp_name+'/atmos.nc'):
            ds = xr.open_dataset(
                    path+exp_name+'/atmos.nc', 
                    decode_times = False,
                    )
            ds = ds.sortby("pfull", ascending=False)

            dset = ds.rename(
                        {
                            'pfull':'level',
                            'lat'  :'latitude',
                            'lon'  :'longitude',
                        })
            x = []
            for tname in ['test_tracer']:#, 'PV']:
                if not os.path.isfile(path+exp_name+'/keff_%s.nc' % tname):
                    #x.append(setup(dset, tracer_name=tname))
                    x = setup(dset, tracer_name=tname, planet='mars')
                    x["mars_solar_long"] = ds.mars_solar_long
                    x.to_netcdf(path + exp_name + '/keff_%s.nc' % tname, mode='w')
                    print('%s keff calculated' % tname)
                else:
                    print('%s already exists' % tname)
            #d = xr.concatenate(x, dim = "tracer_name")
            #d.to_netcdf(path + exp_name + '/keff.nc', mode='w')

def iterate_over_isentropic(eps):
    gamma = [0.0,0.093]
    
    for ga in gamma:
        exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' +\
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps, ga)
        if os.path.isfile(path+exp_name+'/atmos_isentropic.nc'):
            ds = xr.open_dataset(
                    path+exp_name+'/atmos_isentropic.nc', 
                    decode_times = False,
                    )

            dset = ds.rename(
                        {
                            'lat'  :'latitude',
                            'lon'  :'longitude',
                        })
            x = []
            for tname in ['test_tracer', 'PV']:
                if not os.path.isfile(path+exp_name+'/keff_isentropic_%s.nc' % tname):
                    print(exp_name)
                    x = setup(dset, tracer_name=tname, planet='mars')
                    x["mars_solar_long"] = ds.mars_solar_long
                    x.to_netcdf(path + exp_name + '/keff_isentropic_%s.nc' % tname, mode='w')
                    print('%s keff calculated' % tname)
            #d = xr.concatenate(x, dim = "tracer_name")
            #d.to_netcdf(path + exp_name + '/keff_isentropic.nc', mode='w')

def calculate_isentropic(exp_name):
    if os.path.isfile(path+exp_name+'/atmos_isentropic.nc'):
        ds = xr.open_dataset(
                path+exp_name+'/atmos_isentropic.nc', 
                decode_times = False,
                )

        dset = ds.rename(
                    {
                        'lat'  :'latitude',
                        'lon'  :'longitude',
                    })
        x = []
        for tname in ['test_tracer']:
            if not os.path.isfile(path+exp_name+'/keff_isentropic_%s.nc' % tname):
                print(exp_name)
                x = setup(dset, tracer_name=tname, planet='mars')
                x["mars_solar_long"] = ds.mars_solar_long
                x.to_netcdf(path + exp_name + '/keff_isentropic_%s.nc' % tname, mode='w')
                print('%s keff calculated' % tname)
        #d = xr.concatenate(x, dim = "tracer_name")
        #d.to_netcdf(path + exp_name + '/keff_isentropic.nc', mode='w')

def keff_era5():
    dset = xr.open_dataset('/user/work/xz19136/2010-01-01_PV.nc')
    tname = "PV"
    dset = dset[["PV", "grdSpv"]]
    dset = dset.rename({'pfull':'level'})
    
    #lats = np.linspace(-90,90,241)
    #lons = np.linspace(0,360, 480)
    #dset = dset.interp(latitude=lats)
    #dset = dset.interp(longitude=lons)
    #dset = dset.where(dset.level <= 800, drop = True)
    x = setup(dset, tracer_name="PV", planet='earth')
    x["time"] = dset.time
    x.to_netcdf('/user/work/xz19136/keff_%s.nc' % tname, mode='w')

def keff_era5_isentropic():
    dset = xr.open_dataset('/user/work/xz19136/2010-01-01_PV_isentropic.nc')
    ds   = xr.open_dataset('/user/home/xz19136/Py_Scripts/mars_analysis/PV.nc')
    dset = dset.interp(latitude=ds.latitude.values)
    dset = dset.interp(longitude=ds.longitude.values)
    tname = "PV"
    dset = dset[["PV", "grdSpv"]]
    #dset = dset.rename({'pfull':'level'})
    
    #lats = np.linspace(-90,90,241)
    #lons = np.linspace(0,360, 480)
    #dset = dset.interp(latitude=lats)
    #dset = dset.interp(longitude=lons)
    #dset = dset.where(dset.level <= 800, drop = True)
    x = setup(dset, tracer_name="PV", planet='earth')
    x["time"] = dset.time
    x.to_netcdf('/user/work/xz19136/keff_%s_isentropic.nc' % tname)


def plot_mars_keff(exp):
    # Select those levels above ground (400k above)
    path = '/user/work/xz19136/Isca_data/'
    dset = xr.open_dataset(path+exp+'/keff_test_tracer.nc', decode_times=False)
    dset = dset.sel(time=dset.time[4])
    preLats = dset.new
    tracer = dset.test_tracer
    # Plot effective diffusiviy again in equivalent latitude space
    fontsize = 13
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    for i, ax in enumerate(fig.axes):
        ax.set_ylim([6,0.01])
        ax.set_yscale('log')
        ax.set_ylabel('pressure (hPa)', fontsize=fontsize-2)
    
#
    m=axs[1].contourf(preLats, dset.level, np.log(dset.nkeff),
        cmap='jet', levels=np.linspace(0,6,24), extend='both')
    fig.colorbar(m, orientation='vertical', label='normalized effective diffusivity', ax=axs[1])
    axs[1].set_xlabel('equivalent latitude', fontsize=fontsize-2)
    axs[1].set_title('Ls = %.1f' % dset.mars_solar_long.values, fontsize=fontsize)
#
    m=axs[0].contourf(tracer.new, tracer.level, tracer,
        cmap='jet', levels=np.linspace(0,2.4,24), extend='both')
    fig.colorbar(m, orientation='vertical', label='', ax=axs[0])
    axs[0].set_xlabel('latitude', fontsize=fontsize-2)
    axs[0].set_title('tracer mass mixing ratio', fontsize=fontsize)
    fig.savefig('mars_test_keff.png')

    #plt.clf()

def plot_era5_keff():
    ds_latEq = xr.open_dataset('/user/work/xz19136/keff_PV.nc')
    fontsize = 13

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    ax = axes
    m=ax.contourf(np.log(ds_latEq.nkeff), cmap='jet', #levels=np.linspace(0,4.6,24),
                 extend='both')
    fig.colorbar(m, orientation='vertical', label='')
    ax.set_xlabel('equivalent latitude', fontsize=fontsize-2)
    ax.set_ylabel('pressure level (K)', fontsize=fontsize-2)
    ax.set_ylim([np.max(ds_latEq.level),np.min(ds_latEq.level)])
    ax.set_yscale('log')
    ax.set_title('normalized effective diffusivity', fontsize=fontsize)

def plot_era5_keff_isentropic():
    ds_latEq   = xr.open_dataset('/user/work/xz19136/keff_PV_isentropic.nc')
    fontsize = 13

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    m=axs.contourf(ds_latEq.new, ds_latEq.level[8:], np.log(ds_latEq.nkeff[8:]),
            cmap='jet', levels=np.linspace(0,4.6,24), extend='both',)
    fig.colorbar(m, orientation='vertical', label='')
    axs.set_xlabel('equivalent latitude', fontsize=fontsize-2)
    axs.set_ylabel('pressure level (hPa)', fontsize=fontsize-2)
    axs.set_title('normalized effective diffusivity', fontsize=fontsize)

if __name__ == "__main__":
    
    eps = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    gamma = [0.0,0.093]
    
    exps = []
    
    for l in ['', '_lh']:
        for d in ['', '_cdod_clim_scenario_7.4e-05']:
            for t in ['', '_mola_topo']:
                exps.append('tracer_soc_mars%s%s_eps_25_gamma_0.093%s' % (t, l, d))

    for ep in eps:
        for gam in gamma:
            exps.append('tracer_soc_mars_mola_topo_lh_eps_' + \
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (ep, gam))

    for dust_scale in [7.4e-05, 2.96e-4, 3.7e-5,1.48e-4]:
      exps.append('tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_'+str(dust_scale))

    
    for i in exps:
        iterate_over_all(i)
        
        calculate_isentropic(i)
        
        iterate_over_isentropic(i)
