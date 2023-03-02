# %%

import os, sys
sys.path.append('/user/home/xz19136/Py_Scripts/atmospy/')
import xarray as xr

import analysis_functions as funcs
import tropd_exo as pyt
from multiprocessing import Pool, cpu_count
from scipy.signal import find_peaks
import numpy as np
import math

path = '/user/work/xz19136/Isca_data'

# Mars-specific!
theta0 = 200. # reference temperature
kappa = 0.25 # ratio of specific heats
p0 = 610. # reference pressure
omega = 7.08822e-05 # planetary rotation rate
gmars = 3.72076 # gravitational acceleration
rsphere = 3.3962e6 # mean planetary radius

rmars = 3.3962e6

def calculate_EDJ(exp_name):
    swap_file = '%s/%s/atmos.nc' % (path, exp_name)
    new_file = '%s/%s/EDJ.nc' % (path, exp_name)
    try:
        
        d1 = xr.open_dataset(swap_file, decode_times=False)

        if not os.path.isfile(new_file):
            a = []
            b = []
            c = []
            d = []
            e = []
        else:
            dat = xr.open_dataset(new_file,
                    decode_times=False)

            if (d1.time[0].values != dat.time[0].values) or \
                    (d1.time[-1].values != dat.time[-1].values):
                a = []
                b = []
                c = []
                d = []
                e = []
            else:
                if len(dat.lat) != 64:
                    a = []
                    b = []
                    c = []
                    d = []
                    e = []
                else:
                    return
        #try:
        ls = d1.mars_solar_long.values
        d1 = d1[["ucomp"]]
        d2 = d1.mean(dim="lon").squeeze()
        d2 = d2.transpose("lat","pfull","time")
        d2 = d2.sortby("pfull", ascending = True)
        d2 = d2.sortby("lat", ascending = True)
        lat = d2.lat.values
        lev = d2.pfull.values
        time = d2.time.values
        d2 = d2.ucomp

        d2 = d2.transpose("time","lat","pfull")
        d50 = d2.sel(pfull=0.5,method="nearest").squeeze()
        d50 = d50.interpolate_na(
            dim="time",method="quadratic",
            fill_value="extrapolate",limit=15)

        dmean = d50.mean(dim='time').compute()
        b = find_peaks(dmean)[0]
        bn = b[lat[b]>0]
        bs = b[lat[b]<0]
        a = []
        u_max_n = np.ma.max(dmean[bn])
        for l in range(len(bn)):
            if dmean[b[l]]/u_max_n*100 < 25:
                a.append(l)
        u_max_s = np.ma.max(dmean[bs])
        
        for l in range(len(bs)):
            if dmean[b[l]]/u_max_s*100 < 25:
                a.append(l)
        b = np.delete(b, a)

        if len(bn) >= 2:
            if np.ma.min([np.abs(lat[l]-lat[i]) for i, l in zip(bn[:-1], bn[1:])]) < 10:
                i,l = np.argmin([np.abs(lat[l]-lat[i]) for i, l in zip(bn[:-1], bn[1:])])
                l = np.argmin(dmean[i,l])
                b = np.delete(b,l)

        if len(bs) >= 2:
            if np.ma.min([np.abs(lat[l]-lat[i]) for i, l in zip(bs[:-1], bs[1:])]) < 10:
                i,l = np.argmin([np.abs(lat[l]-lat[i]) for i, l in zip(bs[:-1], bs[1:])])
                l = np.argmin(dmean[i,l])
                b = np.delete(b,l)
    

        
        jets_n, jets_s = pyt.find_STJ_jets(d50.values, lat,  lat[b])

        for i in range(jets_n.shape[1]):
            x = sum([math.isnan(l) for l in jets_n[:,i]])
            if x > len(d1.time) * 7/10:
                jets_n[:,i] = np.full(len(d1.time), np.nan)
        for i in range(jets_s.shape[1]):
            y = sum([math.isnan(l) for l in jets_s[:,i]])
            if y > len(d1.time) * 7/10:
                jets_s[:,i] = np.full(len(d1.time), np.nan)


        dat = xr.Dataset(data_vars = {"jets_n":(["time","njets"],jets_n),
                                      "jets_s":(["time","sjets"],jets_s),
                                      "u50":(["time","lat"],d50.data),
                                      "mars_solar_long":(["time"],ls)},
                    coords = {"time" :(["time"], time),
                              "lat"  :(["lat"], lat),
                              "njets":(["njets"], range(jets_n.shape[1])),
                              "sjets":(["sjets"], range(jets_s.shape[1]))},
                    attrs = dict(units="degrees N"))
        dat.to_netcdf(new_file,mode='w')
    except:
        print(exp_name+' failed. Continuing.')

def find_single_EDJ(exp_name):
        swap_file = '%s/%s/atmos.nc' % (path, exp_name)
        new_file = '%s/%s/EDJ.nc' % (path, exp_name)
    #try:
        
        d1 = xr.open_dataset(swap_file, decode_times=False)

        if not os.path.isfile(new_file):
            a = []
            b = []
            c = []
            d = []
            e = []
        else:
            dat = xr.open_dataset(new_file,
                    decode_times=False)

            if (d1.time[0].values != dat.time[0].values) or \
                    (d1.time[-1].values != dat.time[-1].values):
                a = []
                b = []
                c = []
                d = []
                e = []
            else:
                if len(dat.lat) != 64:
                    a = []
                    b = []
                    c = []
                    d = []
                    e = []
                else:
                    return
        #try:
        ls = d1.mars_solar_long.values
        d1 = d1[["ucomp"]]
        d2 = d1.sel(pfull=0.5,method="nearest").squeeze()
        d2 = d2.mean(dim="lon").squeeze()
        d2 = d2.sortby("lat", ascending = True)
        lat_n = d2.lat.where(d2.lat > 0, drop = True).values
        lat_s = d2.lat.where(d2.lat < 0, drop = True).values

        time = d2.time.values
        d2 = d2.ucomp

        d2 = d2.transpose("time","lat")
        
        d50 = d2.interpolate_na(
            dim="time",method="quadratic",
            fill_value="extrapolate",limit=15)

        for l in range(len(d2.time)):
            du = d50.isel(time=l)
            dn = du.where(du.lat > 0, drop=True)
            ds = du.where(du.lat < 0, drop=True)

            phi, mx = funcs.calc_jet_lat(dn, lat_n)
            a.append(phi)
            b.append(mx)

            phi, mx = funcs.calc_jet_lat(ds, lat_s)
            c.append(phi)
            d.append(mx)


        dat = xr.Dataset(data_vars = {"phi_n":(["time"],a),
                                      "jet_n":(["time"],b),
                                      "phi_s":(["time"],c),
                                      "jet_s":(["time"],d),
                                      "u50":(["time","lat"],d50.data),
                                      "mars_solar_long":(["time"],ls)},
                    coords = {"time" :(["time"], time),
                              "lat"  :(["lat"], d2.lat.values),},
                    attrs = dict(units="degrees N"))
        dat.to_netcdf(new_file,mode='w')
    #except:
    #    print(exp_name+' failed. Continuing.')
# %%
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

    #for i in exps:
    #    calculate_psi()

    with Pool(processes=7) as pool:
        pool.map(find_single_EDJ, exps)
                    
# %%
