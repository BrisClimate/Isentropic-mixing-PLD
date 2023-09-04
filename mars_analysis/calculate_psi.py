# %%

import numpy as np
import xarray as xr
import os, sys
import math
sys.path.append('../')

from atmospy import TropD_Metric_PSI

from multiprocessing import Pool, cpu_count

path = '/user/work/xz19136/Isca_data'

# Mars-specific!
theta0 = 200. # reference temperature
kappa = 0.25 # ratio of specific heats
p0 = 610. # reference pressure
omega = 7.08822e-05 # planetary rotation rate
gmars = 3.72076 # gravitational acceleration
rsphere = 3.3962e6 # mean planetary radius

rmars = 3.3962e6

def calculate_psi(exp_name):
    swap_file = '%s/%s/atmos.nc' % (path, exp_name)
    new_file = '%s/%s/psi.nc' % (path, exp_name)
    
    try:
        print(exp_name)
        
        d1 = xr.open_dataset(swap_file, decode_times=False)

        spb = 90
        pb = 90
        #if (1.5 <= j <= 2) and (np.abs(k) <= 15):
        #    pb = 35

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
        ls = d1.mars_solar_long.values
        d1 = d1[["vcomp","ucomp","temp",]]
        d2 = d1.mean(dim="lon")
        #d2 = d2.where(d2.pfull < 5.75, drop=True)
        d2 = d2.transpose("lat","pfull","time")
        #d2 = d2.interpolate_na(
        #    dim="time",method="linear",
        #    fill_value="extrapolate",limit=15)

        d2 = d2.sortby("pfull", ascending = True)
        d2 = d2.sortby("lat", ascending = True)
        lat = d2.lat.values
        lev = d2.pfull.values
        time = d2.time.values

        for l in range(len(d2.time)):
            try:
                dv = d2.vcomp.sel(time=d2.time[l]).squeeze()
                du = d2.ucomp.sel(time=d2.time[l]).squeeze()
                dT = d2.temp.sel(time=d2.time[l]).squeeze()
                dv = dv.transpose("lat","pfull").values
                du = du.transpose("lat","pfull").values
                dT = dT.transpose("lat","pfull").values
                psi0s, psi0n, psi = TropD_Metric_PSI(dv, lat, lev,
                                    method = "Psi_3_0.3",spb=spb,pb=pb,
                                    Radius=rmars, Grav=gmars)
                a.append(psi0s)
                b.append(psi0n)
                c.append(psi)
                d.append(du)
                e.append(dT)
            except:
                a.append(np.nan)
                b.append(np.nan)
                c.append(np.full((len(lat),len(lev)),np.nan))
                d.append(np.full((len(lat),len(lev)),np.nan))
                e.append(np.full((len(lat),len(lev)),np.nan))

        x = sum([math.isnan(i) for i in a])
        y = sum([math.isnan(i) for i in b])

        
        dat = xr.Dataset(
            data_vars = {
                "psi":(["time","lat","pfull"],c),
                "ucomp" :(["time","lat","pfull"],d),
                "temp" :(["time","lat","pfull"],e),
                "psi_0s":(["time"],a),
                "psi_0n":(["time"],b),
                "mars_solar_long":(["time"],ls)
                },
            coords = {"pfull":(["pfull"], lev),
                "lat"  :(["lat"], lat),
                "time" :(["time"], time)
                },
            attrs = dict(
                description="Meridional streamfunction",
                units="kg/s"
                )
            )
        
        
        dat.to_netcdf(new_file, mode='w')
    except:
        print(exp_name+' failed. Continuing.')


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

    for dust_scale in [7.4e-05, 3.7e-5,1.48e-4,2.96e-4,5.92e-4]:
      exps.append('tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_'+str(dust_scale))
      exps.append('tracer_vert_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_'+str(dust_scale))
      exps.append('tracer_soc_mars_mola_topo_lh_eps_25_gamma_0.093_clim_latlon_'+str(dust_scale))
    
    exps.append('tracer_MY28_soc_mars_mola_topo_lh_eps_25_gamma_0.093_cdod_clim_scenario_7.4e-05')
    
      
    for i in exps:
        calculate_psi(i)

                
# %%
