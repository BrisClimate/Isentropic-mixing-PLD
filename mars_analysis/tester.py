'''
File to add grdStracer to output
'''
#%%
import xarray as xr
import os, sys
import numpy as np
sys.path.append('/user/home/xz19136/Py_Scripts/atmospy/')
import analysis_functions as funcs
import PVmodule as PV
import tropd_exo as pyt

import calculate_PV_Isca

from multiprocessing import Pool, cpu_count
import windspharm.xarray as windx

path = '/user/work/xz19136/Isca_data/'

Om_E = 7.292e-5
g_E = 9.81
r_E = 6.371e6
om_E = 360*60*60*24
ps_E = 1.0e5

def calc_grdS(w, tr):
    '''
    Calculate the gradient squared of a field using a VectorWind object

    Input
    -----
    w : VectorWind object
    tr: tracer field

    Output
    ------
    returns the gradient squared of the given tracer field
    '''
    grd_tr_z, grd_tr_m = w.gradient(tr)

    return grd_tr_z**2 + grd_tr_m**2

def append_all_era5():
    t = xr.open_dataset('/user/work/xz19136/2010-01-01_T.nc')
    u = xr.open_dataset('/user/work/xz19136/2010-01-01_U.nc')
    v = xr.open_dataset('/user/work/xz19136/2010-01-01_V.nc')


    dset = xr.Dataset({
      "temp"  : (("pfull","latitude","longitude"), t.t.squeeze().data),
      "ucomp" : (("pfull","latitude","longitude"), u.u.squeeze().data),
      "vcomp" : (("pfull","latitude","longitude"), v.v.squeeze().data),
    },
    coords = {
      "pfull"    : t.level.values,
      "latitude" : t.latitude.values,
      "longitude": t.longitude.values,
      "time"     : t.time,
    })

    #ds = xr.open_dataset('/user/home/xz19136/Py_Scripts/mars_analysis/PV.nc')

    #dset = dset.interp(latitude=ds.latitude.values)
    #dset = dset.interp(longitude=ds.longitude.values)

    r = 6371200.0
    deg2earth = 2.0 * np.pi * r / 360.0
    ge = 9.80665
    omega_E = 7.292e-5
    p0 = 101300
    Rd = 8.31446/(28.965*1e-3)
    Cpd = 1.4 * Rd / (1.4 - 1)
    kappa = Rd/Cpd


    dset['pfull'] = dset.pfull*100
    dset = dset.transpose('latitude','longitude','pfull')
    ds   = xr.open_dataset('/user/home/xz19136/Py_Scripts/mars_analysis/PV.nc')
    dset = dset.interp(latitude=ds.latitude.values)
    dset = dset.interp(longitude=ds.longitude.values)
    w = windx.VectorWind(dset.ucomp.fillna(0), dset.vcomp.fillna(0))
    _, PV_isobaric = calculate_PV_Isca.calculate_PV(dset,
                kappa=kappa, p0=p0, omega=omega_E, g=ge, rsphere=r)

    dset['PV']    = PV_isobaric

    dset["grdSpv"] = calc_grdS(w, dset.PV.fillna(0))
    dset['pfull'] = dset.pfull/100
    dset.to_netcdf('/user/work/xz19136/2010-01-01_PV.nc')

    dset['pfull'] = dset.pfull*100
    levs = [265,275,285,300,315,330,350,370,395,430,475,530,600,700,850]
    print("Interpolating")
    d = calculate_PV_Isca.interpolate_to_isentropic(dset,
          levels=levs, p0=p0, kappa=kappa)
    print("Interpolated to isentropic levels")

    d["pressure"] = d.pressure/100
    w = windx.VectorWind(d.ucomp.fillna(0), d.vcomp.fillna(0))
    d["grdSpv_new"] = calc_grdS(w, d.PV.fillna(0))
    d.to_netcdf('/user/work/xz19136/2010-01-01_PV_isentropic.nc')

def append_grdStrac(exp):
    p_file = 'atmos_daily_interp.nc'
    
    _, _, i_files = funcs.filestrings(exp, path, 1, 403, p_file)
    dset = xr.open_mfdataset(
            i_files, concat_dim = 'time', 
            decode_times = False, combine = 'nested',
            )
    if len(dset.time) == 690:
      dset = dset[[
          "ucomp",
          "vcomp",
          "test_tracer",
          "mars_solar_long",
          "ps","t_surf",
          "temp",
          "omega",
          #"height",
          "lh_rel","dt_tg_lh_condensation",
      ]].squeeze()
      dset['pfull'] = dset.pfull*100
      #print(dset.shape)
      dset = dset.transpose('lat','lon','pfull','time')
      w = windx.VectorWind(dset.ucomp.fillna(0), dset.vcomp.fillna(0))
      
      dset["grdStr"] = calc_grdS(w, dset.test_tracer.fillna(0))
      print("Calculated gradient squared of test_tracer")
      theta, PV_isobaric = calculate_PV_Isca.calculate_PV(dset)
      print("Calculated PV")
      dset['theta'] = theta
      dset['PV']    = PV_isobaric

      dset["grdSpv"] = calc_grdS(w, dset.PV.fillna(0))

      return dset

def iterate_over_all(eps):
    gamma = [0.00,0.093]
    i = eps
    for j in gamma:
      #for j in gamma:
        exp_name = 'tracer_soc_mars_mola_topo_lh_eps_%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (i, j)
        print(exp_name)
        if os.path.isfile(path+exp_name+'/run0023/atmos_daily_interp.nc'):
            
            if not os.path.isfile(path+exp_name+'/atmos.nc'):
                print('starting')
              #try:
                _, _, i_files = funcs.filestrings(exp, path, 1, 403, p_file)
                dset = xr.open_mfdataset(
                    i_files, concat_dim = 'time', 
                    decode_times = False, combine = 'nested',
            )
              #except:
              #  continue



if __name__ == "__main__":
    
    eps = [15,20,25,30,35,40,45,50,10]

    for i in eps:
        iterate_over_all(i)
    for i in eps:
        interp_all(i)
    #append_all_era5()
#
    #with Pool(processes=len(eps)) as pool:
        
    #    pool.map(interp_all, eps)
    #    pool.map(iterate_over_all, eps)
        
    
# %%
