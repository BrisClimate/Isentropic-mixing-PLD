# %%
import xarray as xr
import numpy as np
import sys

from cartopy import crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.geodesic import Geodesic
from pyproj import Geod
from shapely.geometry import LineString, Polygon
import string
from matplotlib import (cm, colors, cycler)
import matplotlib.pyplot as plt

sys.path.append('/user/home/xz19136/Py_Scripts/Paper_scripts/')
#sys.path.append('/user/home/xz19136/Py_Scripts/mars_analysis/')

#from xcontour.xcontour import latitude_lengths_at, Contour2D, add_latlon_metrics

import matplotlib
import matplotlib.pyplot as plt
import analysis_functions as funcs


from skimage import measure

rmars = 3.3962e6
rearth = 6.378e6

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
    proj   = kwargs.pop(  'proj', ccrs.PlateCarree())
    levels = kwargs.pop('levels', np.linspace(np.min(z),np.max(z),50))

    close  = kwargs.pop( 'close', True)
    proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(nrows=1,ncols=1,
                            subplot_kw = {'projection': proj})
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
    return x

#def equivalent_latitude(dat, r = rearth):
#    '''
#    Given a contour, calculate its equivalent latitude (currently
#    calculating mean latitude)
#    '''
#    phi_e = []
#    for i in range(len(dat)):
#        phi = np.mean(dat[i][:,1])
#        phi_e.append(phi)
#    phi_e = np.mean(phi_e)
#    return phi_e

def effective_diffusivity(dat, **kwargs):
    '''
    Calculate effective diffusivity keff from equivalent length

    Inputs
    ------
    L_eq   : equivalent length,
    lat_eq : equivalent latitude,
    r      : radius,      optional, default r=r_e=m
    D      : diffusivity, optional, default D = 3.24 x 10^5 m^2 s^-1

    '''

    r   = kwargs.pop(  'r', rearth)
    D   = kwargs.pop(  'D', 3.24e5)
    hem = kwargs.pop('hem', 'nh')

    A, Leq =   equivalent_length(  dat, r = r)    # calculate its equivalent length
    phi_e  = equivalent_latitude(    A, r = r)    # calculate its equivalent latitude
    if np.shape(np.shape(dat))[0] == 1:
        l = []
        for i in range(len(dat)):
            for a in dat[i]:
                l.append(a)
    else:
        l = np.mean(dat,axis=1)

    if np.mean(l,axis=0)[1] < 0:
            phi_e = -phi_e
        
    a, L = length_0(phi_e, r = r)    # calculate its actual length
    #print('area 1 %5e area 2 %5e ' % (A, a))
    #print(Leq/L)
    keff = D*Leq**2/L**2
    if Leq < L:
        print(phi_e,np.mean(l,axis=0)[1])

    return keff, phi_e

def equivalent_length(dat, **kwargs):
    '''
    Calculate the equivalent length of a contour
    
    Inputs
    ------
    L : length
    gradC : gradient of tracer concentration
    '''
    r = kwargs.pop('r', rearth)
    #x = line_av(np.abs(1/gradC))
    #y = line_av(np.abs(gradC))

    #L_eq_sq = L**2*x*y
    #return np.sqrt(L_eq_sq)
    myGeod = Geod('+a=%6f +f=0' % r,sphere=True)
    #making my list of latlon (in decimal degrees) into a shapely
    ls = 0
    a  = 0

    for i in range(len(dat)):
        shapelyObject = LineString(dat[i])
        A, Leq = myGeod.geometry_area_perimeter(
            Polygon(shapelyObject))
        #if np.sign(A) == -1:
        #    print(Leq)
        #    Leq = - Leq
        ls += Leq
        a  += A

    return a, ls


def equivalent_latitude(A, **kwargs):
    '''
    Inputs
    ------
    A : area above contour
    r : radius, optional
    '''

    r = kwargs.pop('r', rearth)
    sinphi = 1 - np.abs(A)/(2*np.pi*r**2)
    phi_e = np.arcsin(sinphi)
    return np.rad2deg(phi_e)

def length_0(phi_e, **kwargs):
    '''
    Given an equivalent latitude, calculate the actual length
    '''
    r = kwargs.pop('r', rearth)
    lon = np.linspace(0,360,181)
    lat = np.linspace(-90,90,181)
    latlon, _ = np.meshgrid(lat, lon)
    ds = get_contours(lat,lon,np.transpose(latlon), levels=[-90,phi_e,90], close=True)
    
    a, L = equivalent_length(ds[0], r=r)
    return a, L #2*np.pi*r*np.cos(np.deg2rad(phi_e))
    

def PV_keff_test():
    
    dset = xr.open_dataset(
        'PV.nc', #concat_dim = 'time', 
        #decode_times = False, combine = 'nested',
        )
    dset = dset.where(dset.latitude > 0, drop = True)
    dset = dset.where( dset.level > 400, drop = True)
    
    tracer = dset.pv*10**4
    
    lats = np.linspace(0,90, 91)
    x = tracer.longitude.values
    y = tracer.latitude.values
    level = tracer.level.values
    r = rearth
    phi_full = []
    kap_full = []
    for j in level:
        ds = get_contours(y,x,tracer.sel(level=j))
        phi = []
        kap = []
        for i in range(len(ds)):                   # iterate over all tracer values
            dat = ds[i]                            # select contour for one tracer value
            A, Leq = equivalent_length(dat,r=rearth)  # calculate its equivalent length
            phi_e = equivalent_latitude(A, r=rearth)       # calculate its equivalent latitude
            L = actual_length(phi_e,r=rearth)      # calculate its actual length
            kappa = effective_diffusivity(Leq, phi_e,r=rearth,D=1)
            #print(phi_e,kappa)
            phi.append(phi_e)
            kap.append(kappa)
        phi_full.append(phi)
        kap_full.append(kap)
    
    interpd = np.zeros((len(lats),len(level)))
    for j in range(len(level)):
        x = phi_full[j]
        y = kap_full[j]
        #print(y)
        f = np.interp(lats,x,y)
        for i in range(len(lats)):
            interpd[i,j] = f[i]
    plt.clf()
    plt.contourf(lats,level,np.log(np.transpose(interpd)),
            levels = np.arange(0,4.7,0.2),cmap='jet')
    plt.colorbar()
    # we now have phi_full which has an array of equiv lat on each pressure level
    # and kap_full which has the corresponding array on each pressure level


    #r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
    #contours = measure.find_contours(tracer, 0.8)

def mars_keff_zonalmean(exp_name):
    path     = '/user/work/xz19136/Isca_data/'
    
    p_file   = 'atmos_daily.nc'
    #exp_name = 'tracer_held_suarez_default'
    #p_file   = 'atmos_monthly.nc'

    _, _, i_files = funcs.filestrings(exp_name, path, 20, 23, p_file)
    dset = xr.open_mfdataset(
        i_files, concat_dim = 'time', 
        decode_times = False, combine = 'nested',
        )

    #dset = dset.where(dset.lat>0,drop=True)
    dset = dset.where(dset.pfull<5.5,drop=True)
    dset = dset.where(dset.pfull>0.01,drop=True)

    tracer = dset.test_tracer#.sel(time=dset.time[-350:])
    x = tracer.lon.values
    y = tracer.lat.values
    level = tracer.pfull.values
    tracer = tracer.mean(dim="time")
    
    lats = np.linspace(-90,90, 181)
    r = rmars
    phi_full = []
    kap_full = []
    for j in level:
        ds = get_contours(y,x,tracer.sel(pfull=j))
        phi = []
        kap = []
        for i in range(len(ds)):                   # iterate over all tracer values
            dat = ds[i]
            kappa, phi_e = effective_diffusivity(dat, r=r,D=1)
            #print(phi_e,kappa)
            phi.append(phi_e)
            kap.append(kappa)
        
        phi = np.array(phi)
        kap = np.array(kap)
        inds = phi.argsort()
        phi_full.append(phi[inds])
        kap_full.append(kap[inds])

    interpd = np.zeros((len(lats),len(level)))
    for j in range(len(level)):
        x = phi_full[j]
        y = kap_full[j]
        #print(y)
        f = np.interp(lats,x,y)
        for i in range(len(lats)):
            interpd[i,j] = f[i]
    plt.clf()
    
    
    cf = axs.contourf(lats,level,np.transpose(interpd),
            levels = np.linspace(np.min(interpd),np.max(interpd),20),cmap='jet')
    axs.set_ylim([6,0.01])
    axs.set_yscale("log")
    plt.colorbar(cf)

def calculate_keff(dset, r = rearth):

    phi = []
    kap = []
    lats = np.linspace(-90,90,64)


    tracer = dset
    hem = 'nh'
    try:
        x = tracer.lon.values
        y = tracer.lat.values
    except:
        x = tracer.longitude.values
        y = tracer.latitude.values



    ds = get_contours(y,x,tracer,close=False)
    
    for i in range(len(ds)):                   # iterate over all tracer values
        
        dat = ds[i]                            # select contour for one tracer value

        if len(dat) > 3:
            continue
        kappa, phi_e = effective_diffusivity(dat, r=r, D=1, hem=hem)

        phi.append(phi_e)
        kap.append(kappa)

    kap = [x for _,x in sorted(zip(phi,kap))]
    phi = sorted(phi)
    keff = np.interp(lats,phi,kap)
    #phi = np.array(phi)
    #kap = np.array(kap)
    #inds = phi.argsort()
    plt.clf()
    #plt.plot(phi[inds],kap[inds])
    #plt.plot(lats,keff)
    plt.plot(phi, kap)
    try:
        plt.plot(dset.lat.values,dset.mean(dim="lon").values)
    except:
        plt.plot(dset.latitude.values,dset.mean(dim="longitude").values)
    return lats, keff

if __name__ == "__main__":
    dset = xr.open_dataset(
        'PV.nc', #concat_dim = 'time', 
        #decode_times = False, combine = 'nested',
        )
    #dset = dset.where(dset.latitude > 0, drop = True)
    dset = dset.sel(level = 500, method='nearest')

    dset = dset.pv*10**4
    phi, kap = calculate_keff(dset, r = rearth)


    exps = [
        'tracer_soc_mars_mola_topo_lh_eps_15_gamma_0.060_cdod_clim_scenario_7.4e-05',
        #'tracer_soc_mars_mola_topo_lh_eps_15_gamma_0.065_cdod_clim_scenario_7.4e-05',
        #'tracer_soc_mars_mola_topo_lh_eps_15_gamma_0.070_cdod_clim_scenario_7.4e-05',
        #'tracer_soc_mars_mola_topo_lh_eps_15_gamma_0.075_cdod_clim_scenario_7.4e-05',
        #'tracer_soc_mars_mola_topo_lh_eps_30_gamma_0.060_cdod_clim_scenario_7.4e-05',
        #'tracer_soc_mars_mola_topo_lh_eps_30_gamma_0.065_cdod_clim_scenario_7.4e-05',
        #'tracer_soc_mars_mola_topo_lh_eps_30_gamma_0.070_cdod_clim_scenario_7.4e-05',
        #'tracer_soc_mars_mola_topo_lh_eps_30_gamma_0.075_cdod_clim_scenario_7.4e-05',
    ]
    #mars_keff(0.2)
    pr = 0.2
    for exp_name in exps:
        path     = '/user/work/xz19136/Isca_data/'
    
        p_file   = 'atmos_daily.nc'
        #exp_name = 'tracer_held_suarez_default'
        #p_file   = 'atmos_monthly.nc'

        _, _, i_files = funcs.filestrings(exp_name, path, 0, 403, p_file)

        dset = xr.open_mfdataset(
            i_files, concat_dim = 'time', 
            decode_times = False, combine = 'nested',
            )
        dset = dset.test_tracer.sel(pfull=pr, method='nearest')
        timesteps = range(107,111)
        dset = dset.isel(time=107)#.mean(dim="time")
        phi, kap = calculate_keff(dset, r = rmars)
# %%
# %%
