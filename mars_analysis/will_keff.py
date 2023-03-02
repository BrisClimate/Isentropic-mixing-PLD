# %%

from pyproj import Proj
from shapely.geometry import shape
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.mlab as mlab
from scipy.interpolate import griddata
import pylab
import sys
sys.path.append('/user/home/xz19136/Py_Scripts/Paper_scripts/')
import analysis_functions as funcs

rmars = 3.3962e6
rearth = 6.371e6

def pstere_proj(ds,timestep=600,R=rearth, hem='nh'):
    if hem == 'nh':
        lat0 = 90
        lats = slice(0,90)
    else:
        lat0 = -90
        lats = slice(-90,0)
    d = ds.test_tracer.isel(time=timestep).sel(lat = lats)

    pa = Proj("+proj=stere +lat_0=%i +R=%5f +lon_wrap=180" % (lat0,R),preserve_units=True)
    lonv, latv = np.meshgrid(d.lon.data, d.lat.data)

    x, y = pa(lonv,latv)
    reg_x = np.linspace(np.nanmin(x),np.nanmax(x),200)
    reg_y = np.linspace(np.nanmin(y),np.nanmax(y),200)    
    #xi, yi = np.mgrid[np.min(x):np.max(x):100j, np.min(y):np.max(y):100j]
    xi, yi = np.meshgrid(reg_x, reg_y)
    d2 = griddata((x.flatten(),y.flatten()),d.values.flatten(),(xi,yi),method='linear')
    
    return d2, xi, yi

def calc_keff(c, dx, dy, rac, timestep, ds, lats=np.linspace(30,88,2),R=rearth,hem='nh'):

    A = 2*np.pi*(R**2)*(1-np.sin(np.deg2rad(lats)))
    Ncon = len(lats)
    cons = ds.test_tracer.isel(time = timestep).mean(dim='lon').interp(lat=lats)
    #.data
    grad_c = np.gradient(c)
    grad_c_y = grad_c[0] / dy
    grad_c_x = grad_c[1] / dx
    # we need to integrate |grad c|^2, this makes it easy
    grad_c2xRAC = np.nan_to_num(grad_c_x**2 + grad_c_y**2, nan=0.) * np.ma.filled(rac,0.)
    grad_c2xRAC = np.where(np.isfinite(grad_c2xRAC), grad_c2xRAC, 0.)
    A_C_lin = np.zeros(Ncon) # the area inside each contour
    grad_c2_C_lin = np.zeros(Ncon) # integral of c^2 inside contour
    for n in range(Ncon):
        mask = np.ma.masked_where(c <= cons[n].values, c).mask
        
        A_C_lin[n] = np.ma.masked_array(rac, mask=mask).sum()
        #plt.imshow(np.ma.masked_array(rac, mask=mask))
        #plt.show()
        
        
        grad_c2_C_lin[n] = np.ma.masked_array(grad_c2xRAC, mask=mask).sum()
    
    #C = np.interp(A, A_C_lin, cons)
    #X = np.interp(A, A_C_lin, grad_c2_C_lin)

    #dCdA = np.gradient(C) / np.gradient(A)
    #dXdA = np.gradient(X) / np.gradient(A)
    dCdA = np.gradient(cons.values, A_C_lin)

    dXdA = np.gradient(grad_c2_C_lin, A_C_lin)
    
    Le2 = dXdA / (dCdA**2)

    Eqlat = np.rad2deg(np.arcsin(1 - A_C_lin/(2*np.pi*(R**2))))
    if hem == 'sh':
        Eqlat = -Eqlat
    keff = Le2/((2*np.pi*R*np.cos(np.deg2rad(Eqlat)))**2)

    return Eqlat, keff

if __name__ == '__main__':

    kt = ['2.0']#['0.0','0.4','1.0','2.0']#['0.0','1.0','2.0']
    dq = ['','-dq20.2']
    lat = [65]
    res = 170
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    #ax2 = fig.add_subplot(212)
    
    ax1.set_ylabel('$\kappa_{\mathrm{eff}}/\kappa$',fontsize=12)
    ax1.set_xlabel('$\phi_e$',fontsize=12)
    ax1.set_xlim(-90,90)
    #ax2.set_xlim(40,90)
    #ax1.set_ylim(1,30)
    #ax2.set_ylim(0.0,1.9)
    ax2.tick_params('y',colors='red')
    ax2.set_ylabel('tracer mix ratio', color='red')

    path     = '/user/work/xz19136/Isca_data/'
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
    #exp_name = 'hs_om_scale_1.00000_eps_0_0'
    p_file   = 'atmos_daily.nc'
    #exp_name = 'tracer_held_suarez_default'
    #p_file   = 'atmos_monthly.nc'
    ls = ['--','-',':']

    R = rmars

    timesteps = range(105,115)
    #timesteps=[450]
    lats = [np.arange(-89,-1,1),np.arange(1,89,1)]
    hems = ['sh','nh']

    for exp_name in exps:
        _, _, i_files = funcs.filestrings(exp_name, path, 0, 403, p_file)
        ds = xr.open_mfdataset(
            i_files, concat_dim = 'time', 
            decode_times = False, combine = 'nested',
            )
        ds = ds.sel(pfull = 0.2, method="nearest")
        ds = ds.sortby('lat',ascending=True)

        #ds = ds.sortby("lat", ascending=False)
        titles = ['$q_p = 0.2 \cdot 2\Omega/H$','$q_p = 0.7 \cdot 2\Omega/H$','$q_p = 2\Omega/H$']

        
        #lats = [55]
        keffi  = np.zeros((2, len(lats[0]), len(timesteps)))
        eqlat = []


        for timestep in timesteps:
            for i in [0,1]:
                c, xi, yi = pstere_proj(ds,timestep=timestep,R=R,hem=hems[i])

                dx = np.ones_like(c) * np.diff(xi)[0,0]

                dy = np.ones_like(c) * np.diff(xi)[0,0]
                #print(np.diff(yi).shape)
                rac = np.ones_like(c) * np.diff(xi)[0,0] * np.diff(xi)[0,0]
                Eqlat, keff = calc_keff(c, dx, dy, rac, timestep, ds, lats=lats[i], hem=hems[i])

                keff1 = [x for _,x in sorted(zip(Eqlat,keff))]
                Eqlat = sorted(Eqlat)

                keffi[i,:,timesteps.index(timestep)] = np.interp(lats[i],Eqlat,keff1)

        keffi = np.ma.masked_array(keffi, mask = np.isnan(keffi))
        k = np.append(keffi[0,:,:],keffi[1,:,:], axis=0)
        lats = np.append(lats[0],lats[1])
        #for i in range(len(k[0,:])):
        ax1.plot(lats, np.mean(k,axis=1),color='k',ls=ls[0],label=titles[0])
        ax2.plot(ds.coords['lat'].data,
            ds.test_tracer.mean(dim='lon')[timesteps[0]:timesteps[-1],:].mean(dim='time').data,
            color='r',ls=ls[0])

        #ax1.legend()
    plt.savefig('/user/home/xz19136/keff.png')
    plt.show()
# %%
print(keff1)
print(Eqlat)
print(lats)
# %%
