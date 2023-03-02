'''
A selection of functions used in the analysis of OpenMARS and Isca data, for the given paper.
'''
# Written by Emily Ball, final version 16/03/2021
# Tested on Anthropocene

# NOTES and known problems
# Flexible to planetary parameters such as radius/gravity/rotation rate etc.

# Updates
# 16/03/2021 Upload to GitHub and commenting EB

# Begin script ========================================================================================================

import numpy as np
import xarray as xr
import dask
import os, sys
import pandas as pd
import scipy.stats as st
from scipy.signal import convolve2d
from cartopy.util import add_cyclic_point

import metpy.interpolate
from metpy.units import units
import windspharm.xarray as windx
#import time
import scipy.optimize as so
from metpy.calc.tools import (broadcast_indices, find_bounding_indices,
                              _less_or_close)

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import (cm, colors)
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#### planetary parameters, set for Mars. These will be the default values used
#### in any subsequent function

g       = 3.72076
p0      = 610. *units.Pa
kappa   = 1/4.4
omega   = 7.08822e-05
rsphere = 3.3962e6

eps = np.arange(10,55,5)
gamma = [0.093,0.00]

def get_exps(exps):
    
    exp_names = []
    if exps == 'parameter':
        titles = []
        for i in gamma:
            for j in eps:
                titles.append('$\epsilon = %i$,\n$\gamma = %.3f$' %(j,i))
                exp_names.append('tracer_soc_mars_mola_topo_lh_eps_' + \
                    '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' %(j,i))
        nrows = len(gamma)
        ncols = len(eps)
        

    elif exps == 'attribution':
        titles = ['Control', 'LH', 'D', 'LH+D', 'T', 'LH+T', 'D+T', 'LH+D+T']
        for t in ['', '_mola_topo']:
            for d in ['', '_cdod_clim_scenario_7.4e-05']:
                for l in ['', '_lh']:
                    exp_names.append('tracer_soc_mars%s%s_eps_25_gamma_0.093%s' % (t, l, d))
        nrows = 2
        ncols = 4

    elif exps == 'dust':
        titles = ['1/2', '1', '2', '4']
        for ds in [3.7e-5, 7.4e-5,1.48e-4,2.96e-4]:
            exp_names.append('tracer_soc_mars_mola_topo_lh_eps_' + \
                    '25_gamma_0.093_cdod_clim_scenario_%s' % str(ds))
        nrows = 1
        ncols = 4
    elif exps == '0-ecc':
        titles = []
        for j in eps:
            titles.append('$\epsilon = %i$' %(j))
            exp_names.append('tracer_soc_mars_mola_topo_lh_eps_' + \
                    '%i_gamma_0.000_cdod_clim_scenario_7.4e-05' %(j))
        nrows = 1
        ncols = len(eps)
    elif exps == 'curr-ecc':
        titles = []
        for j in eps:
            titles.append('$\epsilon = %i$' %(j))
            exp_names.append('tracer_soc_mars_mola_topo_lh_eps_' + \
                    '%i_gamma_0.093_cdod_clim_scenario_7.4e-05' %(j))
        nrows = 1
        ncols = len(eps)

    
    return exp_names, titles, nrows, ncols
  
def open_files(path, exp_name, isentropic=False, tname='test_tracer'):
    
    if not isentropic:
        isent = ''
    else:
        isent = '_isentropic'
    
    if tname is not 'test_tracer':
        tname = 'test_tracer'
    ds = xr.open_dataset(
        	path + exp_name + '/keff%s_%s.nc' % (isent, tname), decode_times = False,)

    d = xr.open_dataset(
        	path + exp_name + '/atmos%s.nc' % isent, decode_times = False,)
    
    return ds, d


def duplicate_axis(ax, newpos):
    ax2 = ax.twiny()
    ax2.set_xticks(newpos)
    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top') 
    ax2.tick_params(length = 6)
    ax2.set_xlim(ax.get_xlim())
    
    return ax2
 

def xr_add_cyclic_point(da):
    """
    Inputs
    da: xr.DataArray with dimensions (time,lat,lon)
    """

    # Use add_cyclic_point to interpolate input data
    lon_idx = da.dims.index('lon')
    wrap_data, wrap_lon = add_cyclic_point(da.values, coord=da.lon, axis=lon_idx)

    # Generate output DataArray with new data but same structure as input
    outp_da = xr.DataArray(data=wrap_data, 
                           coords = {'time': da.time, 'lat': da.lat, 'lon': wrap_lon}, 
                           dims=da.dims, 
                           attrs=da.attrs)

    return outp_da

def swap_lats(da):
    """
    Inputs
    ------
    da: xr.Dataset with dimensions including lat
    
    Outputs
    -------
    outp_da: xr.Dataset with lat dimension swapped
    """
    outp_da = da.assign_coords({'lat': - da.lat})
    return outp_da
    

def new_cmap(d, override=False, user_choice=False, **kwargs):
    '''
    Takes input data and calculates an appropriate colormap
    '''

    i = kwargs.pop('i', 20)
    extend = kwargs.pop('extend', 'both')

    vmin = np.nanmin(d)
    vmax = np.nanmax(d)
        
    if not user_choice:
        if np.sign(vmax) == np.sign(vmin):
            if np.sign(vmax) == 1:
                cols = 'Reds'
            else:
                cols = 'Blues_r'
        else:
            if override:
                if np.abs(vmax) > np.abs(vmin):
                    cols = 'Reds'
                    vmin = 0
                else:
                    cols = 'Blues_r'
                    vmax = 0
            else:
                cols = 'coolwarm'
                if np.abs(vmax) > np.abs(vmin):
                    vmin = -np.abs(vmax)
                    vmax = np.abs(vmax)
                else:
                    vmax = np.abs(vmin)
                    vmin = -np.abs(vmin)
    else:
        vmin = np.nanmin(d)
        vmax = np.nanmax(d)
        cols = 'Reds'

    diff = vmax - vmin
    
    cols = kwargs.pop('cols', cols)
    

    while int(diff)/i == 0:
        diff = diff*10
        i = i*10
    step = int(diff)/i
    #vmin = float("{:.5f}".format(step*int(vmin/step)))
    #vmax = float("{:.5f}".format(step*int(vmax/step)))
    vmin = float("{:.5f}".format(vmin))
    vmax = float("{:.5f}".format(vmax))
    boundaries, _, _, cmap, norm = make_colourmap(vmin, vmax+step/2, step,
                        col = cols, extend = extend)

    return boundaries, cmap, norm

def corr_masked(x,y):
    mask = np.isfinite(x) & np.isfinite(y)
    y = y[mask]
    x = x[mask]
    try: 
        i = np.corrcoef(y,x)
        ret = i[0,1]
    except:
        ret = np.nan
    return ret

def corr_masked_alpha(x,y,test):

    mask = np.isfinite(x) & np.isfinite(y)
    y = y[mask]
    x = x[mask]
    try:
        if test=='pearson':
            r, alpha = st.stats.pearsonr(y,x)
        elif test=='spearman':
            r, alpha = st.stats.spearmanr(y,x)
        else:
            raise ValueError('Unrecognized correlation test')
    except:
        r = np.nan
        alpha = 1.
    return r, alpha

def significant(x,y,**kwargs):
    ''''
    Mask array x by statistical significance to the level alpha.
    Input
    -----
    x: array-like, input data
    y: array-like, p-values
    alpha: limiting p-value for statistical significance, default = 0.05
    '''
    alpha = kwargs.pop('alpha', 0.05)
    ret = np.where(y < alpha, x, np.nan)
    return ret


def add_text(ax, data):
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if np.isfinite(data[y,x]):
                ax.text(
                    x + 0.5, y + 0.5, '%.2f' % data[y, x],
                    horizontalalignment='center',
                    verticalalignment='center',
                )
            else:
                continue
def add_sig_text(ax, data, sig,**kwargs):
    fontsize=kwargs.pop('fontsize','regular')
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if np.isfinite(data[y,x]):
                if np.isfinite(sig[y,x]):
                    weight = 'bold'
                    ast = '*'
                else:
                    weight = 'regular'
                    ast = ''
                ax.text(
                    x + 0.5, y + 0.5, '%.2f%s' % (data[y, x],ast),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontweight=weight,fontsize=fontsize,
                )
            else:
                continue

def calculate_pfull(psurf, siglev):
    '''
    Calculates full pressures using surface pressures and sigma coordinates

    psurf  : array-like
            Surface pressures
    siglev : array-like
            Sigma-levels
    '''

    return psurf*siglev

def calculate_pfull_EMARS(ps,bk,ak):
    '''
    Calculates full pressures using surface pressures and sigma coordinates

    psurf  : array-like
            Surface pressures
    siglev : array-like
            Sigma-levels
    '''
    p_i = ps*bk + ak

    p = xr.zeros_like()
    p[k] = (p_i[k+1]-p_i[k])/np.log(p_i[k+1]/p_i[k])
    return 

def calculate_theta(tmp, plevs, **kwargs):
    '''
    Calculates potential temperature theta

    Input
    -----
    tmp   : temperature, array-like
    plevs : pressure levels, array-like
    p0    : reference pressure, optional. Default: 610. Pa
    kappa : optional. Default: 0.25
    '''
    p0 = kwargs.pop('p0', 610.)
    kappa = kwargs.pop('kappa', 1/4.4)

    ret = tmp * (p0/plevs)**kappa
    return ret
 
def wrapped_gradient(da, coord, spacing=1.):
    '''
    Finds the gradient along a given dimension of a dataarray.
    '''

    dims_of_coord = da.coords[coord].dims
    if len(dims_of_coord) == 1:
        dim = dims_of_coord[0]
    else:
        raise ValueError('Coordinate ' + coord + ' has multiple dimensions: ' + str(dims_of_coord))

    coord_vals = da.coords[coord].values*spacing

    return xr.apply_ufunc(np.gradient, da, coord_vals, kwargs={'axis': -1},
                          input_core_dims=[[dim], [dim]],
                          output_core_dims=[[dim]],
                          dask='parallelized',
                          output_dtypes=[da.dtype])


def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


class nf(float):
    def __repr__(self):
        s = f'{self:.1f}'
        return f'{self:.0f}' if s[-1] == '0' else s


def lait(PV,theta,theta0, **kwargs):
    '''
    Perform Lait scaling of PV
    kwargs
    ------
    kappa: R/c_p, optional, defaults to 0.25.
    '''
    kappa = kwargs.pop('kappa', 0.25)
    ret = PV*(theta/theta0)**(-(1+1/kappa))

    return ret



def calc_eddy_enstr(q):
    '''
    Calculate eddy enstrophy
    -------------
    Input:
    q : xarray DataArray with dimensions "lat","lon","time"
    Output:
    Z : xarray DataArray with dimensions "time"
    '''
    q = q.where(q.lon < 359.5, drop = True)
    qbar = q.mean(dim='lon')
    qbar = qbar.expand_dims({'lon':q.lon})

    qprime = q - qbar
    
    cosi = np.cos(np.pi/180 * (q.lat))


    qp2 = qprime ** 2 * cosi
    qpi = qp2.sum(dim = "lat")
    qp = qpi.sum(dim = "lon")

    sumc = sum(cosi)
    
    Z = 1/(4*np.pi)* qp/sumc

    return Z
  
def calc_streamfn(lats, pfull, vz, **kwargs):
    '''
    Calculate meridional streamfunction from zonal mean meridional wind.
    
    Parameters
    ----------

    lats   : array-like, latitudes, units (degrees)
    pfull  : array-like, pressure levels, units (Pa)
    vz     : array-like, zonal mean meridional wind, dimensions (lat, pfull)
    radius : float, planetary radius, optional, default 3.39e6 m
    g      : float, gravity, optional, default 3.72 m s**-2

    Returns
    -------

    psi   : array-like, meridional streamfunction, dimensions (lat, pfull),
            units (kg/s)
    '''

    radius = kwargs.pop('radius', 3.39e6)
    g      = kwargs.pop('g', 3.72)

    coeff = 2 * np.pi * radius / g

    psi = np.empty_like(vz.values)
    for ilat in range(lats.shape[0]):
        psi[0, ilat] = coeff * np.cos(np.deg2rad(lats[ilat]))*vz[0, ilat] * pfull[0]
        for ilev in range(pfull.shape[0])[1:]:
            psi[ilev, ilat] = psi[ilev - 1, ilat] + coeff*np.cos(np.deg2rad(lats[ilat])) \
                              * vz[ilev, ilat] * (pfull[ilev] - pfull[ilev - 1])
    
    #psi = xr.DataArray(psi, coords = {"pfull" : pfull.values,
    #                                  "lat"   : lats.values})
    #psi.attrs['units'] = 'kg/s'

    return psi
  

def calc_jet_lat(u, lats):
    '''
    Function to calculate location and strength of maximum given zonal wind
    u(lat) field

    Parameters
    ----------

    u    : array-like
    lats : array-like. Default use will be to calculate jet on a given pressure
           level, but this array may also represent pressure level.

    Returns
    -------

    jet_lat : latitude (pressure level) of maximum zonal wind
    jet_max : strength of maximum zonal wind
    '''

    # Restrict to 10 points around maximum
    try:
        u_max = np.where(u == np.ma.max(u.values))[0][0]
    except:
        u_max = np.where(u == np.ma.max(u))[0][0]
    if u_max == 0:
        jet_lat = lats[u_max]
        jet_max = u[u_max]
    else:
        u_near = u[u_max-1:u_max+2]
        lats_near = lats[u_max-1:u_max+2]
        # Quartic fit, with smaller lat spacing
        coefs = np.ma.polyfit(lats_near,u_near,2)
        fine_lats = np.linspace(lats_near[0], lats_near[-1],200)
        quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2
        # Find jet lat and max
        jet_lat = fine_lats[np.where(quad == max(quad))[0][0]]
        jet_max = coefs[2]+coefs[1]*jet_lat+coefs[0]*jet_lat**2
        # Plot fit?
    
    return jet_lat, jet_max


#Converted to python by Paul Staten Jul.29.2017
def Calculate_ZeroCrossing(F, lat, lat_uncertainty=0.0):

  ''' Find the first (with increasing index) zero crossing of the function F

      Args:
  
        F: array

        lat: latitude array (same length as F)

        lat_uncertainty (float, optional): The minimal distance allowed between adjacent zero crossings of indetical sign change for example, for lat_uncertainty = 10, if the most equatorward zero crossing is from positive to negative, the function will return a NaN value if an additional zero crossings from positive to negative is found within 10 degrees of that zero crossing.

      Returns:

        float: latitude of zero crossing by linear interpolation
  '''
  # Make sure a zero crossing exists
  a = np.where(F > 0)[0]
  if len(a) == len(F) or not any(a):
    return np.nan

  # Find first zero crossing in index units.
  D = np.diff(np.sign(F))

  # If more than one zero crossing exists in proximity to the first zero crossing.
  a = np.where(np.abs(D)>0)[0]
  if len(a)>2 and np.abs(lat[a[2]] - lat[a[0]]) < lat_uncertainty:
    return np.nan

  a1 = np.argmax(np.abs(D) > 0)
  # if there is an exact zero, use its latitude...
  if np.abs(D[a1])==1:
    ZC = lat[a1]
  else:
    ZC = lat[a1] - F[a1]*(lat[a1+1]-lat[a1])/(F[a1+1]-F[a1])
  return ZC

def calc_Hadley_lat(u, lats, plot = False):
    '''
    Function to calculate location of 0 streamfunction.

    Parameters
    ----------

    u    : array-like
    lats : array-like. Default use will be to calculate jet on a given pressure
           level, but this array may also represent pressure level.

    Returns
    -------

    jet_lat : latitude (pressure level) of 0 streamfunction
    '''

    asign = np.sign(u)#.values)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    signchange[0] = 0

    

    for i in range(len(signchange)):
        if u[i] > 0 and i < len(signchange) - 4:
            continue
        signchange[i] = 0
    
    for i in range(len(signchange)):
        if signchange[i] == 0:
            continue
        u_0 = i
    
    if all(signchange[i] == 0 for i in range(len(signchange))):
        if u[0] > 0:
            u_0 = 0
        else:
            u_0 = -1

        #u_0 = np.where(u == np.ma.min(np.absolute(u)))[0][0]

    # Restrict to 10 points around maximum
    #u_0 = np.where(u == np.ma.min(np.absolute(u.values)))[0][0]
    if u_0 > 1:
        u_near = u[u_0-2:u_0+2]
        lats_near = lats[u_0-2:u_0+2]

        # Quartic fit, with smaller lat spacing
        coefs = np.ma.polyfit(lats_near,u_near,3)
        fine_lats = np.linspace(lats_near[0], lats_near[-1],300)
        quad = coefs[3]+coefs[2]*fine_lats+coefs[1]*fine_lats**2 \
                    +coefs[0]*fine_lats**3
        # Find jet lat and max
        #jet_lat = fine_lats[np.where(quad == max(quad))[0][0]]

        minq = min(np.absolute(quad))
        jet_lat = fine_lats[np.where(np.absolute(quad) == minq)[0][0]]
        jet_max = coefs[3]+coefs[2]*jet_lat+coefs[1]*jet_lat**2 \
                    +coefs[0]*jet_lat**3

    elif u_0 == 0 or u_0 == -1:
        jet_lat = lats[u_0]
        jet_max = u[u_0]

    elif u_0 == 1:
        u_near = u[u_0-1:u_0+2]
        lats_near = lats[u_0-1:u_0+2]

        # Quartic fit, with smaller lat spacing
        coefs = np.ma.polyfit(lats_near,u_near,2)
        fine_lats = np.linspace(lats_near[0], lats_near[-1],200)
        quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2 
        # Find jet lat and max
        #jet_lat = fine_lats[np.where(quad == max(quad))[0][0]]

        minq = min(np.absolute(quad))
        jet_lat = fine_lats[np.where(np.absolute(quad) == minq)[0][0]]
        jet_max = coefs[2]+coefs[1]*jet_lat+coefs[0]*jet_lat**2

    else:
        
        jet_lat = np.nan
        jet_max = np.nan

    return jet_lat, jet_max
  


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def moving_average_2d(x, w):
    return convolve2d(x, np.ones((1, w)), 'valid') / w
  
def filestrings(exp, filepath, start, end, filename, **kwargs):
    '''
    Generates lists of strings, for Isca runs.
    '''
    outpath = kwargs.pop('outpath', filepath)

    if start<10:
        st='000'+str(start)
    elif start<100:
        st='00'+str(start)
    elif start<1000:
        st='0'+str(start)
    else:
        st=str(start)

    if end<10:
        en='000'+str(end)
    elif end<100:
        en='00'+str(end)
    elif end<1000:
        en='0'+str(end)
    else:
        en=str(end)


    nfiles = end - start + 1
    infiles = []
    runs = []
    out = []

    for i in range(nfiles):
        run_no = start+i
        if run_no<10:
            run='run000'+str(run_no)
        elif run_no<100:
            run='run00'+str(run_no)
        elif run_no<1000:
            run='run0'+str(run_no)
        else:
            run='run'+str(run_no)

        if os.path.exists(outpath +'/'+exp+'/'+run+'/'+filename):
            runs.append(run)
            out.append(outpath +'/'+exp+'/'+run+'/'+filename)
            infiles.append(filepath+'/'+exp+'/'+run+'/'+filename)
        else:
            continue

    return runs, out, infiles
  
def stereo_plot():
    '''
    Returns variables to define a circular plot in matplotlib.
    '''
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    return theta, center, radius, verts, circle

def make_stereo_plot(ax, lats, lons, circle, **kwargs):
    '''
    Makes the polar stereographic plot and plots gridlines at choice of lats
    and lons.
    '''
    linewidth = kwargs.pop('linewidth', 1)
    linestyle = kwargs.pop('linestyle', '-')
    color = kwargs.pop('color', 'black')
    alpha = kwargs.pop('alpha', 1)

    gl = ax.gridlines(crs = ccrs.PlateCarree(), linewidth = linewidth,
                      linestyle = linestyle, color = color, alpha = alpha)

    ax.set_boundary(circle, transform=ax.transAxes)

    gl.ylocator = ticker.FixedLocator(lats)
    gl.xlocator = ticker.FixedLocator(lons)

def make_colourmap(vmin, vmax, step, **kwargs):
    '''
    Makes a colormap from ``vmin`` (inclusive) to ``vmax`` (exclusive) with
    boundaries incremented by ``step``. Optionally includes choice of color and
    to extend the colormap.
    '''
    col = kwargs.pop('col', 'viridis')
    extend = kwargs.pop('extend', 'none')

    boundaries = list(np.arange(vmin, vmax, step))

    if extend == 'both':
        cmap_new = cm.get_cmap(col, len(boundaries) + 1)
        colours = list(cmap_new(np.arange(len(boundaries) + 1)))
        cmap = colors.ListedColormap(colours[1:-1],"")
        cmap.set_over(colours[-1])
        cmap.set_under(colours[0])

    elif extend == 'max':
        cmap_new = cm.get_cmap(col, len(boundaries))
        colours = list(cmap_new(np.arange(len(boundaries))))
        cmap = colors.ListedColormap(colours[:-1],"")
        cmap.set_over(colours[-1])

    elif extend == 'min':
        cmap_new = cm.get_cmap(col, len(boundaries))
        colours = list(cmap_new(np.arange(len(boundaries))))
        cmap = colors.ListedColormap(colours[1:],"")
        cmap.set_under(colours[0])
    
    else:
        cmap_new = cm.get_cmap(col, len(boundaries))
        colours = list(cmap_new(np.arange(len(boundaries))))
        cmap = colors.ListedColormap(colours[:],"")

    norm = colors.BoundaryNorm(boundaries, ncolors = len(boundaries) - 1,
                               clip = False)

    return boundaries, cmap_new, colours, cmap, norm
  
def assign_MY(d):
    '''
    Calculates new MY for Isca simulations and adds this to input dataset.
    Also returns the indices of the time axis that correspond to a new MY.
    '''
    t = np.zeros_like(d.time)
    index=[]
    for i in range(len(t)-1):
        if d.mars_solar_long[i+1]<d.mars_solar_long[i]:
            print(d.mars_solar_long[i].values)
            print(d.mars_solar_long[i+1].values)
            t[i+1] = t[i]+1
            index.append(d.time[i])
        else:
            t[i+1] = t[i]
    t1 = xr.Dataset({"MY" : (("time"), t)},
                    coords = {"time":d.time})
    d = d.assign(MY=t1["MY"])
    return d, index

def make_coord_MY(x, index):
    x = x.where(x.time > index[0], drop=True)
    x = x.where(x.time <= index[-1], drop=True)

    N=int(np.max(x.MY))
    n = range(N)

    y = x.time[:len(x.time)//N]

    ind = pd.MultiIndex.from_product((n,y),names=('MY','new_time'))
    dsr = x.assign_coords({'time':ind}).unstack('time')
    #dsr = dsr.squeeze()

    return dsr, N, n
  
def calc_PV_max(PV, lats, plot=False):
    '''
    Function to calculate location and strenth of maximum given zonal-mean PV
    PV(height) field
    '''
    # Restict to 10 points around maximum
    PV_max = np.where(PV == np.ma.max(PV))[0][0]
    PV_near = PV[PV_max-1:PV_max+2]
    lats_near = lats[PV_max-1:PV_max+2]
    # Quartic fit, with smaller lat spacing
    coefs = np.ma.polyfit(lats_near,PV_near,2)
    fine_lats = np.linspace(lats_near[0], lats_near[-1],200)
    quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2
    # Find jet lat and max
    jet_lat = fine_lats[np.where(quad == max(quad))[0][0]]
    jet_max = coefs[2]+coefs[1]*jet_lat+coefs[0]*jet_lat**2
    # Plot fit?
    if plot:
        print (jet_max)
        print (jet_lat)
        plt.plot(lats_near, PV_near)
        plt.plot(fine_lats, quad)
        plt.show()

    return jet_lat, jet_max

def get_timeslice(tind, mean):
    if mean is not None:
        m = int(mean/2)
    else:
        m = 0
    while tind % 30 in [2,3,4]:
        tind += 1
    while tind % 30 in [0,1]:
        tind -= 1
    while (tind - m) % 30 in [0,1,2,3,4]:
        tind += 1
    while (tind + m) % 30 in [0,1,2,3,4]:
        tind -= 1

    return tind, m

def get_nth_sol(tind, n):
    if tind % 30 in np.arange(15):
        tind = tind - (tind % 30)
    else:
        tind = tind + 30 - (tind % 30)

    if n > 30:
        n = n % 30

    return tind, n

def get_init_sol(tind):
    if tind % 30 in np.arange(15):
        tind = tind - (tind % 30)
    else:
        tind = tind + 30 - (tind % 30)

    return tind
