# %%
import xarray as xr
import numpy as np
import sys, os
import math

sys.path.append('../')
import atmospy
import string

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import (cm, colors, cycler)
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# -*- coding: utf-8 -*-s
import numpy as np
#from calendar_calc import day_number_to_date
from netCDF4 import Dataset, date2num
import sys
import pdb
import os

def create_grid(manual_grid_option, new_dust):

    if(manual_grid_option):

        lons =  new_dust.lon.values

        lats =  new_dust.lat.values

        nlon=len(lons)
        nlat=len(lats)


    else:
        t_res=42

        resolution_file = Dataset('t'+str(t_res)+'.nc', 'r', format='NETCDF3_CLASSIC')

        lons = resolution_file.variables['lon'][:]
        lats = resolution_file.variables['lat'][:]


        nlon=lons.shape[0]
        nlat=lats.shape[0]


    return lons,lats,nlon,nlat


def create_pressures():

    p_full = [300.,900.]
    p_half=[0.,600.,1200.]

#    p_full = [0.017223, 0.078654, 0.180749, 0.375337, 0.71314, 1.253787, 2.060447, 3.193908, 4.707791, 6.646065, 9.04322, 11.926729, 15.321008, 19.251999, 23.751706, 28.862245, 34.639281, 41.154864, 48.49985, 56.786085, 66.14858, 76.747823, 88.772421, 102.442154, 118.011585, 135.77429, 156.067832, 179.279569, 205.853424, 236.297753, 271.194477, 311.209677, 357.105858, 409.756139, 470.160652, 539.465462, 618.984358, 710.223896, 814.912138, 935.031579, 1100.]

#    p_half = [0., 0.046817, 0.115544, 0.254986, 0.510238, 0.937468, 1.599324, 2.558913, 3.874132, 5.593891, 7.756986, 10.393581, 13.528609, 17.186225, 21.394476, 26.18965, 31.62002, 37.748946, 44.657459, 52.446517, 61.23914, 71.182614, 82.450941, 95.247663, 109.809174, 126.408616, 145.360464, 167.025894, 191.81904, 220.214288, 252.754733, 290.062007, 332.84765, 381.926291, 438.230874, 502.830259, 576.949493, 661.993148, 759.57212, 871.53435, 1000., 1200.]
    
    if(np.min(p_half)!=0.):
        print('Must have minimum p_half level = 0., as otherwise model data will be missing near the top levels.')
        sys.exit(0)

    if(np.max(p_half)<=1000.):
        print('Must have maximum p_half level > 1000., as otherwise vertical interpolation will fail near the surface when p_surf>1000.')
        sys.exit(0)

    npfull=len(p_full)
    nphalf=len(p_half)

    return p_full,p_half,npfull,nphalf


def create_time_arr(num_years,is_climatology,time_spacing):

    if(is_climatology):
        if(num_years!=1.):
            print('note that for climatology file only one year is required, so setting num_years=1.')
        num_days=669.
        num_years=1.
#        time_spacing=num_days//10
        day_number = np.linspace(0,num_days,time_spacing+1)[1:]-(num_days/(2.*time_spacing))
        
        print(day_number)

        time_units='days since 0000-00-00 00:00:00.0'
        print('when creating a climatology file, the year of the time units must be zero. This is how the model knows it is a climatology.')
    else:
        num_days=num_years*360.
#        time_spacing=num_years
        day_number = np.linspace(0,num_days,time_spacing+1)
        time_units='days since 0001-01-01 00:00:00.0'

    
    ntime=len(day_number)

    return day_number,ntime,time_units


def output_to_file(data,lats,lons,p_full,p_half,time_arr,time_units,file_name,variable_name,number_dict,time_bounds=None):

    output_file = Dataset(file_name, 'w', format='NETCDF3_CLASSIC')

    if p_full is None:
        is_thd=False
    else:
        is_thd=True

    lat = output_file.createDimension('lat', 36)
    lon = output_file.createDimension('lon', 72)


    if is_thd:
        pfull = output_file.createDimension('pfull', number_dict['npfull'])
        phalf = output_file.createDimension('phalf', number_dict['nphalf'])

    time = output_file.createDimension('time', 0) #s Key point is to have the length of the time axis 0, or 'unlimited'. This seems necessary to get the code to run properly. 

    latitudes = output_file.createVariable('lat','d',('lat',))
    longitudes = output_file.createVariable('lon','d',('lon',))
    
    if is_thd:
        pfulls = output_file.createVariable('pfull','d',('pfull',))
        phalfs = output_file.createVariable('phalf','d',('phalf',))

    times = output_file.createVariable('time','d',('time',))

    latitudes.units = 'degrees_N'.encode('utf-8')
    latitudes.cartesian_axis = 'Y'
    latitudes.long_name = 'latitude'

    longitudes.units = 'degrees_E'.encode('utf-8')
    longitudes.cartesian_axis = 'X'
    longitudes.long_name = 'longitude'

    if is_thd:
        pfulls.units = 'hPa'
        pfulls.cartesian_axis = 'Z'
        pfulls.positive = 'down'
        pfulls.long_name = 'full pressure level'

        phalfs.units = 'hPa'
        phalfs.cartesian_axis = 'Z'
        phalfs.positive = 'down'
        phalfs.long_name = 'half pressure level'


    times.units = time_units
    times.calendar = 'NO_CALENDAR'
    times.calendar_type = 'NO_CALENDAR'
    times.cartesian_axis = 'T'



    if is_thd:
        output_array_netcdf = output_file.createVariable(variable_name,'f4',('time','pfull', 'lat','lon',))
    else:
        output_array_netcdf = output_file.createVariable(variable_name,'f4',('time','lat','lon',))

    print(latitudes)
    print(lats)
    latitudes[:] = lats
    longitudes[:] = lons

    if is_thd:
        pfulls[:]     = p_full
        phalfs[:]     = p_half

    if type(time_arr[0])!=np.float64 and type(time_arr[0])!=np.int64 :
        times[:]     = date2num(time_arr,units='days since 0001-01-01 00:00:00.0',calendar='360_day')
    else:
        times[:]     = time_arr

    output_array_netcdf[:] = data

    output_file.close()

def check_Ls():
    dclim = xr.open_dataset(
        '/user/work/xz19136/Isca_data/' + \
        'soc_mars_mola_topo_lh_eps_25_gamma_0.093_' + \
        'cdod_clim_scenario_7.4e-05/run0001/atmos_monthly.nc',
        decode_times = False
    )

def reorder_dustclim():
    #dclim = xr.open_dataset(
    #    '/user/home/xz19136/Isca/exp/socrates_mars/input/cdod_clim_scenario.nc',
    #    decode_times = False)
    #dclim = dclim.isel(time=slice(0,669))

    disca = xr.open_mfdataset(
        '/user/work/xz19136/Isca_data/soc_mars_mk36_per_value70.85_none_mld_2.0/' + \
            'run00[012]*/atmos_daily.nc',
        decode_times=False
        )
    
    disca = disca.sel(time=disca.time[0:669])
    disca["mars_solar_long"] = disca.mars_solar_long.where(
        disca.mars_solar_long!=3.54378082e+02,other=359.776).squeeze()
    itime = disca.time
    
    lsisca = disca.mars_solar_long.values

    dclim = xr.open_dataset(
        '/user/home/xz19136/dust_clim.nc',
        decode_times = False)
    
    
    
    #dclim = dclim.mean(dim="longitude")
    Ls = dclim.Ls
    ls1 = dclim.where(Ls>=180,drop=True)
    ls1 = ls1.assign_coords(Time = ls1.Time - ls1.Time[0])
    ls2 = dclim.where(Ls< 180,drop=True)
    ls2 = ls2.assign_coords(Time = ls2.Time + ls2.Time[-1])
    ### swap back ###
    newdata = [ls1,ls2]
    dnew = xr.concat(newdata, dim="Time")
    dnew = dnew.rename({'Time':'time'})
    time = dnew.time
    #dnew = dnew.assign_coords(time=(dnew.Ls))

    ### uncomment ###
    #dnew = dnew.interp({'time':lsisca},'quadratic',kwargs={'fill_value':'extrapolate'})
    dnew = dnew.assign_coords(time=(itime))
    fig, axs = plt.subplots(nrows=1,ncols=2, figsize = (8,4),)

    lims = [0,1]

    boundaries, cmap, norm = atmospy.new_cmap(lims, extend='max', i = 10, override=True, cols='OrRd')
    for i, ax in enumerate(fig.axes):
        ax.text(0, 1.05, string.ascii_lowercase[i]+')', transform=ax.transAxes, 
            size='large')
        
        ax.set_ylim([-90,90])
        ax.set_xlabel('Sol')

    axs[0].set_title('old')
    axs[1].set_title('new')

    c1=axs[0].contourf(dclim.Time, dclim.latitude, dclim.cdod.mean(dim="longitude").squeeze().transpose(),
                cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])
    #c2=axs[1].contourf(dnew.time, dnew.latitude, dnew.cdod.mean(dim="longitude").squeeze().transpose(),
    #            cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])
    #print(len(dnew.time))
    #d1 = dnew.where(dnew.time < dnew.time.isel(time=30*15),drop=True)
    #d2 = dnew.where(dnew.time >= dnew.time.isel(time=30*16),drop=True)
    #d3 = dnew.isel(time=slice(30*15,30*16,None))
    #d3 = d3.mean(dim="longitude").expand_dims({'longitude':d2.longitude})
    #print(len(d1.time)+len(d2.time)+len(d3.time))
#
    #data = [d1, d3, d2]
    #data =xr.concat(data, dim="time")
    #print(len(data.time))


    data = []
    t0 = dnew.time[0].values
    t1 = dnew.time[-1].values
    data.append(dnew.isel(time=0).assign_coords(time=0.0))
    for i in range(10):
        data.append(dnew)
        dnew = dnew.assign_coords(time=(dnew.time+t0+t1))
    
    data = xr.concat(data, dim="time")
    data = data.rename({'latitude':'lat',
                        'longitude':'lon'})
    
    c2=axs[1].contourf(data.time, data.lat, data.cdod.mean(dim="lon").squeeze(),
                cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])
    
    data = data.assign_coords(lon=(data.lon % 360))
    
    data = data.sortby('lon', ascending=True)

    return data.cdod

def save_data(new_dust):
    manual_grid_option=True

    lons,lats,nlon,nlat=create_grid(manual_grid_option, new_dust)
    #create times
    is_climatology=True
    num_years=1
    time_spacing=num_years

    day_number,ntime,time_units=create_time_arr(num_years,is_climatology, time_spacing)
    #create time series based on times
    cdod = np.zeros((ntime, nlat, nlon))

    cdod = new_dust.transpose('time','lat','lon').values

    #Output it to a netcdf file. 
    file_name='clim_latlon.nc'
    variable_name='cdod'

    number_dict={}
    number_dict['nlat']=nlat
    number_dict['nlon']=nlon
    number_dict['ntime']=ntime

    time_arr=new_dust.time.values

    p_full=None
    p_half=None

    output_to_file(cdod,lats,lons,p_full,p_half,time_arr,time_units,file_name,variable_name,number_dict)

#%%
def plot_dust_lat():
    d = xr.open_dataset('clim_latlon.nc', decode_times = False)
    #d = d.isel(time=slice(14*30, 16*30,None))
    print(d.lon.values)
    
    boundaries, cmap, norm = atmospy.new_cmap([0,1], extend='max', i = 10, override=True, cols='OrRd')
    #for i in [22,23,24,25,26]:
    #    print(d.time.isel(time=30*15+i).values)
    #    print(np.isnan(d.cdod.isel(time=30*15+i)).count())
    #    plt.contourf(d.lon,d.lat,d.cdod.isel(time=30*15+i),
    #                 cmap=cmap, norm=norm,levels=[boundaries[0]-50]+boundaries+[boundaries[-1]+ 150])
    #    plt.show()
    disca = xr.open_mfdataset(
        '/user/work/xz19136/Isca_data/soc_mars_mola_topo_lh_eps_25_gamma_0.093_clim_latlon_7.4e-05/' + \
            'run*/atmos_daily.nc',
        decode_times=False
        )
    boundaries, cmap, norm = atmospy.new_cmap([-50,150], extend='max', i = 10, override=True, cols='OrRd')
    print(len(disca.time),15*30)
    ### in test output, time needs to be 170 or so long to get to buggy 15*30+24 
    for i in range(len(disca.pfull)):
        plt.contourf(disca.lon,disca.lat,
             disca.temp.isel(pfull=i).isel(time=15*30+25).squeeze())
        plt.show()
    plt.contourf(disca.time,disca.lat,
             disca.temp.sel(pfull=5,method="nearest").mean(dim="lon").transpose().squeeze())
    plt.show()

    #d = xr.open_dataset('../../Isca/exp/socrates_mars/input/cdod_clim_scenario.nc', decode_times = False)
    #print(d.lat.values)
plot_dust_lat()

#%%

if __name__ == "__main__":
    data = reorder_dustclim()

    save_data(data)

# %%
