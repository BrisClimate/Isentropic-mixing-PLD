# Written by Ori Adam Mar.21.2017
# Edited by Alison Ming Jul.4.2017
from __future__ import division
import numpy as np
import scipy as sp
from scipy import integrate
from scipy import interpolate
from scipy.signal import find_peaks
import math

def find_u850(U, lev=np.array([1])):
  '''TropD Eddy Driven Jet (EDJ) metric
       
     Latitude of maximum of the zonal wind at the level closest to the 850 hPa level
     
     Args:
       U (lat,lev) or U (lat,): Zonal mean zonal wind. Also takes surface wind 
       lat : latitude vector
       lev: vertical level vector in hPa units

     Returns:
       u (lat): Zonal mean zonal wind at 850hPa

  '''
  
  if len(lev) > 1:
    u = U[:,find_nearest(lev, 850)]
  else:
    u = np.copy(U)
  
  return u

def TropD_Metric_EDJ(U, lat, lev=np.array([1]), method='peak', n=0, n_fit=1):
  '''TropD Eddy Driven Jet (EDJ) metric
       
     Latitude of maximum of the zonal wind at the level closest to the 850 hPa level
     
     Args:
       U (lat,lev) or U (lat,): Zonal mean zonal wind. Also takes surface wind 
       lat : latitude vector
       lev: vertical level vector in hPa units

       method (str, optional): 'peak' (default) |  'max' | 'fit'
       
        peak (Default): Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=30)
        
        max: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=6)
        fit: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level using a quadratic polynomial fit of data from gridpoints surrounding the gridpoint of the maximum
        
       n (int, optional): If n is not set (0), n=6 (default) is used in TropD_Calculate_MaxLat. Rank of moment used to calculate the position of max value. n = 1,2,4,6,8,...  
     
     Returns:
       tuple: PhiSH (ndarray), PhiNH (ndarray) Latitude of EDJ in SH and NH

  '''

  try:
    assert (not hasattr(n, "__len__") and n >= 0)  
  except AssertionError:
   print('TropD_Metric_EDJ: ERROR : the smoothing parameter n must be >= 0')
   
  try:
    assert(method in ['max','peak'])
  except AssertionError:
    print('TropD_Metric_EDJ: ERROR : unrecognized method ', method)

  eq_boundary = 15
  polar_boundary = 70
  
  if len(lev) > 1:
    u = U[:,find_nearest(lev, 850)]
  else:
    u = np.copy(U)
    
  if method == 'max':
    if n:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
              lat[(lat > eq_boundary) & (lat < polar_boundary)], n=n)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)], n=n)

    else:
      #Default value of n=6 is used
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
              lat[(lat > eq_boundary) & (lat < polar_boundary)])
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)])
  
  elif method == 'peak':
    if n:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
              lat[(lat > eq_boundary) & (lat < polar_boundary)], n=n)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)], n=n)
    else:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
              lat[(lat > eq_boundary) & (lat < polar_boundary)],n=30)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)],n=30)
  
  elif method == 'fit':
    Uh = u[(lat > eq_boundary) & (lat < polar_boundary)]
    Lat = lat[(lat > eq_boundary) & (lat < polar_boundary)]
    m = np.nanmax(Uh)
    Im = np.nanargmax(Uh)
     
    if (Im == 0 or Im == len(Uh)-1):
      PhiNH = Lat[Im]
    
    elif (n_fit > Im or n_fit > len(Uh)-Im+1):
      N = np.min(Im, len(Uh)-Im+1)
      p = np.polyfit(Lat[Im-N:Im+N+1], Uh[Im-N:Im+N+1],2) 
      PhiNH = -p[1]/(2*p[0])
    else:
      p = np.polyfit(Lat[Im-n_fit:Im+n_fit+1], Uh[Im-n_fit:Im+n_fit+1],2) 
      PhiNH = -p[1]/(2*p[0])
    
    Uh = u[(lat > -polar_boundary) & (lat < -eq_boundary)]
    Lat = lat[(lat > -polar_boundary) & (lat < -eq_boundary)]
    
    m = np.nanmax(Uh)
    Im = np.nanargmax(Uh)
    
    if (Im == 0 or Im == len(Uh)-1):
      PhiSH = Lat[Im]
    
    elif (n_fit > Im or n_fit > len(Uh)-Im+1):
      N = np.min(Im, len(Uh)-Im+1)
      p = np.polyfit(Lat[Im-N:Im+N+1], Uh[Im-N:Im+N+1],2) 
      PhiSH = -p[1]/(2*p[0])
    else:
      p = np.polyfit(Lat[Im-n_fit:Im+n_fit+1], Uh[Im-n_fit:Im+n_fit+1],2) 
      PhiSH = -p[1]/(2*p[0])
  
  else:
    print('TropD_Metric_EDJ: ERROR: unrecognized method ', method)

  return PhiSH, PhiNH


def TropD_Metric_OLR(olr, lat, method='250W', Cutoff=50, n=int(6)):
  """TropD Outgoing Longwave Radiation (OLR) metric
     
     Args:
     
       olr(lat,): zonal mean TOA olr (positive)
       
       lat: equally spaced latitude column vector
        
       method (str, optional):

         '250W'(Default): the first latitude poleward of the tropical OLR maximum in each hemisphere where OLR crosses 250W/m^2
         
         '20W': the first latitude poleward of the tropical OLR maximum in each hemisphere where OLR crosses the tropical OLR max minus 20W/m^2
         
         'cutoff': the first latitude poleward of the tropical OLR maximum in each hemisphere where OLR crosses a specified cutoff value
         
         '10Perc': the first latitude poleward of the tropical OLR maximum in each hemisphere where OLR is 10# smaller than the tropical OLR maximum
         
         'max': the latitude of maximum of tropical olr in each hemisphere with the smoothing paramerer n=6 in TropD_Calculate_MaxLat
         
         'peak': the latitude of maximum of tropical olr in each hemisphere with the smoothing parameter n=30 in TropD_Calculate_MaxLat
       
       
       Cutoff (float, optional): Scalar. For the method 'cutoff', Cutoff specifies the OLR cutoff value. 
       
       n (int, optional): For the 'max' method, n is the smoothing parameter in TropD_Calculate_MaxLat
     
     Returns:
     
       tuple: PhiSH (ndarray), PhiNH (ndarray) Latitude of near equator OLR threshold crossing in SH and NH
     
  """

  try:
    assert(isinstance(n, int)) 
  except AssertionError:
    print('TropD_Metric_OLR: ERROR: the smoothing parameter n must be an integer')
  
  try:
    assert(n>=1) 
  except AssertionError:
    print('TropD_Metric_OLR: ERROR: the smoothing parameter n must be >= 1')

  
  # make latitude vector monotonically increasing
  if lat[-1] < lat[0]:
    olr = np.flip(olr,0)
    lat = np.flip(lat,0)
    
  eq_boundary = 5
  subpolar_boundary = 40
  polar_boundary = 60
  # NH
  olr_max_lat_NH = TropD_Calculate_MaxLat(olr[(lat > eq_boundary) & (lat < subpolar_boundary)],\
                    lat[(lat > eq_boundary) & (lat < subpolar_boundary)])
  olr_max_NH = max(olr[(lat > eq_boundary) & (lat < subpolar_boundary)])

  # SH
  olr_max_lat_SH = TropD_Calculate_MaxLat(olr[(lat > -subpolar_boundary) & (lat < -eq_boundary)],\
                    lat[(lat > -subpolar_boundary) & (lat < -eq_boundary)])
  olr_max_SH = max(olr[(lat > -subpolar_boundary) & (lat < -eq_boundary)])

  if method == '20W':
    PhiNH = TropD_Calculate_ZeroCrossing(olr[(lat > olr_max_lat_NH) & (lat < polar_boundary)] - olr_max_NH + 20,\
                    lat[(lat > olr_max_lat_NH) & (lat < polar_boundary)])
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(olr[(lat < olr_max_lat_SH) & \
                    (lat > -polar_boundary)],0) - olr_max_SH + 20,\
                    np.flip(lat[(lat < olr_max_lat_SH) & (lat > -polar_boundary)],0))

  elif method == '250W':
    PhiNH = TropD_Calculate_ZeroCrossing(olr[(lat > olr_max_lat_NH) & (lat < polar_boundary)] - 250,\
                    lat[(lat > olr_max_lat_NH) & (lat < polar_boundary)])
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(olr[(lat < olr_max_lat_SH) & (lat > -polar_boundary)],0) - 250,\
                    np.flip(lat[(lat < olr_max_lat_SH) & (lat > -polar_boundary)],0))

  elif method == 'cutoff':
    PhiNH = TropD_Calculate_ZeroCrossing(olr[(lat > olr_max_lat_NH) & (lat < polar_boundary)] - Cutoff,\
                    lat[(lat > olr_max_lat_NH) & (lat < polar_boundary)])
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(olr[(lat < olr_max_lat_SH) & (lat > -polar_boundary)],0) - Cutoff,\
                    np.flip(lat[(lat < olr_max_lat_SH) & (lat > -polar_boundary)],0))
  
  elif method == '10Perc':
    PhiNH = TropD_Calculate_ZeroCrossing(olr[(lat > olr_max_lat_NH) & (lat < polar_boundary)] / olr_max_NH - 0.9,\
                    lat[(lat > olr_max_lat_NH) & (lat < polar_boundary)])
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(olr[(lat < olr_max_lat_SH) & (lat > -polar_boundary)],0) \
                    / olr_max_SH - 0.9, np.flip(lat[(lat < olr_max_lat_SH) & (lat > -polar_boundary)],0))

  elif method == 'max':
    if Cutoff_is_set:
      PhiNH = TropD_Calculate_MaxLat(olr[(lat > eq_boundary) & (lat < subpolar_boundary)],\
                    lat[(lat > eq_boundary) & (lat < subpolar_boundary)], n=n)
      PhiSH = TropD_Calculate_MaxLat(olr[(lat > -subpolar_boundary) & (lat < -eq_boundary)],\
                    lat[(lat > -subpolar_boundary) & (lat < -eq_boundary)], n=n)
    else:
      PhiNH = np.copy(olr_max_lat_NH)
      PhiSH = np.copy(olr_max_lat_SH)
 
  elif method == 'peak':
    PhiNH = TropD_Calculate_MaxLat(olr[(lat > eq_boundary) & (lat < subpolar_boundary)],\
                    lat[(lat > eq_boundary) & (lat < subpolar_boundary)],n=30)
    PhiSH = TropD_Calculate_MaxLat(olr[(lat > -subpolar_boundary) & (lat < -eq_boundary)],\
                    lat[(lat > -subpolar_boundary) & (lat < -eq_boundary)], n=30)

  else:
    print('TropD_Metric_OLR: unrecognized method ', method)

    PhiNH = np.empty(0)
    PhiSH = np.empty(0)
  
  return PhiSH, PhiNH

def TropD_Metric_PE(pe,lat,method='zero_crossing',lat_uncertainty=0.0):

  ''' TropD Precipitation minus Evaporation (PE) metric
     
      Args:

        pe(lat,): zonal-mean precipitation minus evaporation
   
        lat: equally spaced latitude column vector

        method (str): 
       
          'zero_crossing': the first latitude poleward of the subtropical minimum where P-E changes from negative to positive values. Only one method so far.
  
        lat_uncertainty (float, optional): The minimal distance allowed between the first and second zero crossings along lat

      Returns:
        tuple: PhiSH (ndarray), PhiNH (ndarray) Latitude of first subtropical P-E zero crossing in SH and NH

  '''    
  try:
    assert(method in ['zero_crossing'])
  except AssertionError:
    print('TropD_Metric_PE: ERROR : unrecognized method ', method)
    
  # make latitude vector monotonically increasing
  if lat[-1] < lat[0]:
      pe = np.flip(pe)
      lat = np.flip(lat)
    
  # The gradient of PE is used to determine whether PE becomes positive at the zero crossing
  ped = np.interp(lat, (lat[:-1] + lat[1:])/2.0, np.diff(pe))
    
  # define latitudes of boundaries certain regions 
  eq_boundary = 5
  subpolar_boundary = 50
  polar_boundary = 60

    
  # NH
  M1 = TropD_Calculate_MaxLat(-pe[(lat > eq_boundary) & (lat < subpolar_boundary)],\
                 lat[(lat > eq_boundary) & (lat < subpolar_boundary)], n=30)
  ZC1 = TropD_Calculate_ZeroCrossing(pe[(lat > M1) & (lat < polar_boundary)], \
                 lat[(lat > M1) & (lat < polar_boundary)], lat_uncertainty=lat_uncertainty)
  if np.interp(ZC1, lat, ped) > 0:
    PhiNH = ZC1
  else:
    PhiNH = TropD_Calculate_ZeroCrossing(pe[(lat > ZC1) & (lat < polar_boundary)], \
                  lat[(lat > ZC1) & (lat < polar_boundary)], lat_uncertainty=lat_uncertainty)
  
  # SH
  # flip arrays to find the most equatorward zero crossing
  M1 = TropD_Calculate_MaxLat(np.flip(-pe[(lat < -eq_boundary) & (lat > -subpolar_boundary)],0),\
                 np.flip(lat[(lat < -eq_boundary) & (lat > -subpolar_boundary)],0), n=30)               
  ZC1 = TropD_Calculate_ZeroCrossing(np.flip(pe[(lat < M1) & (lat > -polar_boundary)],0), \
                 np.flip(lat[(lat < M1) & (lat > -polar_boundary)],0), lat_uncertainty=lat_uncertainty)

  if np.interp(ZC1, lat, ped) < 0:
    PhiSH = ZC1
  else:
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(pe[(lat < ZC1) & (lat > -polar_boundary)],0), \
                  np.flip(lat[(lat < ZC1) & (lat > -polar_boundary)],0), lat_uncertainty=lat_uncertainty)

  return PhiSH, PhiNH

def TropD_Metric_PSI(V, lat, lev, method='Psi_500', \
     lat_uncertainty=0, **kwargs):
  ''' TropD Mass streamfunction (PSI) metric

      Latitude of the meridional mass streamfunction subtropical zero crossing
     
      Args:
  
        V(lat,lev): zonal-mean meridional wind
      
        lat: latitude vector

        lev: vertical level vector in hPa units
  
        method (str, optional):
  
          'Psi_500'(default): Zero crossing of the stream function (Psi) at the 500hPa level

          'Psi_500_10Perc': Crossing of 10# of the extremum value of Psi in each hemisphre at the 500hPa level

          'Psi_300_700': Zero crossing of Psi vertically averaged between the 300hPa and 700 hPa levels

          'Psi_500_Int': Zero crossing of the vertically-integrated Psi at the 500 hPa level

          'Psi_Int'    : Zero crossing of the column-averaged Psi
    
        lat_uncertainty (float, optional): The minimal distance allowed between the first and second zero crossings. For example, for lat_uncertainty = 10, the function will return a NaN value if a second zero crossings is found within 10 degrees of the most equatorward zero crossing.   
  
      Returns:

        tuple: PhiSH (ndarray), PhiNH (ndarray), Psi (ndarray) Latitude of Psi zero crossing in SH and NH, streamfunction

  '''
  Radius = kwargs.pop('Radius', 6371220.0)
  Grav = kwargs.pop('Grav', 9.80616)

  try:
    assert (lat_uncertainty >= 0)  
  except AssertionError:
    print('TropD_Metric_PSI: ERROR : lat_uncertainty must be >= 0')
  
  try:
    assert(method in [
      'Psi_500','Psi_500_10Perc','Psi_300_700','Psi_500_Int','Psi_Int','Psi_Int_10Perc'])
  except AssertionError:
    print('TropD_Metric_PSI: ERROR : unrecognized method ', method)
    
  subpolar_boundary = kwargs.pop('spb', 65)
  polar_boundary = kwargs.pop('pb', 90)
    
  Psi = TropD_Calculate_StreamFunction(V, lat, lev, Radius=Radius, Grav=Grav)
  Psi[np.isnan(Psi)]=0
  # make latitude vector monotonically increasing
  if lat[-1] < lat[0]:
      Psi = np.flip(Psi, 0)
      lat = np.flip(lat, 0)
    
  COS = np.repeat(np.cos(lat*np.pi/180), len(lev), axis=0).reshape(len(lat),len(lev))
    
  if ( method == 'Psi_500' or method == 'Psi_500_10Perc'):
    # Use Psi at the level nearest to 500 hPa
    P = Psi[:,find_nearest(lev, 500)]

  elif (method == 'Psi_300_700'):
    # Use Psi averaged between the 300 and 700 hPa level
    P = np.trapz(Psi[:,(lev <= 700) & (lev >= 300)] * COS[:,(lev <= 700) & (lev >= 300)],\
                  lev[(lev <= 700) & (lev >= 300)]*100, axis=1)
  
  elif (method == 'Psi_3_0.3'):
    P = np.trapz(Psi[:,(lev <= 5) & (lev >= 0.5)] * COS[:,(lev <= 5) & (lev >= 0.5)],\
                  lev[(lev <= 5) & (lev >= 0.5)]*100, axis=1)

  elif (method == 'Psi_500_Int' or method == 'Psi_Int_10Perc'):
    # Use integrated Psi from p=0 to level mearest to 500 hPa
    PPsi_temp = sp.integrate.cumtrapz(Psi*COS, lev, axis=1)
    PPsi = np.zeros(np.shape(Psi))
    PPsi[:,1:] = PPsi_temp
    P = PPsi[:,find_nearest(lev, 500)]
     
  elif method == 'Psi_Int' or method == 'Int_10Perc':
    # Use vertical mean of Psi 
    P = np.trapz(Psi*COS, lev, axis=1)
  
  else:
    print('TropD_Metric_PSI: ERROR : Unrecognized method ', method)
  
    
  # 1. Find latitude of maximal (minimal) tropical Psi in the NH (SH)
  # 2. Find latitude of minimal (maximal) subtropical Psi in the NH (SH)
  # 3. Find the zero crossing between the above latitudes

  # NH
  Lmax = TropD_Calculate_MaxLat(P[(lat >= 0) & (lat < subpolar_boundary)],\
                                lat[(lat >= 0) & (lat < subpolar_boundary)])

  Lmin = TropD_Calculate_MaxLat(-P[(lat > Lmax) & (lat < polar_boundary)],\
                                lat[(lat > Lmax) & (lat < polar_boundary)])
  
  if (method == 'Psi_500_10Perc' or method == 'Psi_Int_10Perc' or method=="Int_10Perc"):
    Pmax = max(P[(lat > 0) & (lat < subpolar_boundary)])
    PhiNH = TropD_Calculate_ZeroCrossing(P[(lat > Lmax) & (lat < Lmin)] - 0.1*Pmax,\
            lat[(lat > Lmax) & (lat < Lmin)])


  else:
    PhiNH = TropD_Calculate_ZeroCrossing(P[(lat > Lmax) & (lat < Lmin)],\
            lat[(lat > Lmax) & (lat < Lmin)], lat_uncertainty=lat_uncertainty)
  
  # SH
  Lmax = TropD_Calculate_MaxLat(-P[(lat < 0) & (lat > -subpolar_boundary)],\
         lat[(lat < 0) & (lat > -subpolar_boundary)])
  Lmin = TropD_Calculate_MaxLat(P[(lat < Lmax) & (lat > -polar_boundary)],\
         lat[(lat < Lmax) & (lat > -polar_boundary)])

  if (method == 'Psi_500_10Perc' or method == 'Psi_Int_10Perc' or method=='Int_10Perc'):
    Pmin = min(P[(lat < 0) & (lat > -subpolar_boundary)])
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(P[(lat < Lmax) & (lat > Lmin)], 0) - 0.1*Pmin,\
            np.flip(lat[(lat < Lmax) & (lat > Lmin)], 0))
  else:
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(P[(lat < Lmax) & (lat > Lmin)], 0),\
            np.flip(lat[(lat < Lmax) & (lat > Lmin)], 0), lat_uncertainty=lat_uncertainty)
  return PhiSH, PhiNH, Psi

def TropD_Metric_PSL(ps, lat, method='peak', n=0):

  ''' TropD Sea-level pressure (PSL) metric

      Latitude of maximum of the subtropical sea-level pressure
  
      Args:
  
        ps(lat,): sea-level pressure
      
        lat: equally spaced latitude column vector

        method (str, optional): 'peak' (default) | 'max'
  
      Returns:

        tuple: PhiSH (ndarray), PhiNH (ndarray) Latitude of subtropical sea-level pressure maximum SH and NH

  '''
  try:
    assert(method in ['max','peak'])
  except AssertionError:
    print('TropD_Metric_PSL: ERROR : unrecognized method ', method)

  try:
    assert (not hasattr(n, "__len__") and n >= 0)  
  except AssertionError:
    print ('TropD_Metric_PSL: ERROR : the smoothing parameter n must be >= 0')

  eq_boundary = 15
  polar_boundary = 60
    
  if method == 'max':
    if n:
      PhiNH = TropD_Calculate_MaxLat(ps[(lat > eq_boundary) & (lat < polar_boundary)],\
             lat[(lat > eq_boundary) & (lat < polar_boundary)],n=n)
      PhiSH = TropD_Calculate_MaxLat(ps[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)], n=n)
    else:
      PhiNH = TropD_Calculate_MaxLat(ps[(lat > eq_boundary) & (lat < polar_boundary)],\
              lat[(lat > eq_boundary) & (lat < polar_boundary)])
      PhiSH = TropD_Calculate_MaxLat(ps[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)])

  elif method == 'peak':
    if n:
      PhiNH = TropD_Calculate_MaxLat(ps[(lat > eq_boundary) & (lat < polar_boundary)],\
              lat[(lat > eq_boundary) & (lat < polar_boundary)], n=n)
      PhiSH = TropD_Calculate_MaxLat(ps[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)], n=n)
    else:
      PhiNH = TropD_Calculate_MaxLat(ps[(lat > eq_boundary) & (lat < polar_boundary)],\
              lat[(lat > eq_boundary) & (lat < polar_boundary)], n=30)
      PhiSH = TropD_Calculate_MaxLat(ps[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)], n=30)
  else:
    print('TropD_Metric_PSL: ERROR: unrecognized method ', method)
  
  return PhiSH, PhiNH

def TropD_Metric_STJ(U, lat, lev, method='adjusted_peak', n=0):

  ''' TropD Subtropical Jet (STJ) metric
  
      Args:
  
        U(lat,lev): zonal mean zonal wind

        lat: latitude vector
      
        lev: vertical level vector in hPa units
  
        method (str, optional): 

          'adjusted_peak': Latitude of maximum (smoothing parameter n=30) of the zonal wind averaged between the 100 and 400 hPa levels minus the zonal mean zonal wind at the level closes to the 850 hPa level, poleward of 10 degrees and equatorward of the Eddy Driven Jet latitude
          
	        'adjusted_max' : Latitude of maximum (smoothing parameter n=6) of the zonal wind averaged between the 100 and 400 hPa levels minus the zonal mean zonal wind at the level closes to the 850 hPa level, poleward of 10 degrees and equatorward of the Eddy Driven Jet latitude

          'core_peak': Latitude of maximum of the zonal wind (smoothing parameter n=30) averaged between the 100 and 400 hPa levels, poleward of 10 degrees and equatorward of 70 degrees
          
	        'core_max': Latitude of maximum of the zonal wind (smoothing parameter n=6) averaged between the 100 and 400 hPa levels, poleward of 10 degrees and equatorward of 70 degrees
    
      Returns:

        tuple: PhiSH (ndarray), PhiNH (ndarray) Latitude of STJ SH and NH

  '''

  try:
    assert (not hasattr(n, "__len__") and n >= 0)  
  except AssertionError:
    print('TropD_Metric_STJ: ERROR : the smoothing parameter n must be >= 0')
  
  try:
    assert(method in ['adjusted_peak','core_peak','adjusted_max','core_max'])
  except AssertionError:
    print('TropD_Metric_STJ: ERROR : unrecognized method ', method)

  eq_boundary = 10
  polar_boundary = 80

  lev_int = lev[(lev >= 100) & (lev <= 400)]

  if (method == 'adjusted_peak' or method == 'adjusted_max'): 
    idx_850 = find_nearest(lev, 850)

    # Pressure weighted vertical mean of U minus near surface U
    if len(lev_int) > 1:
      u = np.trapz(U[:, (lev >= 100) & (lev <= 400)], lev_int, axis=1) \
          / (lev_int[-1] - lev_int[0]) - U[:,idx_850]

    else:
      u = np.mean(U[:,(lev >= 100) & (lev <= 400)], axis=1) - U[:,idx_850]

  elif (method == 'core_peak' or method == 'core_max'):
    # Pressure weighted vertical mean of U
    if len(lev_int) > 1:
      u = np.trapz(U[:, (lev >= 100) & (lev <= 400)], lev_int, axis=1) \
          / (lev_int[-1] - lev_int[0])

    else:
      u = np.mean(U[:, (lev >= 100) & (lev <= 400)], axis=1)

  else:
    print('TropD_Metric_STJ: unrecognized method ', method)
    print('TropD_Metric_STJ: optional methods are: adjusted_peak (default), adjusted_max, core_peak, core_max')

  if method == 'core_peak':
    if n:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
          lat[(lat > eq_boundary) & (lat < polar_boundary)], n=n)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
          lat[(lat > -polar_boundary) & (lat < -eq_boundary)], n=n)
    else:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
          lat[(lat > eq_boundary) & (lat < polar_boundary)],n=30)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
          lat[(lat > -polar_boundary) & (lat < -eq_boundary)], n=30)

  elif method == 'core_max':
    if n:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
          lat[(lat > eq_boundary) & (lat < polar_boundary)], n=n)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < - eq_boundary)],\
          lat[(lat > -polar_boundary) & (lat < -eq_boundary)], n=n)
    else:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
          lat[(lat > eq_boundary) & (lat < polar_boundary)], n=6)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
          lat[(lat > -polar_boundary) & (lat < -eq_boundary)], n=6)

  elif method == 'adjusted_peak':
    PhiSH_EDJ, PhiNH_EDJ = TropD_Metric_EDJ(U,lat,lev)
    if n:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < PhiNH_EDJ)],\
          lat[(lat > eq_boundary) & (lat < PhiNH_EDJ)], n=n)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > PhiSH_EDJ) & (lat < -eq_boundary)],\
          lat[(lat > PhiSH_EDJ) & (lat < -eq_boundary)], n=n)

    else:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < PhiNH_EDJ)],\
          lat[(lat > eq_boundary) & (lat < PhiNH_EDJ)], n=30)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > PhiSH_EDJ) & (lat < -eq_boundary)],\
          lat[(lat > PhiSH_EDJ) & (lat < -eq_boundary)], n=30)

  elif method == 'adjusted_max':
    PhiSH_EDJ,PhiNH_EDJ = TropD_Metric_EDJ(U,lat,lev)
    if n:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < PhiNH_EDJ)],\
          lat[(lat > eq_boundary) & (lat < PhiNH_EDJ)], n=n)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > PhiSH_EDJ) & (lat < -eq_boundary)],\
          lat[(lat > PhiSH_EDJ) & (lat < -eq_boundary)], n=n)
    else:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < PhiNH_EDJ)],\
          lat[(lat > eq_boundary) & (lat < PhiNH_EDJ)], n=6)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > PhiSH_EDJ) & (lat < -eq_boundary)],\
          lat[(lat > PhiSH_EDJ) & (lat < -eq_boundary)], n=6)

  return PhiSH, PhiNH, u

def TropD_Metric_TPB(T, lat, lev, method='max_gradient', n=0, Z=None, Cutoff=15*1000):

  ''' TropD Tropopause break (TPB) metric
  
      Args:

        T(lat,lev): temperature (K)

        lat: latitude vector

        lev: pressure levels column vector in hPa

        method (str, optional): 
  
          'max_gradient' (default): The latitude of maximal poleward gradient of the tropopause height
  
          'cutoff': The most equatorward latitude where the tropopause crosses a prescribed cutoff value
  
          'max_potemp': The latitude of maximal difference between the potential temperature at the tropopause and at the surface
  
        Z(lat,lev) (optional): geopotential height (m)

        Cutoff (float, optional): geopotential height (m) cutoff that marks the location of the tropopause break

      Returns:
        tuple: PhiSH (ndarray), PhiNH (ndarray) Latitude of tropopause break SH and NH

  '''


  Rd = 287.04
  Cpd = 1005.7
  k = Rd / Cpd
  try:
    assert (not hasattr(n, "__len__") and n >= 0)  
  except AssertionError:
    print('TropD_Metric_TPB: ERROR : the smoothing parameter n must be >= 0')

  try:
    assert(method in ['max_gradient','max_potemp','cutoff'])
  except AssertionError:
    print('TropD_Metric_TPB: ERROR : unrecognized method ', method)
  
  polar_boundary = 60

  if method == 'max_gradient':
    Pt = TropD_Calculate_TropopauseHeight(T,lev)
    Ptd = np.diff(Pt) / (lat[1] - lat[0])
    lat2 = (lat[1:] + lat[:-1]) / 2
    
    if (n >= 1):
      PhiNH = TropD_Calculate_MaxLat(Ptd[:,(lat2 > 0) & (lat2 < polar_boundary)],\
              lat2[(lat2 > 0) & (lat2 < polar_boundary)], n=n)
      PhiSH = TropD_Calculate_MaxLat(-Ptd[:,(lat2 > -polar_boundary) & (lat2 < 0)],\
              lat2[(lat2 > -polar_boundary) & (lat2 < 0)], n=n)
    
    else:
      PhiNH = TropD_Calculate_MaxLat(Ptd[:,(lat2 > 0) & (lat2 < polar_boundary)],\
              lat2[(lat2 > 0) & (lat2 < polar_boundary)])
      PhiSH = TropD_Calculate_MaxLat(-Ptd[:,(lat2 > -polar_boundary) & (lat2 < 0)],\
              lat2[(lat2 > -polar_boundary) & (lat2 < 0)])
     
  elif method == 'max_potemp':
    XF = np.tile((lev / 1000) ** k, (len(lat), 1))
    PT = T / XF
    Pt, PTt = TropD_Calculate_TropopauseHeight(T, lev, Z=PT)
    PTdif = PTt - np.nanmin(PT, axis = 1)
    if (n >= 1):
      PhiNH = TropD_Calculate_MaxLat(PTdif[:,(lat > 0) & (lat < polar_boundary)],\
              lat[(lat > 0) & (lat < polar_boundary)], n=n)
      PhiSH = TropD_Calculate_MaxLat(PTdif[:,(lat > - polar_boundary) & (lat < 0)],\
              lat[(lat > -polar_boundary) & (lat < 0)], n=n)
    
    else:
      PhiNH = TropD_Calculate_MaxLat(PTdif[:,(lat > 0) & (lat < polar_boundary)],\
              lat[(lat > 0) & (lat < polar_boundary)], n=30)
      PhiSH = TropD_Calculate_MaxLat(PTdif[:,(lat > - polar_boundary) & (lat < 0)],\
              lat[(lat > -polar_boundary) & (lat < 0)], n=30)
   
  elif method == 'cutoff':
    Pt, Ht = TropD_Calculate_TropopauseHeight(T, lev, Z)
    
    # make latitude vector monotonically increasing
    if lat[-1] < lat[0]:
      Ht = np.flip(np.squeeze(Ht),0)
      lat = np.flip(lat,0)
    
    polar_boundary = 60
      
    PhiNH = TropD_Calculate_ZeroCrossing(Ht[(lat > 0) & (lat < polar_boundary)] - Cutoff,
              lat[(lat > 0) & (lat < polar_boundary)])
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(Ht[(lat < 0) & (lat > -polar_boundary)], 0) - Cutoff,
              np.flip(lat[(lat < 0) & (lat > -polar_boundary)], 0))
  
  else:
    print('TropD_Metric_TPB: ERROR : Unrecognized method ', method)

  return PhiSH, PhiNH

def TropD_Metric_UAS(U, lat, lev=np.array([1]), method='zero_crossing', lat_uncertainty = 0):
  
  ''' TropD near-surface zonal wind (UAS) metric
  
      Args:

        U(lat,lev) or U (lat,)-- Zonal mean zonal wind. Also takes surface wind 
        
        lat: latitude vector
        
        lev: vertical level vector in hPa units. lev=np.array([1]) for single-level input zonal wind U(lat,)

        method (str): 
          'zero_crossing': the first subtropical latitude where near-surface zonal wind changes from negative to positive

        lat_uncertainty (float, optional): the minimal distance allowed between the first and second zero crossings
  
      Returns:
        tuple: PhiSH (ndarray), PhiNH (ndarray) Latitude of first subtropical zero crossing of the near surface zonal wind in SH and NH
        
  '''

  try:
    assert (lat_uncertainty >= 0)  
  except AssertionError:
    print('TropD_Metric_PSI: ERROR : lat_uncertainty must be >= 0')
    
  try:
    assert(method in ['zero_crossing'])
  except AssertionError:
    print('TropD_Metric_PSI: ERROR : unrecognized method ', method)
    
  if len(lev) > 1:
    uas = U[:,find_nearest(lev, 850)]
  else:
    uas = np.copy(U)
    
  # make latitude vector monotonically increasing
  if lat[-1] < lat[0]:
      uas = np.flip(uas)
      lat = np.flip(lat)

  # define latitudes of boundaries certain regions 
  eq_boundary = 5
  subpolar_boundary = 30
  polar_boundary = 60

  # NH
  uas_min_lat_NH = TropD_Calculate_MaxLat(-uas[(lat > eq_boundary) & (lat < subpolar_boundary)],\
                   lat[(lat > eq_boundary) & (lat < subpolar_boundary)])
  # SH
  uas_min_lat_SH = TropD_Calculate_MaxLat(-uas[(lat > -subpolar_boundary) & (lat < -eq_boundary)],\
      lat[(lat > -subpolar_boundary) & (lat < -eq_boundary)])
  try:
    assert(method == 'zero_crossing')
    PhiNH = TropD_Calculate_ZeroCrossing(uas[(lat > uas_min_lat_NH) & (lat < polar_boundary)],\
            lat[(lat > uas_min_lat_NH) & (lat < polar_boundary)], lat_uncertainty=lat_uncertainty)
    # flip arrays to find the most equatorward zero crossing
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(uas[(lat < uas_min_lat_SH) & (lat > -polar_boundary)],0),\
            np.flip(lat[(lat < uas_min_lat_SH) & (lat > -polar_boundary)],0), lat_uncertainty=lat_uncertainty)

    return PhiSH, PhiNH
  except AssertionError:
    print('TropD_Metric_UAS: ERROR : unrecognized method ', method)

def find_nearest(array, value):
  ''' Find the index of the item in the array nearest to the value
      
      Args:
        
        array: array

        value: value be found

      Returns:
          
        int: index of value in array

  '''
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return idx


def calc_2d_max(u, pfull, lats, plot = False):
    '''
    Function to calculate location and strength of maximum given zonal wind
    u(lat, pfull) field

    Parameters
    ----------

    lats : array-like
    pfull: array-like
    u    : array-like, with shape (pfull,lat)

    Returns
    -------

    jet_lat : location of maximum
    jet_max : strength of maximum
    '''

    # Restrict to 10 points around maximum

    u_max = np.where(u == np.nanmax(u.values))
    p_max = u_max[0][0]
    l_max = u_max[1][0]
    
    u_near = u[p_max-3:p_max+4,l_max-3:l_max+4].values
    p_near = pfull[p_max-3:p_max+4]
    l_near = lats[l_max-3:l_max+4]
    f = interpolate.interp2d(l_near,p_near,u_near, 
             kind='cubic',bounds_error=True)
    fine_p = np.linspace(p_near[0], p_near[-1], 600)
    fine_l = np.linspace(l_near[0], l_near[-1], 400)
    # Quartic fit, with smaller lat spacing
    #coefs = np.ma.polyfit(lats_near,u_near,2)
    #quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2
    # Find jet lat and max
    #jet_lat = fine_lats[np.where(quad == max(quad))[0][0]]
    #jet_max = coefs[2]+coefs[1]*jet_lat+coefs[0]*jet_lat**2
    # Plot fit?
    u_new = f(fine_l,fine_p)
    jet_max = np.max(u_new)
    u_max = np.where(u_new == jet_max)
    jet_p = fine_p[u_max[0][0]]
    jet_l = fine_l[u_max[1][0]]

    return jet_p, jet_l, jet_max


#Converted to python by Paul Staten Jul.29.2017
def TropD_Calculate_MaxLat(F,lat,n=int(6)):
  ''' Find latitude of absolute maximum value for a given interval

      Args:

        F: 1D array

        lat: equally spaced latitude array

        n (int): rank of moment used to calculate the position of max value. n = 1,2,4,6,8,...  

      Returns:
  
        float: location of max value of F along lat
  '''

  try:
    assert(isinstance(n, int)) 
  except AssertionError:
    print('TropD_Calculate_MaxLat: ERROR: the smoothing parameter n must be an integer')
  
  try:
    assert(n>=1) 
  except AssertionError:
    print('TropD_Calculate_MaxLat: ERROR: the smoothing parameter n must be >= 1')

  try: 
    assert(np.isfinite(F).all())
  except AssertionError:
    print('TropD_Calculate_MaxLat: ERROR: input field F has NaN values')

  F = F - np.min(F)
  F = F / np.max(F) 

  Ymax = np.trapz((F**n)*lat, lat) / np.trapz(F ** n, lat)

  return Ymax

#Converted to python by Paul Staten Jul.29.2017
def TropD_Calculate_MaxLat_nan(F,lat,n=int(6)):
  ''' Find latitude of absolute maximum value for a given interval

      Args:

        F: 1D array

        lat: equally spaced latitude array

        n (int): rank of moment used to calculate the position of max value. n = 1,2,4,6,8,...  

      Returns:
  
        float: location of max value of F along lat
  '''

  try:
    assert(isinstance(n, int)) 
  except AssertionError:
    print('TropD_Calculate_MaxLat: ERROR: the smoothing parameter n must be an integer')
  
  try:
    assert(n>=1) 
  except AssertionError:
    print('TropD_Calculate_MaxLat: ERROR: the smoothing parameter n must be >= 1')

  #try: 
  #  assert(np.isfinite(F).all())
  #except AssertionError:
  #  print('TropD_Calculate_MaxLat: ERROR: input field F has NaN values')

  if np.nanmax(F) < 0.:
    Ymax = np.nan
  else:
    F = F - np.nanmin(F)
    F = F / np.nanmax(F) 

    Ymax = np.trapz((F**n)*lat, lat) / np.trapz(F ** n, lat)

  return Ymax

def TropD_Calculate_Mon2Season(Fm, season=np.arange(12), m=0):
  ''' Calculate seasonal means from monthly time series

      Args:
  
        Fm: array of dimensions (time, latitude, level) or (time, level) or (time, latitude) 

        season: array of months e.g., [-1,0,1] for DJF

        m (int): index of first of January

      Returns:

        ndarray: the annual time series of the seasonal means
  '''

    
  try:
    assert(np.max(season)<12 and np.min(season)>=0)
  except AssertionError:
    print('season can only include indices from 1 to 12')
  
  End_Index = np.shape(Fm)[0]-m+1 - np.mod(np.shape(Fm)[0]-m+1,12)  
  Fm = Fm[m:End_Index,...]
  F = Fm[m + season[0]::12,...]
  if len(season) > 1:
    for s in season[1:]:
      F = F + Fm[m + s::12,...]
    F = F/len(season)  

  return F

def TropD_Calculate_StreamFunction(V, lat, lev, **kwargs):
  ''' Calculate streamfunction by integrating meridional wind from top of the atmosphere to surface

      Args:

        V: array of zonal-mean meridional wind with dimensions (lat, lev)
      
        lat: equally spaced latitude array

        lev: vertical level array in hPa

        radius: mean planetary radius in m (optional, default: 6.371220e^6)

        gravity: surface gravity in ms^-2 (optional, default: 9.80616)

      Returns:
  
        ndarray: the streamfunction psi(lat,lev) 
  '''

    
  Radius = kwargs.pop('radius', 6371220.0)
  Grav = kwargs.pop('gravity', 9.80616)
  
  B = np.ones(np.shape(V)) 
  # B = 0 for subsurface data
  B[np.isnan(V)]=0
  psi = np.zeros(np.shape(V))

  COS = np.repeat(np.cos(lat*np.pi/180), len(lev), axis=0).reshape(len(lat),len(lev))

  psi = (Radius/Grav) * 2 * np.pi \
       * sp.integrate.cumtrapz(B * V * COS, lev*100, axis=1, initial=0) 
  
  return psi

def TropD_Calculate_TropopauseHeight(T ,P, Z=None):
  ''' Calculate the Tropopause Height in isobaric coordinates 

      Based on the method described in Birner (2010), according to the WMO definition: first level at which the lapse rate <= 2K/km and for which the lapse rate <= 2K/km in all levels at least 2km above the found level 

      Args:

        T: Temperature array of dimensions (latitude, levels) on (longitude, latitude, levels)

        P: pressure levels in hPa

        Z (optional): geopotential height [m] or any field with the same dimensions as T

      Returns:

        ndarray or tuple: 

          If Z = None, returns Pt(lat) or Pt(lon,lat), the tropopause level in hPa 

          If Z is given, returns Pt and Ht with shape (lat) or (lon,lat). The field Z evaluated at the tropopause. For Z=geopotential height, Ht is the tropopause altitude in m 
  '''


  Rd = 287.04
  Cpd = 1005.7
  g = 9.80616
  k = Rd/Cpd
  PI = (np.linspace(1000,1,1000)*100)**k
  Factor = g/Cpd * 1000
  

  if len(np.shape(T)) == 2:
    T = np.expand_dims(T, axis=0)
    Z = np.expand_dims(Z, axis=0)
  # make P monotonically decreasing
  if P[-1] > P[0]:
    P = np.flip(P,0)
    T = np.flip(T,2)
    if Z.any():
      Z = np.flip(Z,2)

  Pk = np.tile((P*100)**k, (np.shape(T)[0], np.shape(T)[1], 1))
  Pk2 = (Pk[:,:,:-1] + Pk[:,:,1:])/2
  
  T2 = (T[:,:,:-1] + T[:,:,1:])/2
  Pk1 = np.squeeze(Pk2[0,0,:])

  Gamma = (T[:,:,1:] - T[:,:,:-1])/(Pk[:,:,1:] - Pk[:,:,:-1]) *\
          Pk2 / T2 * Factor
  Gamma = np.reshape(Gamma, (np.shape(Gamma)[0]*np.shape(Gamma)[1], np.shape(Gamma)[2]))

  T2 = np.reshape(T2, (np.shape(Gamma)[0], np.shape(Gamma)[1]))
  Pt = np.zeros((np.shape(T)[0]*np.shape(T)[1], 1))
  
  for j in range(np.shape(Gamma)[0]):
    G_f = sp.interpolate.interp1d(Pk1, Gamma[j,:], kind='linear', fill_value='extrapolate')
    G1 = G_f(PI)
    T2_f = sp.interpolate.interp1d(Pk1,T2[j,:], kind='linear', fill_value='extrapolate')
    T1 = T2_f(PI)
    idx = np.squeeze(np.where((G1 <=2) & (PI < (550*100)**k) & (PI > (75*100)**k)))
    Pidx = PI[idx] 

    if np.size(Pidx):
      for c in range(len(Pidx)):
        dpk_2km =  -2000 * k * g / Rd / T1[c] * Pidx[c]
        idx2 = find_nearest(Pidx[c:], Pidx[c] + dpk_2km)

        if sum(G1[idx[c]:idx[c]+idx2+1] <= 2)-1 == idx2:
          Pt[j]=Pidx[c]
          break
    else:
      Pt[j] = np.nan
      
 
  Pt = Pt ** (1 / k) / 100
    
  if Z.any():
    Zt =  np.reshape(Z, (np.shape(Z)[0]*np.shape(Z)[1], np.shape(Z)[2]))
    Ht =  np.zeros((np.shape(T)[0]*np.shape(T)[1]))
    
    for j in range(np.shape(Ht)[0]):
      f = sp.interpolate.interp1d(P, Zt[j,:])
      Ht[j] = f(Pt[j])

    Ht = np.reshape(Ht, (np.shape(T)[0], np.shape(T)[1]))
    Pt = np.reshape(Pt, (np.shape(T)[0], np.shape(T)[1]))
    return Pt, Ht
  
  else:
    
    Pt = np.reshape(Pt, (np.shape(T)[0], np.shape(T)[1]))
    return Pt

    
#Converted to python by Paul Staten Jul.29.2017
def TropD_Calculate_ZeroCrossing(F, lat, lat_uncertainty=0.0):

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

def TropD_Metric_STJ_all_lats(U, lat, lev, method='adjusted_peak', n=0, **kwargs):
  ''' TropD Subtropical Jet (STJ) metric
  
      Args:
  
        U(lat,lev): zonal mean zonal wind

        lat: latitude vector
      
        lev: vertical level vector in hPa units
  
        method (str, optional): 

          'adjusted_peak': Latitude of maximum (smoothing parameter n=30) of the zonal wind averaged between the 100 and 400 hPa levels minus the zonal mean zonal wind at the level closes to the 850 hPa level, poleward of 10 degrees and equatorward of the Eddy Driven Jet latitude
          
	        'adjusted_max' : Latitude of maximum (smoothing parameter n=6) of the zonal wind averaged between the 100 and 400 hPa levels minus the zonal mean zonal wind at the level closes to the 850 hPa level, poleward of 10 degrees and equatorward of the Eddy Driven Jet latitude

          'core_peak': Latitude of maximum of the zonal wind (smoothing parameter n=30) averaged between the 100 and 400 hPa levels, poleward of 10 degrees and equatorward of 70 degrees
          
	        'core_max': Latitude of maximum of the zonal wind (smoothing parameter n=6) averaged between the 100 and 400 hPa levels, poleward of 10 degrees and equatorward of 70 degrees
    
      Returns:

        tuple: PhiSH (ndarray), PhiNH (ndarray) Latitude of STJ SH and NH

  '''

  try:
    assert (not hasattr(n, "__len__") and n >= 0)  
  except AssertionError:
    print('TropD_Metric_STJ: ERROR : the smoothing parameter n must be >= 0')
  
  try:
    assert(method in ['adjusted_peak','core_peak','adjusted_max','core_max'])
  except AssertionError:
    print('TropD_Metric_STJ: ERROR : unrecognized method ', method)

  eq_b_n    = kwargs.pop('eq_b_n',   5)
  polar_b_n = kwargs.pop('po_b_n',  85)
  eq_b_s    = kwargs.pop('eq_b_n',  -5)
  polar_b_s = kwargs.pop('po_b_n', -85)


  lev_int = lev[(lev >= 100) & (lev <= 400)]

  if (method == 'adjusted_peak' or method == 'adjusted_max'): 
    idx_850 = find_nearest(lev, 850)

    # Pressure weighted vertical mean of U minus near surface U
    if len(lev_int) > 1:
      u = np.trapz(U[:, (lev >= 100) & (lev <= 400)], lev_int, axis=1) \
          / (lev_int[-1] - lev_int[0]) - U[:,idx_850]

    else:
      u = np.mean(U[:,(lev >= 100) & (lev <= 400)], axis=1) - U[:,idx_850]

  elif (method == 'core_peak' or method == 'core_max'):
    # Pressure weighted vertical mean of U
    if len(lev_int) > 1:
      u = np.trapz(U[:, (lev >= 100) & (lev <= 400)], lev_int, axis=1) \
          / (lev_int[-1] - lev_int[0])

    else:
      u = np.mean(U[:, (lev >= 100) & (lev <= 400)], axis=1)

  else:
    print('TropD_Metric_STJ: unrecognized method ', method)
    print('TropD_Metric_STJ: optional methods are: adjusted_peak (default), adjusted_max, core_peak, core_max')

  if method == 'core_peak':
    if n:
      PhiNH = TropD_Calculate_MaxLat_nan(u[(lat > eq_b_n) & (lat < polar_b_n)],\
          lat[(lat > eq_b_n) & (lat < polar_b_n)], n=n)
      PhiSH = TropD_Calculate_MaxLat_nan(u[(lat > polar_b_s) & (lat < eq_b_s)],\
          lat[(lat > polar_b_s) & (lat < eq_b_s)], n=n)
    else:
      PhiNH = TropD_Calculate_MaxLat_nan(u[(lat > eq_b_n) & (lat < polar_b_n)],\
          lat[(lat > eq_b_n) & (lat < polar_b_n)], n=30)
      PhiSH = TropD_Calculate_MaxLat_nan(u[(lat > polar_b_s) & (lat < eq_b_s)],\
          lat[(lat > polar_b_s) & (lat < eq_b_s)], n=30)

  elif method == 'core_max':
    if n:
      PhiNH = TropD_Calculate_MaxLat_nan(u[(lat > eq_b_n) & (lat < polar_b_n)],\
          lat[(lat > eq_b_n) & (lat < polar_b_n)], n=n)
      PhiSH = TropD_Calculate_MaxLat_nan(u[(lat > polar_b_s) & (lat < eq_b_s)],\
          lat[(lat > polar_b_s) & (lat < eq_b_s)], n=n)
    else:
      PhiNH = TropD_Calculate_MaxLat_nan(u[(lat > eq_b_n) & (lat < polar_b_n)],\
          lat[(lat > eq_b_n) & (lat < polar_b_n)], n=6)
      PhiSH = TropD_Calculate_MaxLat_nan(u[(lat > polar_b_s) & (lat < eq_b_s)],\
          lat[(lat > polar_b_s) & (lat < eq_b_s)], n=6)

  elif method == 'adjusted_peak':
    #PhiSH_EDJ, PhiNH_EDJ = TropD_Metric_EDJ(U,lat,lev)
    if n:
      PhiNH = TropD_Calculate_MaxLat_nan(u[(lat > eq_b_n) & (lat < polar_b_n)],\
          lat[(lat > eq_b_n) & (lat < polar_b_n)], n=n)
      PhiSH = TropD_Calculate_MaxLat_nan(u[(lat > polar_b_s) & (lat < eq_b_s)],\
          lat[(lat > polar_b_s) & (lat < eq_b_s)], n=n)
    else:
      PhiNH = TropD_Calculate_MaxLat_nan(u[(lat > eq_b_n) & (lat < polar_b_n)],\
          lat[(lat > eq_b_n) & (lat < polar_b_n)], n=30)
      PhiSH = TropD_Calculate_MaxLat_nan(u[(lat > polar_b_s) & (lat < eq_b_s)],\
          lat[(lat > polar_b_s) & (lat < eq_b_s)], n=30)

  elif method == 'adjusted_max':
    #PhiSH_EDJ,PhiNH_EDJ = TropD_Metric_EDJ(U,lat,lev)
    if n:
      PhiNH = TropD_Calculate_MaxLat_nan(u[(lat > eq_b_n) & (lat < polar_b_n)],\
          lat[(lat > eq_b_n) & (lat < polar_b_n)], n=n)
      PhiSH = TropD_Calculate_MaxLat_nan(u[(lat > polar_b_s) & (lat < eq_b_s)],\
          lat[(lat > polar_b_s) & (lat < eq_b_s)], n=n)
    else:
      PhiNH = TropD_Calculate_MaxLat_nan(u[(lat > eq_b_n) & (lat < polar_b_n)],\
          lat[(lat > eq_b_n) & (lat < polar_b_n)], n=30)
      PhiSH = TropD_Calculate_MaxLat_nan(u[(lat > polar_b_s) & (lat < eq_b_s)],\
          lat[(lat > polar_b_s) & (lat < eq_b_s)], n=30)

  return PhiSH, PhiNH, u


def TropD_Metric_EDJ_pole(U, lat, lev=np.array([1]), method='peak', n=0, n_fit=1):
  '''TropD Eddy Driven Jet (EDJ) metric
       
     Latitude of maximum of the zonal wind at the level closest to the 850 hPa level
     
     Args:
       U (lat,lev) or U (lat,): Zonal mean zonal wind. Also takes surface wind 
       lat : latitude vector
       lev: vertical level vector in hPa units

       method (str, optional): 'peak' (default) |  'max' | 'fit'
       
        peak (Default): Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=30)
        
        max: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=6)
        fit: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level using a quadratic polynomial fit of data from gridpoints surrounding the gridpoint of the maximum
        
       n (int, optional): If n is not set (0), n=6 (default) is used in TropD_Calculate_MaxLat. Rank of moment used to calculate the position of max value. n = 1,2,4,6,8,...  
     
     Returns:
       tuple: PhiSH (ndarray), PhiNH (ndarray) Latitude of EDJ in SH and NH

  '''

  try:
    assert (not hasattr(n, "__len__") and n >= 0)  
  except AssertionError:
   print('TropD_Metric_EDJ: ERROR : the smoothing parameter n must be >= 0')
   
  try:
    assert(method in ['max','peak'])
  except AssertionError:
    print('TropD_Metric_EDJ: ERROR : unrecognized method ', method)

  eq_boundary = 5
  polar_boundary = 80
  
  if len(lev) > 1:
    u = U[:,find_nearest(lev, 850)]
  else:
    u = np.copy(U)
    
  if method == 'max':
    if n:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
              lat[(lat > eq_boundary) & (lat < polar_boundary)], n=n)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)], n=n)

    else:
      #Default value of n=6 is used
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
              lat[(lat > eq_boundary) & (lat < polar_boundary)])
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)])
  
  elif method == 'peak':
    if n:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
              lat[(lat > eq_boundary) & (lat < polar_boundary)], n=n)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)], n=n)
    else:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
              lat[(lat > eq_boundary) & (lat < polar_boundary)],n=30)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)],n=30)
  
  elif method == 'fit':
    Uh = u[(lat > eq_boundary) & (lat < polar_boundary)]
    Lat = lat[(lat > eq_boundary) & (lat < polar_boundary)]
    m = np.nanmax(Uh)
    Im = np.nanargmax(Uh)
     
    if (Im == 0 or Im == len(Uh)-1):
      PhiNH = Lat[Im]
    
    elif (n_fit > Im or n_fit > len(Uh)-Im+1):
      N = np.min(Im, len(Uh)-Im+1)
      p = np.polyfit(Lat[Im-N:Im+N+1], Uh[Im-N:Im+N+1],2) 
      PhiNH = -p[1]/(2*p[0])
    else:
      p = np.polyfit(Lat[Im-n_fit:Im+n_fit+1], Uh[Im-n_fit:Im+n_fit+1],2) 
      PhiNH = -p[1]/(2*p[0])
    
    Uh = u[(lat > -polar_boundary) & (lat < -eq_boundary)]
    Lat = lat[(lat > -polar_boundary) & (lat < -eq_boundary)]
    
    m = np.nanmax(Uh)
    Im = np.nanargmax(Uh)
    
    if (Im == 0 or Im == len(Uh)-1):
      PhiSH = Lat[Im]
    
    elif (n_fit > Im or n_fit > len(Uh)-Im+1):
      N = np.min(Im, len(Uh)-Im+1)
      p = np.polyfit(Lat[Im-N:Im+N+1], Uh[Im-N:Im+N+1],2) 
      PhiSH = -p[1]/(2*p[0])
    else:
      p = np.polyfit(Lat[Im-n_fit:Im+n_fit+1], Uh[Im-n_fit:Im+n_fit+1],2) 
      PhiSH = -p[1]/(2*p[0])
  
  else:
    print('TropD_Metric_EDJ: ERROR: unrecognized method ', method)

  return PhiSH, PhiNH


def EDJ_pole_eq(U, lat, lev=np.array([1]), method='peak', n=0, n_fit=1):
  '''TropD Eddy Driven Jet (EDJ) metric
       
     Latitude of maximum of the zonal wind at the level closest to the 850 hPa level
     
     Args:
       U (lat,lev) or U (lat,): Zonal mean zonal wind. Also takes surface wind 
       lat : latitude vector
       lev: vertical level vector in hPa units

       method (str, optional): 'peak' (default) |  'max' | 'fit'
       
        peak (Default): Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=30)
        
        max: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=6)
        fit: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level using a quadratic polynomial fit of data from gridpoints surrounding the gridpoint of the maximum
        
       n (int, optional): If n is not set (0), n=6 (default) is used in TropD_Calculate_MaxLat. Rank of moment used to calculate the position of max value. n = 1,2,4,6,8,...  
     
     Returns:
       tuple: PhiSH (ndarray), PhiNH (ndarray) Latitude of EDJ in SH and NH

  '''

  try:
    assert (not hasattr(n, "__len__") and n >= 0)  
  except AssertionError:
   print('TropD_Metric_EDJ: ERROR : the smoothing parameter n must be >= 0')
   
  try:
    assert(method in ['max','peak'])
  except AssertionError:
    print('TropD_Metric_EDJ: ERROR : unrecognized method ', method)

  eq_boundary = 5
  polar_boundary = 80
  
  if len(lev) > 1:
    u = U[:,find_nearest(lev, 850)]
  else:
    u = np.copy(U)
    
  if method == 'max':
      #Default value of n=6 is used
      PhiNH_0, PhiNH_1 = Calculate_MaxLat_iterative(u[(lat > eq_boundary) & (lat < polar_boundary)],\
              lat[(lat > eq_boundary) & (lat < polar_boundary)],tol=0.9)
      PhiSH_0, PhiSH_1 = Calculate_MaxLat_iterative(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)],tol=0.9) 
  
  else:
    print('TropD_Metric_EDJ: ERROR: unrecognized method ', method)

  return PhiSH_1, PhiSH_0, PhiNH_0, PhiNH_1

#Converted to python by Paul Staten Jul.29.2017
def Calculate_MaxLat_iterative(F, lat, tol=0.0):

  ''' Find the first (with increasing index) maximum of the function F

      Args:
  
        F: array

        lat: latitude array (same length as F)

        tol (float, optional): The minimal distance allowed between adjacent zero crossings of indetical sign change for example, for lat_uncertainty = 10, if the most equatorward zero crossing is from positive to negative, the function will return a NaN value if an additional zero crossings from positive to negative is found within 10 degrees of that zero crossing.

      Returns:

        float: latitude of local maxima
  '''
  u_max = np.where(F == np.ma.max(F))[0][0]
  
  # Find first maximum in index units.
  D = np.sign(np.diff(F))
  
  a = np.where(np.diff(D)==-2)[0] # selects locations of all local maxima

  # Make sure a local maximum exists
  if not any(a) and D[0] > 0:
    return np.max(np.abs(lat))
  elif not any(a) and D[0] < 0:
    return np.min(np.abs(lat))

  # calculate first maximum
  for j in range(len(a)):
    u_near = F[a[j]:a[j]+3]
    if np.abs((F[u_max] - u_near[1])/F[u_max]) < tol:
      break
    else:
      continue
  
  lats_near = lat[a[j]:a[j]+3]
  # Quartic fit, with smaller lat spacing
  coefs = np.ma.polyfit(lats_near,u_near,2)
  fine_lats = np.linspace(lats_near[0], lats_near[-1],200)
  quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2
  # Find jet lat
  jet_lat_0 = fine_lats[np.where(quad == max(quad))[0][0]]

  # calculate last maximum
  for j in range(len(a)):
    u_near = F[a[-1-j]:a[-1-j]+3]
    if np.abs((F[u_max] - u_near[1])/F[u_max]) < tol:
      break
    else:
      continue

  lats_near = lat[a[-1-j]:a[-1-j]+3]
  # Quartic fit, with smaller lat spacing
  coefs = np.ma.polyfit(lats_near,u_near,2)
  fine_lats = np.linspace(lats_near[0], lats_near[-1],200)
  quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2
  # Find jet lat
  jet_lat_1 = fine_lats[np.where(quad == max(quad))[0][0]]

  return jet_lat_0, jet_lat_1

  
def PSI_new(V, lat, lev, method='Psi_500', lat_uncertainty=0):
  ''' TropD Mass streamfunction (PSI) metric

      Latitude of the meridional mass streamfunction subtropical zero crossing
     
      Args:
  
        V(lat,lev): zonal-mean meridional wind
      
        lat: latitude vector

        lev: vertical level vector in hPa units
  
        method (str, optional):
  
          'Psi_500'(default): Zero crossing of the stream function (Psi) at the 500hPa level

          'Psi_500_10Perc': Crossing of 10# of the extremum value of Psi in each hemisphre at the 500hPa level

          'Psi_300_700': Zero crossing of Psi vertically averaged between the 300hPa and 700 hPa levels

          'Psi_500_Int': Zero crossing of the vertically-integrated Psi at the 500 hPa level

          'Psi_Int'    : Zero crossing of the column-averaged Psi
    
        lat_uncertainty (float, optional): The minimal distance allowed between the first and second zero crossings. For example, for lat_uncertainty = 10, the function will return a NaN value if a second zero crossings is found within 10 degrees of the most equatorward zero crossing.   
  
      Returns:

        tuple: PhiSH (ndarray), PhiNH (ndarray), Psi (ndarray) Latitude of Psi zero crossing in SH and NH, streamfunction

  '''


  try:
    assert (lat_uncertainty >= 0)  
  except AssertionError:
    print('TropD_Metric_PSI: ERROR : lat_uncertainty must be >= 0')
  
  try:
    assert(method in ['Psi_500','Psi_500_10Perc','Psi_300_700','Psi_500_Int','Psi_Int'])
  except AssertionError:
    print('TropD_Metric_PSI: ERROR : unrecognized method ', method)
    
  subpolar_boundary = 45
  polar_boundary = 90
    
  Psi = TropD_Calculate_StreamFunction(V, lat, lev)
  Psi[np.isnan(Psi)]=0
  # make latitude vector monotonically increasing
  if lat[-1] < lat[0]:
      Psi = np.flip(Psi, 0)
      lat = np.flip(lat, 0)
    
  COS = np.repeat(np.cos(lat*np.pi/180), len(lev), axis=0).reshape(len(lat),len(lev))
    
  if ( method == 'Psi_500' or method == 'Psi_500_10Perc'):
    # Use Psi at the level nearest to 500 hPa
    P = Psi[:,find_nearest(lev, 500)]

  elif method == 'Psi_300_700':
    # Use Psi averaged between the 300 and 700 hPa level
    P = np.trapz(Psi[:,(lev <= 700) & (lev >= 300)] * COS[:,(lev <= 700) & (lev >= 300)],\
                  lev[(lev <= 700) & (lev >= 300)]*100, axis=1)

  elif method == 'Psi_500_Int':
    # Use integrated Psi from p=0 to level mearest to 500 hPa
    PPsi_temp = sp.integrate.cumtrapz(Psi*COS, lev, axis=1)
    PPsi = np.zeros(np.shape(Psi))
    PPsi[:,1:] = PPsi_temp
    P = PPsi[:,find_nearest(lev, 500)]
     
  elif method == 'Psi_Int':
    # Use vertical mean of Psi 
    P = np.trapz(Psi*COS, lev, axis=1)
  
  else:
    print('TropD_Metric_PSI: ERROR : Unrecognized method ', method)
  
    
  # 1. Find latitude of maximal (minimal) tropical Psi in the NH (SH)
  # 2. Find latitude of minimal (maximal) subtropical Psi in the NH (SH)
  # 3. Find the zero crossing between the above latitudes

  # NH
  Lmax = TropD_Calculate_MaxLat(P[(lat > 0) & (lat < subpolar_boundary)],\
                                lat[(lat > 0) & (lat < subpolar_boundary)])

  Lmin = TropD_Calculate_MaxLat(-P[(lat > Lmax) & (lat < polar_boundary)],\
                                lat[(lat > Lmax) & (lat < polar_boundary)])
  if method == 'Psi_500_10Perc':
    Pmax = max(P[(lat > 0) & (lat < subpolar_boundary)])
    PhiNH = TropD_Calculate_ZeroCrossing(P[(lat > Lmax) & (lat < Lmin)] - 0.1*Pmax,\
            lat[(lat > Lmax) & (lat < Lmin)])

  else:
    PhiNH = TropD_Calculate_ZeroCrossing(P[(lat > Lmax) & (lat < Lmin)],\
            lat[(lat > Lmax) & (lat < Lmin)], lat_uncertainty=lat_uncertainty)
  
  # SH
  Lmax = TropD_Calculate_MaxLat(-P[(lat < 0) & (lat > -subpolar_boundary)],\
         lat[(lat < 0) & (lat > -subpolar_boundary)])

  Lmin = TropD_Calculate_MaxLat(P[(lat < Lmax) & (lat > -polar_boundary)],\
         lat[(lat < Lmax) & (lat > -polar_boundary)])

  if method == 'Psi_500_10Perc':
    Pmin = min(P[(lat < 0) & (lat > -subpolar_boundary)])
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(P[(lat < Lmax) & (lat > Lmin)], 0) + 0.1*Pmin,\
            np.flip(lat[(lat < Lmax) & (lat > Lmin)], 0))
  else:
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(P[(lat < Lmax) & (lat > Lmin)], 0),\
            np.flip(lat[(lat < Lmax) & (lat > Lmin)], 0), lat_uncertainty=lat_uncertainty)
  return PhiSH, PhiNH, Psi


def find_jets(U, lat, lev=np.array([1]), n=0, n_fit=1):
  '''TropD Eddy Driven Jet (EDJ) metric
       
     Latitude of maximum of the zonal wind at the level closest to the 850 hPa level
     
     Args:
       U (time,lat,lev) or U (time,lat,): Zonal mean zonal wind. Also takes surface wind 
       time: time vector
       lat : latitude vector
       lev: vertical level vector in hPa units

       method (str, optional): 'peak' (default) |  'max' | 'fit'
       
        peak (Default): Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=30)
        
        max: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=6)
        fit: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level using a quadratic polynomial fit of data from gridpoints surrounding the gridpoint of the maximum
        
       n (int, optional): If n is not set (0), n=6 (default) is used in TropD_Calculate_MaxLat. Rank of moment used to calculate the position of max value. n = 1,2,4,6,8,...  
     
     Returns:
       tuple: PhiSH (ndarray), PhiNH (ndarray) Latitude of EDJ in SH and NH

  '''

  try:
    assert (not hasattr(n, "__len__") and n >= 0)  
  except AssertionError:
   print('TropD_Metric_EDJ: ERROR : the smoothing parameter n must be >= 0')
   
  #try:
  #  assert(method in ['max','peak'])
  #except AssertionError:
  #  print('TropD_Metric_EDJ: ERROR : unrecognized method ', method)

  eq_boundary = 15
  polar_boundary = 65
  
  if len(lev) > 1:
    u = U[:,find_nearest(lev, 850)]
  else:
    u = np.copy(U)
  #if method == 'max':
      #Default value of n=6 is used
  jets_n = find_all_max_n(u[:,(lat > eq_boundary) & (lat < polar_boundary)],\
          lat[(lat > eq_boundary) & (lat < polar_boundary)],lat_tol=15)
  jets_s = find_all_max_n(u[:,(lat > -polar_boundary) & (lat < -eq_boundary)],\
          lat[(lat > -polar_boundary) & (lat < -eq_boundary)],lat_tol=15) 
  
  #else:
  #  print('TropD_Metric_EDJ: ERROR: unrecognized method ', method)

  return jets_n, jets_s

#Converted to python by Paul Staten Jul.29.2017
def find_all_max_n(F, lat, lat_tol=2.0,mem=5,max_tol=75):

  ''' Find all local maxima of the function F

      Args:
  
        F: array

        lat: latitude array (same length as F)

        tol (float, optional): The minimal distance allowed between adjacent zero crossings of indetical sign change for example, for lat_uncertainty = 10, if the most equatorward zero crossing is from positive to negative, the function will return a NaN value if an additional zero crossings from positive to negative is found within 10 degrees of that zero crossing.

      Returns:

        float: latitude of local maxima
  '''
  #u_max = np.where(F == np.ma.max(F))[0][0] ## global maximum
  
  #x = np.full_like(F, fill_value=np.nan)
  z = []
  
  # Find maxima for all timesteps.
  for i in range(F.shape[0]):
    a = []
    try:
      u_max = np.where(F[i,:] == np.ma.max(F[i,:]))[0][0] ## global maximum
      b = find_peaks(F[i,:])[0]

      for j in b:
        if F[i,j] <= 0:     # ensure eastward flow
          continue          # could include a min strength threshold as either a constant or dependent on max value
        elif np.abs(F[i,j]/u_max)*100 < max_tol:
          continue
        else:
          a.append(j)
      z.append(a)
    except:
      z.append(a)
    #a = a[np.abs((u_max - a)/u_max) < max_tol]
    
  w = 0
  y = []
  for i in range(F.shape[0]):
    x = []
    for j in range(len(z[i])):
      u_near = F[i,int(z[i][j])-1:int(z[i][j])+2]
      lats_near = lat[int(z[i][j])-1:int(z[i][j])+2]
      # Quartic fit, with smaller lat spacing
      coefs = np.ma.polyfit(lats_near,u_near,2)
      fine_lats = np.linspace(lats_near[0], lats_near[-1],200)
      quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2
      # Find jet lat
      a = fine_lats[np.where(quad == max(quad))[0][0]]
      x.append(a)
    y.append(x)
    if w < len(y[i]):
      w = len(y[i])
  w = w+mem
  
  v = np.full((F.shape[0],w),np.nan)
  z = np.full((F.shape[0],w),np.nan)
  x = np.full((F.shape[0],w),np.nan)
  for i in range(F.shape[0]):
    for j in range(len(y[i])):
      z[i,j] = y[i][j]
  for i in range(F.shape[0]):
    if i == 0:
      x[i,:] = z[i,:]
    else:         # check that this maximum is within 5 degrees of a recent maximum
      for j in range(w):
        for k in range(w):
          for t in range(min(mem, i)):
            if np.abs(z[i,j]-z[i-1-t,k]) < lat_tol:
              x[i,j] = z[i,j]
              break
  for i in range(F.shape[0]):
    if i == 0:
      v[i,:] = x[i,:]
    else:
      for j in range(w):
        if not np.isfinite(x[i,j]):
          continue
        else:
          for t in range(min(mem,i)):
            if not any([np.abs(v[i-1-t,l]-x[i,j])<lat_tol for l in range(w)]):
              continue
            else:
              a = np.argsort(np.abs(x[i,j]-v[i-1-t,:]))
              for b in a:
                if np.abs(x[i,j]-v[i-1-t,b])<lat_tol:
                  if not np.isfinite(v[i,b]):
                    v[i,b] = x[i,j]
                    break
              break
          if t == mem-1 and (not any([v[i,l]==x[i,j] for l in range(w)])):
            for l in range(w):
              if any([np.isfinite(v[i-1-t,l]) for t in range(mem)]):
                continue
              else:
                if np.isfinite(v[i,l]):
                  continue
                else:
                  v[i,l] = x[i,j]
                  break
      
  l = []
  for j in range(w):
    a = sum([math.isnan(v[i,j]) for i in range(F.shape[0])])
    if a <= F.shape[0] * 6/10:
      l.append(j)
  w = np.full((F.shape[0],len(l)),np.nan)
  
  for i in range(F.shape[0]):
    for j in range(len(l)):
      w[i,j] = v[i,l[j]]
  
  return w



def TropD_Metric_STJ_new(u, lat, method='adjusted_peak', n=0, **kwargs):
  ''' TropD Subtropical Jet (STJ) metric
  
      Args:
  
        u(lat): adjusted zonal mean zonal wind

        lat: latitude vector
  
        method (str, optional): 

          'adjusted_peak': Latitude of maximum (smoothing parameter n=30) of the zonal wind averaged between the 100 and 400 hPa levels minus the zonal mean zonal wind at the level closes to the 850 hPa level, poleward of 10 degrees and equatorward of the Eddy Driven Jet latitude
          
	        'adjusted_max' : Latitude of maximum (smoothing parameter n=6) of the zonal wind averaged between the 100 and 400 hPa levels minus the zonal mean zonal wind at the level closes to the 850 hPa level, poleward of 10 degrees and equatorward of the Eddy Driven Jet latitude

          'core_peak': Latitude of maximum of the zonal wind (smoothing parameter n=30) averaged between the 100 and 400 hPa levels, poleward of 10 degrees and equatorward of 70 degrees
          
	        'core_max': Latitude of maximum of the zonal wind (smoothing parameter n=6) averaged between the 100 and 400 hPa levels, poleward of 10 degrees and equatorward of 70 degrees
    
      Returns:

        tuple: PhiSH (ndarray), PhiNH (ndarray) Latitude of STJ SH and NH

  '''

  try:
    assert (not hasattr(n, "__len__") and n >= 0)  
  except AssertionError:
    print('TropD_Metric_STJ: ERROR : the smoothing parameter n must be >= 0')
  
  try:
    assert(method in ['adjusted_peak','core_peak','adjusted_max','core_max'])
  except AssertionError:
    print('TropD_Metric_STJ: ERROR : unrecognized method ', method)

  eq_b_n    = kwargs.pop('eq_b_n',   5)
  polar_b_n = kwargs.pop('po_b_n',  85)
  eq_b_s    = kwargs.pop('eq_b_n',  -5)
  polar_b_s = kwargs.pop('po_b_n', -85)

  if method == 'core_peak':
    if n:
      PhiNH = TropD_Calculate_MaxLat_nan(u[(lat > eq_b_n) & (lat < polar_b_n)],\
          lat[(lat > eq_b_n) & (lat < polar_b_n)], n=n)
      PhiSH = TropD_Calculate_MaxLat_nan(u[(lat > polar_b_s) & (lat < eq_b_s)],\
          lat[(lat > polar_b_s) & (lat < eq_b_s)], n=n)
    else:
      PhiNH = TropD_Calculate_MaxLat_nan(u[(lat > eq_b_n) & (lat < polar_b_n)],\
          lat[(lat > eq_b_n) & (lat < polar_b_n)], n=30)
      PhiSH = TropD_Calculate_MaxLat_nan(u[(lat > polar_b_s) & (lat < eq_b_s)],\
          lat[(lat > polar_b_s) & (lat < eq_b_s)], n=30)

  elif method == 'core_max':
    if n:
      PhiNH = TropD_Calculate_MaxLat_nan(u[(lat > eq_b_n) & (lat < polar_b_n)],\
          lat[(lat > eq_b_n) & (lat < polar_b_n)], n=n)
      PhiSH = TropD_Calculate_MaxLat_nan(u[(lat > polar_b_s) & (lat < eq_b_s)],\
          lat[(lat > polar_b_s) & (lat < eq_b_s)], n=n)
    else:
      PhiNH = TropD_Calculate_MaxLat_nan(u[(lat > eq_b_n) & (lat < polar_b_n)],\
          lat[(lat > eq_b_n) & (lat < polar_b_n)], n=6)
      PhiSH = TropD_Calculate_MaxLat_nan(u[(lat > polar_b_s) & (lat < eq_b_s)],\
          lat[(lat > polar_b_s) & (lat < eq_b_s)], n=6)

  elif method == 'adjusted_peak':
    #PhiSH_EDJ, PhiNH_EDJ = TropD_Metric_EDJ(U,lat,lev)
    if n:
      PhiNH = TropD_Calculate_MaxLat_nan(u[(lat > eq_b_n) & (lat < polar_b_n)],\
          lat[(lat > eq_b_n) & (lat < polar_b_n)], n=n)
      PhiSH = TropD_Calculate_MaxLat_nan(u[(lat > polar_b_s) & (lat < eq_b_s)],\
          lat[(lat > polar_b_s) & (lat < eq_b_s)], n=n)
    else:
      PhiNH = TropD_Calculate_MaxLat_nan(u[(lat > eq_b_n) & (lat < polar_b_n)],\
          lat[(lat > eq_b_n) & (lat < polar_b_n)], n=30)
      PhiSH = TropD_Calculate_MaxLat_nan(u[(lat > polar_b_s) & (lat < eq_b_s)],\
          lat[(lat > polar_b_s) & (lat < eq_b_s)], n=30)

  elif method == 'adjusted_max':
    #PhiSH_EDJ,PhiNH_EDJ = TropD_Metric_EDJ(U,lat,lev)
    if n:
      PhiNH = TropD_Calculate_MaxLat_nan(u[(lat > eq_b_n) & (lat < polar_b_n)],\
          lat[(lat > eq_b_n) & (lat < polar_b_n)], n=n)
      PhiSH = TropD_Calculate_MaxLat_nan(u[(lat > polar_b_s) & (lat < eq_b_s)],\
          lat[(lat > polar_b_s) & (lat < eq_b_s)], n=n)
    else:
      PhiNH = TropD_Calculate_MaxLat_nan(u[(lat > eq_b_n) & (lat < polar_b_n)],\
          lat[(lat > eq_b_n) & (lat < polar_b_n)], n=30)
      PhiSH = TropD_Calculate_MaxLat_nan(u[(lat > polar_b_s) & (lat < eq_b_s)],\
          lat[(lat > polar_b_s) & (lat < eq_b_s)], n=30)

  return PhiSH, PhiNH, u


def find_STJ_jets(U, lat, max_lats, lev=np.array([1]), n=0, n_fit=1):
  '''TropD Eddy Driven Jet (EDJ) metric
       
     Latitude of maximum of the zonal wind at the level closest to the 850 hPa level
     
     Args:
       U (time,lat,lev) or U (time,lat,): Zonal mean zonal wind. Also takes surface wind 
       time: time vector
       lat : latitude vector
       lev: vertical level vector in hPa units

       method (str, optional): 'peak' (default) |  'max' | 'fit'
       
        peak (Default): Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=30)
        
        max: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=6)
        fit: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level using a quadratic polynomial fit of data from gridpoints surrounding the gridpoint of the maximum
        
       n (int, optional): If n is not set (0), n=6 (default) is used in TropD_Calculate_MaxLat. Rank of moment used to calculate the position of max value. n = 1,2,4,6,8,...  
     
     Returns:
       tuple: PhiSH (ndarray), PhiNH (ndarray) Latitude of EDJ in SH and NH

  '''

  try:
    assert (not hasattr(n, "__len__") and n >= 0)  
  except AssertionError:
   print('TropD_Metric_EDJ: ERROR : the smoothing parameter n must be >= 0')
   
  #try:
  #  assert(method in ['max','peak'])
  #except AssertionError:
  #  print('TropD_Metric_EDJ: ERROR : unrecognized method ', method)

  eq_boundary = 5
  polar_boundary = 90
  
  if len(lev) > 1:
    u = U[:,find_nearest(lev, 850)]
  else:
    u = np.copy(U)
  #if method == 'max':
      #Default value of n=6 is used
  jets_n = find_STJ_max(u[:,(lat > eq_boundary) & (lat < polar_boundary)],\
          lat[(lat > eq_boundary) & (lat < polar_boundary)],\
          max_lats[max_lats>0], lat_tol=20)
  jets_s = find_STJ_max(u[:,(lat > -polar_boundary) & (lat < -eq_boundary)],\
          lat[(lat > -polar_boundary) & (lat < -eq_boundary)],\
          max_lats[max_lats<0], lat_tol=20) 
  
  #else:
  #  print('TropD_Metric_EDJ: ERROR: unrecognized method ', method)

  return jets_n, jets_s

#Converted to python by Paul Staten Jul.29.2017
def find_STJ_max(F, lat, max_lats, lat_tol=15,mem=15,max_tol=15):

  ''' Find all local maxima of the function F

      Args:
  
        F: array

        lat: latitude array (same length as F)

        tol (float, optional): The minimal distance allowed between adjacent zero crossings of indetical sign change for example, for lat_uncertainty = 10, if the most equatorward zero crossing is from positive to negative, the function will return a NaN value if an additional zero crossings from positive to negative is found within 10 degrees of that zero crossing.

      Returns:

        float: latitude of local maxima
  '''
  #u_max = np.where(F == np.ma.max(F))[0][0] ## global maximum
  
  #x = np.full_like(F, fill_value=np.nan)
  z = []
  # Find maxima for all timesteps.
  for i in range(F.shape[0]):
    a = []
    try:
      u_max = np.where(F[i,:] == np.ma.max(F[i,:]))[0][0] ## global maximum
    except:
      z.append([np.nan])
      continue
    b = find_peaks(F[i,:])[0]
  
    for j in b:
      if F[i,j] <= 0:     # ensure eastward flow
        continue          # could include a min strength threshold as either a constant or dependent on max value
      elif np.abs(F[i,j]/u_max)*100 < max_tol:
        continue
      else:
        for k in range(len(max_lats)):
          if np.abs(lat[j] - max_lats[k]) < lat_tol:
            a.append(j)
            break
          else:
            continue
        
    z.append(a)
    #except:
    #  z.append(a)
    #a = a[np.abs((u_max - a)/u_max) < max_tol]
  
  w = 0
  y = []
  for i in range(F.shape[0]):
    x = []
    for j in range(len(z[i])):
      u_near = F[i,int(z[i][j])-1:int(z[i][j])+2]
      lats_near = lat[int(z[i][j])-1:int(z[i][j])+2]
      # Quartic fit, with smaller lat spacing
      coefs = np.ma.polyfit(lats_near,u_near,2)
      fine_lats = np.linspace(lats_near[0], lats_near[-1],200)
      quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2
      # Find jet lat
      a = fine_lats[np.where(quad == max(quad))[0][0]]
      x.append(a)
    y.append(x)
    if w < len(y[i]):
      w = len(y[i])
  w = w+mem

  #v = np.full((F.shape[0], len(max_lats), np.nan))
  z = np.full((F.shape[0], len(max_lats)), np.nan)
  for i in range(F.shape[0]):
    try:
      if len(max_lats) == 1:
        ind = np.argmin([np.abs(j-max_lats[0]) for j in y[i]])
        if np.abs(y[i][ind]-max_lats[0]) < lat_tol:
          z[i,0] = y[i][ind]
          del y[i][ind]
        else:
          continue
      else:
        for k in range(len(max_lats)):
          ind = np.argmin([np.abs(j-max_lats[k]) for j in y[i]])
          if np.abs(y[i][ind]-max_lats[k]) < lat_tol:
            z[i,k] = y[i][ind]
            del y[i][ind]
          else:
            continue
    except:
      continue
    
  return z