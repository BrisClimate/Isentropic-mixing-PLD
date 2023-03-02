'''
Script to create solar longitude coordinate, and MY if necessary


'''

# %%
import xarray as xr
import numpy as np
import sys, os

sys.path.append('/user/home/xz19136/Py_Scripts/atmospy/')

import matplotlib.pyplot as plt
import analysis_functions as funcs


path = '/user/work/xz19136/Isca_data/'

if __name__ == "__main__":
    eps = 25
    gamma = 0.093

    tname = 'test_tracer'

    exp_name = 'tracer_soc_mars_mola_topo_lh_eps_' + \
                '%i_gamma_%.3f_cdod_clim_scenario_7.4e-05' % (eps, gamma)
    ds = xr.open_dataset(
        path + exp_name + '/keff_%s.nc' % tname, decode_times = False,)

    d = xr.open_dataset(
        path + exp_name + '/atmos.nc', decode_times = False,)

    print(d.mars_solar_long.values)
    