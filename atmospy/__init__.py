# -*- coding: utf-8 -*-
from .analysis_functions import filestrings, \
                                calc_jet_lat, \
                                stereo_plot, \
                                Calculate_ZeroCrossing, \
                                moving_average, \
                                new_cmap, lait, \
                                make_stereo_plot, \
                                get_nth_sol, \
                                get_timeslice, nf, \
                                calc_PV_max, \
                                moving_average_2d, \
                                get_init_sol, \
                                open_files, get_exps
from .tropd_exo import find_STJ_jets, \
                       TropD_Metric_Psi

from xcontour import Contour2D, \
                     latitude_lengths_at, \
                     add_latlon_metrics, \
                     get_planet_parameters

__version__ = "0.0.1"
