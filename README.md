# PLD-analysis #

## Scripts used in analysis for Ball et al. (2023).

[atmospy/](https://github.com/emilyrball/PLD-analysis/tree/main/atmospy) contains useful calculation tools.
[pot_vort/](https://github.com/emilyrball/PLD-analysis/tree/main/pot_vort) contains tools for calculating potential vorticity.

##### [atmospy/xcontour](https://github.com/emilyrball/PLD-analysis/tree/main/atmospy/xcontour) is adapted from [https://github.com/miniufo/xcontour](https://github.com/miniufo/xcontour), with thanks to the author for making their code available.

[mars_analysis/](https://github.com/emilyrball/PLD-analysis/tree/main/mars_analysis) contains scripts used to plot figures.

In the following files, savedata=False will allow the user to plot figures from the published data. To calculate various diagnostics from model output, savedata=True can be turned on. All files can be run using the plot_env environment ([plot_env.yml](https://github.com/emilyrball/PLD-analysis/blob/main/plot_env.yml)), unless otherwise specified.

### Figure 3
Plotted in [mars_analysis/plot_HC_anomaly.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/plot_HC_anomaly.py) using the function plot_psi_cross_section_one

### Figure 4, Figure 5
Plotted in [mars_analysis/plot_HC_anomaly.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/plot_HC_anomaly.py) using the functions plot_HC_edge_evolution and plot_HC_strength_evolution

### Figure 6, Figure 7
Plotted in [mars_analysis/plot_jet_anomaly.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/plot_jet_anomaly.py) using the functions plot_jet_lat_evolution and plot_jet_strength_evolution

### Figure 8, Figure 9
Plotted in [mars_analysis/plot_PV_anomaly.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/plot_PV_anomaly.py) using the functions plot_PV_lat_evolution and plot_PV_max_evolution

### Figure 10
Plotted in [mars_analysis/polar_tracer_ratios.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/polar_tracer_ratios.py)

### Figure 11, Figure 12, Figure 13
Plotted in [mars_analysis/plot_keff_cross_sections.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/plot_keff_cross_sections.py)

### Figure 14, Figure 15
Plotted in [mars_analysis/plot_keff_lat.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/plot_keff_lat.py)

### Figure 16
Plotted in [mars_analysis/plot_summary_of_lats.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/plot_summary_of_lats.py)

### Figure 17
Plotted in [mars_analysis/plot_deposition_timeseries.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/plot_deposition_timeseries.py)


## Supplementary Information Figures
### Figure 1-6
Plotted in [mars_analysis/plot_PV_maps_anomaly.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/plot_PV_maps_anomaly.py)
These should be run in the environment plotting using [plotting.yml]((https://github.com/emilyrball/PLD-analysis/blob/main/keff.yml)

### Figure 7-12
Plotted in [mars_analysis/plot_HC_anomaly.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/plot_HC_anomaly.py) using the function plot_psi_cross_section

## To calculate effective diffusivity from model output

The gradient of tracer is calculated in [mars_analysis/add_gradStracer.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/add_gradStracer.py) (either on pressure or theta levels)

Effective diffusivity is then calculated in [mars_analysis/calculate_keff.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/calculate_keff.py) (either on pressure or theta levels; this should be run in keff environment using [keff.yml](https://github.com/emilyrball/PLD-analysis/blob/main/keff.yml))

This is then all saved in one place using the script [mars_analysis/concat_eps_gamma.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/concat_eps_gamma.py)

