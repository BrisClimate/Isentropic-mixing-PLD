# PLD-analysis #

## Scripts used in analysis for Ball et al. (2023).

[atmospy/](https://github.com/emilyrball/PLD-analysis/tree/main/atmospy) contains useful calculation tools.

##### [atmospy/xcontour](https://github.com/emilyrball/PLD-analysis/tree/main/atmospy/xcontour) is adapted from [https://github.com/miniufo/xcontour](https://github.com/miniufo/xcontour), with thanks to the author for making their code available.

[mars_analysis/](https://github.com/emilyrball/PLD-analysis/tree/main/mars_analysis) contains scripts used to plot figures.

In the following files, savedata=False will allow the user to plot figures from the published data. To calculate various diagnostics from model output, savedata=True can be turned on. All files can be run using the plot_env environment ([plot_env.yml](https://github.com/emilyrball/PLD-analysis/blob/main/plot_env.yml)), unless otherwise specified.

### Figure 3
Plotted in [mars_analysis/polar_tracer_ratios.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/polar_tracer_ratios.py)

### Figure 4, Figure 5, Figure 6, Figure 7
Plotted in [mars_analysis/plot_keff_cross_sections.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/plot_keff_cross_sections.py)

### Figure 8, Figure 9
Plotted in [mars_analysis/plot_keff_lat.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/plot_keff_lat.py)

### Figure 10
Plotted in [mars_analysis/plot_summary_of_lats.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/plot_summary_of_lats.py)

### Figure 11, Figure 12
Plotted in [mars_analysis/plot_HC.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/plot_HC.py)

### Figure 13, Figure 14
Plotted in [mars_analysis/plot_jet.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/plot_jet.py)

### Figure 15
Plotted in [mars_analysis/plot_PV.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/plot_PV.py)

## To calculate effective diffusivity from model output...

The gradient of tracer is calculated in [mars_analysis/add_gradStracer.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/add_gradStracer.py) (either on pressure or theta levels)

Effective diffusivity is then calculated in [mars_analysis/calculate_keff.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/calculate_keff.py) (either on pressure or theta levels; this should be run in keff environment using [keff.yml](https://github.com/emilyrball/PLD-analysis/blob/main/keff.yml))

This is then saved all in one place using the script [mars_analysis/concat_eps_gamma.py](https://github.com/emilyrball/PLD-analysis/blob/main/mars_analysis/concat_eps_gamma.py)

