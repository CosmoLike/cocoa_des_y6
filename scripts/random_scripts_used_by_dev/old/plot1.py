"""This script compares two cobaya chains in a GetDist triangle plot.

It reads the chains EXAMPLE_MCMC3 and EXAMPLE_MCMC1 from ../chains, relative
to the working directory, drops the first half of each as burn-in, adds
derived parameters, saves each processed chain as a hidden text file in the
working directory (.VM_P1_TMP1 and .VM_P1_TMP2), and draws the triangle plot
of the parameters listed in `parameter` from those copies. A triangle plot
shows the 1D posterior of each parameter on its diagonal and the 2D contours
of each pair below it. The figure is saved as plot1.pdf.

The script comes from the LSST-Y1 project: `parameter` names LSST_A1_1 and
LSST_A1_2 and the legend labels read "LSST-Y1 Cosmic Shear EE2" and "LSST-Y1
Cosmic Shear", while the DES Y6 chains name their IA amplitudes DES_A1_1 and
DES_A1_2.

Run it from a folder whose parent holds chains/, for example
projects/des_y6/scripts: python random_scripts_used_by_dev/old/plot1.py
"""
import getdist.plots as gplot
from getdist import MCSamples
from getdist import loadMCSamples
import os
import matplotlib
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# Figure style: STIX fonts for text and mathematics, a light grid, no top
# or right ticks, and tight PDF output for saved figures.
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'

# Parameters of the triangle plot, in panel order (SS8 is derived below);
# LSST_A1_1 and LSST_A1_2 are LSST-Y1 names (module docstring).
parameter = [u'omegam', u'sigma8', u'As_1e9', u'ns', u'SS8', u'omegab', u'H0', u'w', u'LSST_A1_1', u'LSST_A1_2']
# The chains are read from ../chains relative to the working directory,
# and the processed copies are written into the working directory.
chaindir=os.getcwd()

# GetDist analysis settings: Gaussian smoothing widths of 0.35 (1D) and
# 0.3 (2D) standard deviations of each parameter, the first half of each
# chain dropped as burn-in (ignore_rows = 0.5), and plot ranges between
# the 0.5% and 99.5% quantiles (range_confidence = 0.005). The second set
# drops nothing, because the saved copies have no burn-in left.
analysissettings={'smooth_scale_1D':0.35, 'smooth_scale_2D':0.3,'ignore_rows': u'0.5',
'range_confidence' : u'0.005'}

analysissettings2={'smooth_scale_1D':0.35,'smooth_scale_2D':0.3,'ignore_rows': u'0.0',
'range_confidence' : u'0.005'}

root_chains = (
  'EXAMPLE_MCMC3',
  'EXAMPLE_MCMC1',
)

# --------------------------------------------------------------------------------
# For each chain: load it, add derived parameters, save it as text.
# gamma = Omega_m h; SS8 = S_8 = sigma_8 (Omega_m/0.3)^0.5
# = s8omegamp5/sqrt(0.3), with sqrt(0.3) = 0.5477225575; om10, ob100 and
# ns10 rescale Omega_m, Omega_b and n_s for readable axis labels.
samples=loadMCSamples(chaindir + '/../chains/' + root_chains[0],settings=analysissettings)
p = samples.getParams()
samples.addDerived(p.omegam*p.H0/100.,name='gamma',label='{\\Omega_m h}')
samples.addDerived(p.s8omegamp5/0.5477225575,name='SS8',label='{S_8}')
samples.addDerived(10*p.omegam,name='om10',label='{10 \\Omega_m}')
samples.addDerived(100*p.omegab,name='ob100',label='{100 \\Omega_b}')
samples.addDerived(10*p.ns,name='ns10',label='{10 n_s}')
samples.saveAsText(chaindir + '/.VM_P1_TMP1')
# --------------------------------------------------------------------------------
samples=loadMCSamples(chaindir + '/../chains/' + root_chains[1],settings=analysissettings)
p = samples.getParams()
samples.addDerived(p.omegam*p.H0/100.,name='gamma',label='{\\Omega_m h}')
samples.addDerived(p.s8omegamp5/0.5477225575,name='SS8',label='{S_8}')
samples.addDerived(10*p.omegam,name='om10',label='{10 \\Omega_m}')
samples.addDerived(100*p.omegab,name='ob100',label='{100 \\Omega_b}')
samples.addDerived(10*p.ns,name='ns10',label='{10 n_s}')
samples.saveAsText(chaindir + '/.VM_P1_TMP2')
# --------------------------------------------------------------------------------


# GetDist triangle plotter, 12.5 inches wide, with its fonts, contour line
# widths and legend style; analysissettings2 applies to the saved copies.
g=gplot.getSubplotPlotter(chain_dir=chaindir,
  analysis_settings=analysissettings2,width_inch=12.5)
g.settings.axis_tick_x_rotation=65
g.settings.lw_contour = 1.2
g.settings.legend_rect_border = False
g.settings.figure_legend_frame = False
g.settings.axes_fontsize = 13.0
g.settings.legend_fontsize = 13.5
g.settings.alpha_filled_add = 0.85
g.settings.lab_fontsize=15.5
g.legend_labels=False

print(chaindir)

# No third parameter colors the samples. The first chain is drawn filled
# in light coral, the second as black dashed contours; the third and
# fourth style entries are unused with two chains.
param_3d = None
g.triangle_plot([chaindir + '/.VM_P1_TMP1',chaindir + '/.VM_P1_TMP2'],
parameter,
plot_3d_with_param=param_3d,line_args=[
{'lw': 1.2,'ls': 'solid', 'color':'lightcoral'},
{'lw': 1.2,'ls': '--', 'color':'black'},
{'lw': 1.6,'ls': '-.', 'color': 'maroon'},
{'lw': 1.6,'ls': 'solid', 'color': 'indigo'},
],
contour_colors=['lightcoral','black','maroon','indigo'],
contour_ls=['solid','--','-.'], 
contour_lws=[1.0,1.5,1.5,1.0],
filled=[True,False,False,True],
shaded=False,
legend_labels=[
'LSST-Y1 Cosmic Shear EE2',
'LSST-Y1 Cosmic Shear',
],
legend_loc=(0.48, 0.80))


# With no file name, GetDist saves the figure as <script name>.pdf in the
# working directory.
g.export()