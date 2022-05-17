# plotting parameters
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

titlesize = 11
fontsize = 10
cm = 1/2.54
figwidth = 14.7*cm
dpi = 1000
params = {'legend.fontsize': 9,
          'legend.title_fontsize': 10,
          'font.family': 'serif',
         'axes.labelsize': fontsize,
         'axes.titlesize':titlesize,
         'xtick.labelsize':9,
         'ytick.labelsize':9,
         'savefig.bbox': 'tight',
         'savefig.dpi': 400,
         'savefig.pad_inches':0}
plt.rcParams.update(params)
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

colors_healthy = ['tab:blue', 'darkblue', 'gray']
colors_patients = ['tab:red', 'darkred', 'gray']
colors = [colors_healthy, colors_patients]