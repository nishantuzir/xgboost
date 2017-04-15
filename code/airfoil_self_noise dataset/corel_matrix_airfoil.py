# Correction Matrix Plot for airfoilselfnoise dataset

import matplotlib.pyplot as plt
import pandas
import numpy
url = "airfoil_self_noise.csv"
names = ['freq', 'angle', 'chord', 'skin','vel','thick']
data = pandas.read_csv(url, names=names)
correlations = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()