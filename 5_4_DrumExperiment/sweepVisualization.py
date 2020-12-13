
import numpy as np
import matplotlib.pyplot as plt
from Math.NadarayaWatson2D import GaussKernel2D
from matplotlib.gridspec import GridSpec
from config import load_configurations

tmp = np.load('results/sweep_data.npz')
current_extension = tmp['arr_0']
current_results = tmp['arr_1']

print(len(current_extension))

current_coordinates = np.zeros([len(current_extension), 2])
idx_count_error = 0
for kk in range(len(current_coordinates)):

    config = load_configurations(current_extension[kk])
    current_coordinates[kk,:] = [5*config['smoothing_width'], 1000*config['label_noise']]
    if config['extension']!=current_extension[kk]:
        print('ConfigFile not found for index: ', kk)
        idx_count_error+=0

if idx_count_error>0:
    print('Errors:', idx_count_error)

pt_color = np.repeat('r',current_results.shape[0])
pt_color[np.mean(np.mean(current_results[:,:,:],axis=2),axis=1)>0.85]='y'
pt_color[np.mean(np.mean(current_results[:,:,:],axis=2),axis=1)>0.9]='b'
pt_color[np.mean(np.mean(current_results[:,:,:],axis=2),axis=1)>0.95]='g'


granularity = 0.2
sigma_range = np.arange(0,150+1,granularity)
noise_range = np.arange(0,100+1,granularity)
xx,yy = np.meshgrid(sigma_range, noise_range)

truncate = 0.83
current_results[current_results<truncate]=truncate

grid = np.vstack([xx.ravel(), yy.ravel()]).T
heat_maps = np.zeros([3, len(noise_range),len(sigma_range)])
for kk in range(3):
    result = GaussKernel2D(grid, current_coordinates, np.mean(current_results[:, :, kk], axis=1), 10)
    heat_maps[kk,:,:] = result.reshape(len(noise_range),len(sigma_range))

from Display.divergingMaps import divergingMaps
min_display = 0.85
max_display = 0.963
heat_maps[heat_maps < min_display]=min_display

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

fig = plt.figure(figsize=(12,5))
gs=GridSpec(3,25)
ax=fig.add_subplot(gs[:,:16])

img = ax.pcolormesh(sigma_range, noise_range, np.mean(heat_maps,axis=0), cmap='RdBu', vmin=min_display, vmax=max_display)

plt.ylabel(r'\textbf{Noise $\boldsymbol{\sigma}$ [ms]}', fontweight='bold', fontsize=13)
plt.xlabel(r'\textbf{Softness $\boldsymbol{\mathcal{S}_{M}}$  [ms]}', fontweight='bold', fontsize=13)

ax.tick_params(width=2)

plt.scatter(current_coordinates[:,0],current_coordinates[:,1], c='w', s=2)

plt.xlim([0,151])
plt.ylim([0,101])

# Process ticks
print([item.get_text() for item in ax.get_xticklabels()])
labels = ['\\textbf{' + item.get_text()[1:-1] + '}' for item in ax.get_xticklabels()]
ax.set_xticklabels(labels)
labels = ['\\textbf{' + item.get_text()[1:-1] + '}' for item in ax.get_yticklabels()]
ax.set_yticklabels(labels)

plt.box('off')


ax2 = plt.subplot(gs[:, 17])
cb = plt.colorbar(img, cax=ax2, extend='both')
labels = ['\\textbf{' + item.get_text()[1:-1] + '}' for item in cb.ax.get_yticklabels()]
cb.ax.set_yticklabels(labels)
ax2.text(1.5,-0.05,"F_{1}", fontweight="bold", fontsize=18)
ax2.text(1.52,-0.05,"F_{1}", fontweight="bold", fontsize=18)

plt.savefig('heatmap.png')
plt.show()



