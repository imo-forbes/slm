import pandas as pd
import numpy as np
from holograms.zernike import zernike
from holograms.mixing import mix
import matplotlib.pyplot as plt


SMALL_SIZE = 11
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


r = [4,4,4,4,4,3,3,3,3,2,2,]
a = [4,2,0,-2,-4,3,1,-1,-3,2,-2]
amplitude = [0.5,0.4,-0.23,0.03,0,-0.2,0.3,0.07,-0.07,-0.33,0]
#[0.15000000000000102, 8.881784197001252e-16, -0.0999999999999992, -0.049999999999999156, 8.881784197001252e-16, -0.049999999999999156, -0.049999999999999156, -0.0999999999999992, 8.881784197001252e-16, 0.05000000000000093, 8.881784197001252e-16, ]

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

zernike_holo=0
for i in range(1,len(r)):
    x = zernike(radial=r[i],azimuthal=a[i],amplitude=amplitude[i],x0=233,y0=251,radius=None,shape=(512,512))
    print(r[i], a[i], amplitude[i])
    zernike_holo = zernike_holo + x

fig = plt.figure(figsize=(6,6))

#plot settings for seminar graph
ax = fig.add_subplot(111)

ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

params = {"text.color" : "white",
          "xtick.color" : "white",
          "ytick.color" : "white",}
plt.rcParams.update(params)

plt.pcolor(zernike_holo, cmap='viridis')
cb = plt.colorbar(label='Normalized Phase Change / $\mathrm{\pi}$ rads')
cb.set_label('Normalized Phase Change / $\mathrm{\pi}$ rads', color="white")
fig.savefig('images_for_seminar/seminar_colormap.png', transparent=True)
plt.show()