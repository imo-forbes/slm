
import numpy as np
from holograms.zernike import zernike
import matplotlib.pyplot as plt


zernike_holo = zernike(radial=4,azimuthal=2,amplitude=1,x0=None,y0=None,radius=None,shape=(512,512))

csfont = {'fontname':'Times New Roman'}

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'cm'



plt.pcolor(zernike_holo, cmap='viridis')
cb = plt.colorbar(label='Normalized Phase Change / 2$\mathrm{\pi}$ rads')

for l in cb.ax.yaxis.get_ticklabels():
    l.set_family("Times New Roman")

#plt.xlabel('Pixels on X axis of SLM')
#plt.ylabel('Pixels on Y axis of SLM')
#plt.title(r'ABC123 vs $\mathrm{ABC}^{123}$')
plt.title(r'Z$\mathrm{^{2}_4}$ - Vertical secondary astigmatism', fontsize=20, **csfont)
plt.axis('off')
plt.show()