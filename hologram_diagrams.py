import numpy as np
from holograms.zernike import zernike
from holograms.gratings import hori_gradient, vert_gradient, hori
from holograms.apertures import circ
from holograms.lenses import focal_plane_shift
from holograms.mixing import mix
import matplotlib.pyplot as plt
from slm import SLM


x =  zernike(radial=4,azimuthal=4,amplitude=-1,radius=None,shape=(512,512)) + focal_plane_shift(shift=-3,wavelength=1064e-9,pixel_size=15e-6,shape=(512,512)) + hori(period= 7, shape=(512,512)) + hori_gradient(gradient= -7.3, shape=(512,512))
holo = circ(x,radius = 233)


plt.figure(figsize=(5,5))
plt.pcolor(holo, cmap='viridis')
cb = plt.colorbar(label='Normalized Phase Change / $\mathrm{\pi}$ rads')

# for l in cb.ax.yaxis.get_ticklabels():
#    l.set_family("Times New Roman")

#plt.xlabel('Pixels on X axis of SLM')
#plt.ylabel('Pixels on Y axis of SLM')
#plt.title(r'ABC123 vs $\mathrm{ABC}^{123}$')
#plt.title(r'Z$\mathrm{^{2}_4}$ - Vertical secondary astigmatism', fontsize=20, **csfont)
plt.axis('off')
plt.show()