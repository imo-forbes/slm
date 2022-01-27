
import numpy as np
from holograms.zernike import zernike
from holograms.mixing import mix
import matplotlib.pyplot as plt


zernike_holo_3_neg3 = zernike(radial=3,azimuthal=-3,amplitude=0.1,x0=None,y0=None,radius=None,shape=(512,512))
zernike_holo_4_neg4 = zernike(radial=4,azimuthal=-4,amplitude=0.02,x0=None,y0=None,radius=None,shape=(512,512))
zernike_holo_4_neg2= zernike(radial=4,azimuthal=-2,amplitude=0.02,x0=None,y0=None,radius=None,shape=(512,512))
zernike_holo_4_0 = zernike(radial=4,azimuthal=0,amplitude=0.14,x0=None,y0=None,radius=None,shape=(512,512))
zernike_holo_4_2 = zernike(radial=4,azimuthal=2,amplitude=-0.6,x0=None,y0=None,radius=None,shape=(512,512))
zernike_holo_4_4 = zernike(radial=4,azimuthal=4,amplitude=0.37,x0=None,y0=None,radius=None,shape=(512,512))

csfont = {'fontname':'Times New Roman'}

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'cm'


x = zernike_holo_4_4 + zernike_holo_4_2 + zernike_holo_4_0 + zernike_holo_4_neg2 +zernike_holo_4_neg4 + zernike_holo_3_neg3
h = mix(x)



plt.pcolor(x, cmap='viridis')
cb = plt.colorbar(label='Normalized Phase Change / $\mathrm{\pi}$ rads')

for l in cb.ax.yaxis.get_ticklabels():
    l.set_family("Times New Roman")

#plt.xlabel('Pixels on X axis of SLM')
#plt.ylabel('Pixels on Y axis of SLM')
#plt.title(r'ABC123 vs $\mathrm{ABC}^{123}$')
#plt.title(r'Z$\mathrm{^{2}_4}$ - Vertical secondary astigmatism', fontsize=20, **csfont)
plt.axis('off')
plt.show()