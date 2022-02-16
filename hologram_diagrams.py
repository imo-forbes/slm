import numpy as np
from holograms.zernike import zernike
from holograms.gratings import hori_gradient, vert_gradient, hori
from holograms.apertures import circ
from holograms.lenses import focal_plane_shift
from holograms.arrays import aags
from matplotlib import gridspec
from holograms.mixing import mix
import matplotlib.pyplot as plt
from slm import SLM


X = [236, 256, 276, 236, 256, 276, 236, 256, 276]
Y = [236,236,236,256,256,256,276,276,276]

radial = [0,1,1,2,2,2,3,3,3,3,4,4,4,4,4]
azimuthal = [0,1,-1,0,-2,2,-1,1,-3,3,0,2,-2,4,-4]
amplitude = [ 0.000,0.016,0.016,-0.043,-0.044,-0.000,-0.011,-0.011,-0.008,0.008, 0.069,0.000,0.057 , 0.128 ,-0.000]



trap_hologram = aags(traps = ((X[0],Y[0]), (X[1],Y[1]),(X[2],Y[2]), (X[3],Y[3]), (X[4],Y[4]),(X[5],Y[5]), (X[6],Y[6]), (X[7],Y[7]), (X[8],Y[8])),iterations=30,
         beam_waist=None,beam_center=(256,256),shape=(512,512))#,circ_aper_center=None,):

MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
fig = plt.figure(figsize=(9,8))

# x_res_ax = fig.add_subplot(gs[-2, :-1])
# y_res_ax = fig.add_subplot(gs[-1, :-1])
# #x_hist_ax = fig.add_subplot(gs[-2, -1])
# #y_hist_ax = fig.add_subplot(gs[-1, -1])

# #x_hist_ax.get_yaxis().set_visible(False)
# #y_hist_ax.get_yaxis().set_visible(False)
# #x_hist_ax.get_xaxis().set_visible(False)
# #y_hist_ax.get_xaxis().set_visible(False)
# residual_y_lim = (-6, 7)

# x_res_ax.set_ylim(residual_y_lim)

# y_res_ax.set_ylim(residual_y_lim)
# x_res_ax.set_yticks([-5, 0,5])
# y_res_ax.set_yticks([-5, 0,5])


plt.pcolor(trap_hologram)
plt.colorbar()
plt.savefig('images_for_seminar/seminar_9_traps.png', transparent = True)
plt.show()

# print(trap_hologram)


# output=[]
# np.set_printoptions(threshold=np.inf,suppress=True)
# print(trap_hologram)
# print(repr(trap_hologram))

# zernike_holo=0
# for i in range(len(radial)):
#     holo = zernike(radial = radial[i], azimuthal = azimuthal[i], amplitude= amplitude[i], radius = 5, shape=(5,5))
#     zernike_holo = zernike_holo + holo
    




# # # x =  zernike(radial=4, azimuthal=4,amplitude=-1,radius=None,shape=(512,512)) + focal_plane_shift(shift=-5,wavelength=1064e-9,pixel_size=15e-6,shape=(512,512)) + hori(period= 7, shape=(512,512)) + hori_gradient(gradient= -7.3, shape=(512,512)) + trap_hologram # + vert_gradient(gradient = 6.98)

# # # holo = circ(x%1,radius = 233)


# plt.figure(figsize=(5,5))
# plt.pcolor(zernike_holo%1, cmap='viridis')
# cb = plt.colorbar(label='Normalized Phase Change / $\mathrm{\pi}$ rads')
# print(zernike_holo%1)

# # # for l in cb.ax.yaxis.get_ticklabels():
# # #    l.set_family("Times New Roman")

# # #plt.xlabel('Pixels on X axis of SLM')
# # #plt.ylabel('Pixels on Y axis of SLM')
# # #plt.title(r'ABC123 vs $\mathrm{ABC}^{123}$')
# # #plt.title(r'Z$\mathrm{^{2}_4}$ - Vertical secondary astigmatism', fontsize=20, **csfont)
# # plt.axis('off')
# plt.show()