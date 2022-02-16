import numpy as np
from sympy import O
from holograms.zernike import zernike, R
from holograms.gratings import hori_gradient, vert_gradient, hori
from holograms.apertures import circ
from holograms.lenses import focal_plane_shift
from holograms.arrays import aags
from matplotlib import gridspec
from holograms.mixing import mix
import matplotlib.pyplot as plt
from slm import SLM
from scipy.optimize import minimize


#same as Zernike polynomial function in SLM code but removes the amplitude factor and the circular aperture
def zernike_no_aperture_no_amp(radial=0,azimuthal=0,x0=None,y0=None,radius=None,shape=(512,512)):

    if azimuthal > radial:
        raise ValueError("azimuthal must be less than or equal to radial")
    if x0 is None:
        x0 = shape[0]/2
    if y0 is None:
        y0 = shape[1]/2
    if radius is None:
        radius = min([shape[0]-x0,shape[1]-y0,x0,y0])
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    xx,yy = np.meshgrid(x,y)
    r = np.sqrt((xx-x0)**2+(yy-y0)**2)
    r /= radius
    phi = np.arctan2(yy-y0,xx-x0)
    if azimuthal >= 0:
        phase = R(r,radial,azimuthal)*np.cos(azimuthal*phi)
    else:
        phase = R(r,radial,-azimuthal)*np.sin(-azimuthal*phi)

    return phase 

def zernike_no_aperture(amplitude = 1, radial=0,azimuthal=0,  x0=None,y0=None,radius=None,shape=(512,512)):

    if azimuthal > radial:
        raise ValueError("azimuthal must be less than or equal to radial")
    if x0 is None:
        x0 = shape[0]/2
    if y0 is None:
        y0 = shape[1]/2
    if radius is None:
        radius = min([shape[0]-x0,shape[1]-y0,x0,y0])
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    xx,yy = np.meshgrid(x,y)
    r = np.sqrt((xx-x0)**2+(yy-y0)**2)
    r /= radius
    phi = np.arctan2(yy-y0,xx-x0)
    if azimuthal >= 0:
        phase = R(r,radial,azimuthal)*np.cos(azimuthal*phi)
    else:
        phase = R(r,radial,-azimuthal)*np.sin(-azimuthal*phi)
    phase *= amplitude

    return phase 

def sum_zernike(coeffs):
    total_sum = 0

    for coeff, zernike in zip(coeffs, CACHED_ZERNIKE):
        total_sum += coeff * zernike
    
    return total_sum
    

def zernike_decomposition(trap_hologram,  x0=None, y0=None, radius=None, shape=(512,512)):

    def objective_func(x):
        return np.sum((trap_hologram - sum_zernike(x))**2)

    amplitudes = minimize(objective_func, x0=np.zeros(len(ZERNIKE_COORDINATES)), bounds=[(-1, 1) for i in range(len(ZERNIKE_COORDINATES))])

    return amplitudes

ZERNIKE_COORDINATES = (
    # radial, azimuthal
    (0,0),
    (1,1),
    (1,-1),
    (2,2),
    (2,0),
    (2,-2),
    (3,3),
    (3,1),
    (3,-1),
    (3,-3),
    (4,4),
    (4,2),
    (4,0),
    (4,-2),
    (4,-4)
)

CACHED_ZERNIKE = [zernike_no_aperture_no_amp(rad, azi) for rad,azi in ZERNIKE_COORDINATES]

trap_hologram = zernike_no_aperture_no_amp(4,4)


print(zernike_decomposition(trap_hologram))

final = 0
for i in range(len(amplitudes)):
    rad, azi = ZERNIKE_COORDINATES[i]
    final = final + zernike_no_aperture(amplitudes[i], rad, azi)


plt.pcolor(final)
plt.show()
