import numpy as np
from holograms.zernike import zernike, R
from holograms.gratings import hori_gradient, vert_gradient, hori
from holograms.apertures import circ
from holograms.lenses import focal_plane_shift
from holograms.arrays import aags
from matplotlib import gridspec
from holograms.mixing import mix
import matplotlib.pyplot as plt
from slm import SLM

def zernike_decomposition(hologram, 
                        test_zernike_radial,
                        test_zernike_azimuthal,
                        test_amplitude, 
                        circ_aperture = True, 
                        x0 = None , 
                        y0 = None , 
                        radius = None, 
                        shape = (512, 512)):


    #if x0, y0 and radius are None, define as x0,y0 as centre of hologram and radius as max radius that can fit on SLM. 
    if x0 is None:
        x0 = shape[0]/2
    if y0 is None:
        y0 = shape[1]/2
    if radius is None:
        radius = min([shape[0]-x0,shape[1]-y0,x0,y0])

    #define grid and coords
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    xx,yy = np.meshgrid(x,y)
    r = np.sqrt((xx-x0)**2+(yy-y0)**2)
    r /= radius
    print('r = ' + str(r))
    phi = np.arctan2(yy-y0,xx-x0)
    print('phi = ' + str(phi))

    #remove circular aperture if there is one present 
    #Zernike function automatically adds this, aags does not so set accordingly.

    if circ_aperture == True:
        hologram -= circ(hologram, x0, y0, radius)

    #scale by amplitude
    hologram /= test_amplitude

    if test_zernike_azimuthal >= 0:

        hologram /= (R(r,test_zernike_radial,test_zernike_azimuthal)*np.cos(test_zernike_azimuthal*phi))

    else:
        hologram /= (R(r,test_zernike_radial,-test_zernike_azimuthal)*np.sin(-test_zernike_azimuthal*phi))
    
    return hologram

hologram = zernike(radial=0,azimuthal=0,amplitude=1,x0=None,y0=None,radius=None,shape=(512,512))

print(zernike_decomposition(hologram, 0, 0, 0.5, 
                        circ_aperture = True, 
                        x0 = None , 
                        y0 = None , 
                        radius = None, 
                        shape = (512, 512)))

