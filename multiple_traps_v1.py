# import functions from directory
import numpy as np
from holograms.zernike import zernike
from holograms.gratings import hori_gradient, hori
from holograms.apertures import circ
from holograms.lenses import focal_plane_shift
from holograms.arrays import aags
from camera import ImageHandler,Camera
import matplotlib.pyplot as plt
from slm import SLM
import time

# CONFIGURATION
#position of tweezers
X = [252, 256, 260]
Y = [256, 256, 256]
X0 = 233
Y0 = 251

#parameters of set-up /SLM
SLM_SHAPE = (512,512)
SLM_PIXEL_SIZE = 15e-6
WAVELENGTH = 1064e-9
APERTURE_RADIUS = 233
POLYNOMIAL_RADIUS = 100
HORI_GRADIENT = -7.3
HORI_PERIOD = 7
LENS_SHIFT = -3
final_holo = []

#set variables for run
amplitude_range = np.arange(-1,1,0.05)
exposure = 60
repeat = 3 

#set polynomials to loop through. r is radial coord, a is azimuthal 
radial_coords = [4,4,4,4,4,3,3,3,3,2,2,2,1,1,0]
azimuthal_coords = [4,2,0,-2,-4,3,1,-1,-3,2,0,-2,1,-1,0]

# END CONFIGURATION

def main():
    #initialise classes
    camera = Camera() #TO DO: check ROI gives correct image
    save_image = ImageHandler()
    slm = SLM()

    assert len(radial_coords) == len(azimuthal_coords), "Radial and azimuthal coords should have same length"

    #take image 0 as background image for comparison
    camera.update_exposure(exposure)
    image = camera.take_image()
    save_image.save(image)

    #set beam array
    trap_hologram = aags(traps= ((X[0],Y[0]), (X[1],Y[1]),(X[2],Y[2])),
                            iterations=20, #must be greater than no. of traps
                            beam_waist=None,
                            beam_center=(256,256),
                            shape=(512,512))

    #set grating and lens holograms
    hor_grating_1 = hori_gradient(gradient=HORI_GRADIENT)
    hor_grating_2 = hori(period = HORI_PERIOD)
    lens = focal_plane_shift(
        shift=LENS_SHIFT,
        x0=X0,
        y0=Y0,
        wavelength=WAVELENGTH,
        pixel_size=SLM_PIXEL_SIZE,
        shape=SLM_SHAPE)
                
    
 
    # loop through Zernike Polynomials and apply one centering on each beam:
    for radial,azimuthal in zip(radial_coords, azimuthal_coords):

    #camera = None # Camera()
        

    #loop through amplitude range
        for amplitude in amplitude_range:
            # loop to apply polynomial centered on each beam
            zernike_hologram = 0
            for i in range(0,len(X),1):
                zernike_polynom = zernike(
                radial=radial,
                azimuthal=azimuthal,
                amplitude=amplitude,
                x0=X[i],
                y0=Y[i],
                radius=None,
                shape=SLM_SHAPE)
                zernike_hologram = zernike_hologram + zernike_polynom

            x = hor_grating_1 + hor_grating_2 + lens + zernike_hologram + trap_hologram
                    
            #apply circular aperture
            holo = circ(
                x,
                x0 = X0,
                y0 = Y0,
                radius = APERTURE_RADIUS)

            #apply hologram to SLM and give time to load
            slm.apply_hologram(holo)

            time.sleep(0.5)

             #loop to take three sets of images for averaging
            for repeat in range(0,3,1):

                #use camera class to take and save photos
                camera.update_exposure(exposure)
                image = camera.take_image()
                save_image.save(image)
                
                
                repeat = repeat + 1

        # Delete causes camera disconnect
        #del camera

        # Sleep for 1 second to allow for side-effects
        # such as the camera releasing its handles
        # to the OS
        #time.sleep(1)

            
if __name__ == "__main__":
    main()
