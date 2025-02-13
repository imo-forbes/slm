# import functions from directory
import numpy as np
from holograms.zernike import zernike
from holograms.gratings import hori_gradient, hori, vert_gradient
from holograms.apertures import circ
from holograms.arrays import aags


from holograms.lenses import focal_plane_shift
from camera import ImageHandler,Camera
import matplotlib.pyplot as plt
from slm import SLM
import time

# CONFIGURATION
X = [256,251,246,241,261,266,271,251,261,256]
Y = [256,251,246,241,251,246,241,246,246,246]


X0 = 233
Y0 = 251
SLM_SHAPE = (512, 512)
SLM_PIXEL_SIZE = 15e-6
WAVELENGTH = 1064e-9
APERTURE_RADIUS = 233
HORI_GRADIENT = -8.76
HORI_PERIOD = 7
LENS_SHIFT = 4
VERT_GRADIENT= 6.98

#set variables for run
amplitude_range = np.arange(-0.5,0.5,0.1)

exposure = 1.5
repeat = 0 

#set polynomials to loop through. r is radial coord, a is azimuthal 
radial_coords = [4,4]
azimuthal_coords = [4,4]


# END CONFIGURATION


def main():
    #initialise classes

    #camera = Camera(roi =[595,305,635,320]) #ROI NEEDS UPDATED AS TRAP SPACING INCREASED

    # camera = Camera()
    # save_image = ImageHandler()

    slm = SLM()

    assert len(radial_coords) == len(azimuthal_coords), "Radial and azimuthal coords should have same length"

    #run algorithm to set trap locations 
    
    trap_hologram = aags(traps = ((X[0],Y[0]),(X[1],Y[1]),(X[2],Y[2]), (X[3],Y[3]), (X[4],Y[4]),(X[5],Y[5]), (X[6],Y[6]), (X[7],Y[7]), (X[8],Y[8]), (X[9],Y[9])), #(X[10], Y[10])),
                        iterations=30, #must be greater than no. of traps
                        beam_waist=None,
                        beam_center=(256,256),
                        shape=(512,512)) 

    #grating and lens for sum
    hor_grating_1 = hori_gradient(gradient=HORI_GRADIENT)
    hor_grating_2 = hori(period = HORI_PERIOD)
    ver_grating = vert_gradient(VERT_GRADIENT)
    lens = focal_plane_shift(
                    shift=LENS_SHIFT,
                    x0=X0,
                    y0=Y0,
                    wavelength=WAVELENGTH,
                    pixel_size=SLM_PIXEL_SIZE,
                    shape=SLM_SHAPE) 

    #loop through Zernike Polynomials:

    for radial,azimuthal in zip(radial_coords, azimuthal_coords):
        
        camera = Camera(roi =[680,509,784,611]) #xmin,ymin,xmax,ymax
        save_image = ImageHandler()

        #take image 0 as background image for comparison
        camera.update_exposure(exposure)
        background_image = camera.take_image()  
        save_image.save(background_image)

        #loop to take three sets of images for averaging
        for repeat in range(0,1,1):

            #loop through amplitude range
            for amplitude in amplitude_range:
                zernike_polynom = zernike(
                        radial=radial,
                        azimuthal=azimuthal,
                        amplitude=amplitude,
                        x0=X0,
                        y0=Y0,
                        radius=None,
                        shape=SLM_SHAPE)

                #sum holograms together
                x = lens + hor_grating_1 + hor_grating_2 + trap_hologram + ver_grating + zernike_polynom 

                #apply circular aperture
                holo = circ(
                        x,
                        x0=X0,
                        y0=Y0,
                        radius = APERTURE_RADIUS)
                
                #apply hologram to SLM
                slm.apply_hologram(holo)

                #pause to allow for grating to loads
                time.sleep(0.5)

                #use camera class to take and save photos
                camera.update_exposure(exposure)
                image = camera.take_image()
                save_image.save(image)
                
            repeat = repeat + 1

        #Delete causes camera disconnect
        del camera
        del save_image

        # Sleep for 1 second to allow for side-effects
        # such as the camera releasing its handles
        # to the OS
        time.sleep(1)



            
if __name__ == "__main__":
    main()