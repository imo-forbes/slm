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
import cv2

# CONFIGURATION
X = [236, 256, 276, 236, 256, 276, 236, 256,  236]
Y = [256, 256, 256, 236, 236, 236, 276, 276,  296]


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

exposure = 100
repeat = 0 

#set polynomials to loop through. r is radial coord, a is azimuthal 
radial_coords = [4,4,4,4,4,3,3,3,3,2,2,2,1,1,0]
azimuthal_coords = [4,2,0,-2,-4,3,1,-1,-3,2,0,-2,1,-1,0]


# END CONFIGURATION


def main():
    #initialise classes

    #camera = Camera(roi =[595,305,635,320]) #ROI NEEDS UPDATED AS TRAP SPACING INCREASED

    # camera = Camera()
    # save_image = ImageHandler()

    slm = SLM()

    assert len(radial_coords) == len(azimuthal_coords), "Radial and azimuthal coords should have same length"

    #run algorithm to set trap locations 
    trap_locations = [(271, 256), (270, 257), (270, 258), (269, 259), (268, 260), (257, 261), (259, 261), (261, 261), (263, 261), (265, 261), (267, 261), (269, 261), (285, 261), (287, 261), (289, 261), (291, 261), (293, 261), (256, 262), (270, 262), (308, 262), (321, 262), (285, 263), (256, 264), (266, 264), (270, 264), (308, 264), (321, 264), (285, 265), (256, 266), (270, 266), (308, 266), (321, 266), (285, 267), (256, 268), (270, 268), (308, 268), (321, 268), (285, 269), (270, 270), (308, 270), (321, 270), (256, 271), (285, 271), (315, 271), (270, 272), (308, 272), (321, 272), (257, 273), (259, 273), (261, 273), (263, 273), (265, 273), (267, 273), (269, 273), (285, 273), (315, 273), (308, 274), (310, 274), (312, 274), (314, 274), (316, 274), (319, 274), (321, 274)]
    trap_hologram = aags(traps = trap_locations,#((X[0],Y[0]), (X[1],Y[1]),(X[2],Y[2]), (X[3],Y[3]), (X[4],Y[4]),(X[5],Y[5]), (X[6],Y[6]), (X[7],Y[7]), (X[8],Y[8])),#, (X[9],Y[9]), (X[10],Y[10])),#,(X[11],Y[11]), (X[12],Y[12]), (X[13],Y[13]),(X[14],Y[14])),
                        iterations=120, #must be greater than no. of traps
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
        
        camera = Camera()#(roi =[680,509,784,611]) #xmin,ymin,xmax,ymax
        save_image = ImageHandler()

        #take image 0 as background image for comparison
        camera.update_exposure(exposure)
        background_image = camera.take_image()  
        save_image.save(background_image)

        #loop to take three sets of images for averaging
        for repeat in range(0,3,1):

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