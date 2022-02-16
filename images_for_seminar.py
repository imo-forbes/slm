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


# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from holograms.arrays import aags
# from camera import ImageHandler,Camera

# import time
# from holograms.zernike import zernike
# from holograms.gratings import hori_gradient, hori, vert_gradient
# from holograms.lenses import focal_plane_shift
# from holograms.arrays import aags

# from slm import SLM



# CONFIGURATION
X = []
Y = []



X0 = 233
Y0 = 251
SLM_SHAPE = (512, 512)
SLM_PIXEL_SIZE = 15e-6
WAVELENGTH = 1064e-9
SLM_SHAPE = (512, 512)
SLM_PIXEL_SIZE = 15e-6
WAVELENGTH = 1064e-9
APERTURE_RADIUS = 233
HORI_GRADIENT = -8.76
HORI_PERIOD = 7
LENS_SHIFT = 4
VERT_GRADIENT= 6.98


exposure = 0.5
repeat = 0 



# END CONFIGURATION


def main():
    #initialise classes

    #camera = Camera(roi =[595,305,635,320]) #ROI NEEDS UPDATED AS TRAP SPACING INCREASED

    camera = Camera()
    save_image = ImageHandler()

    # read the image file
    img = cv2.imread(r"Z:\Tweezer\People\Imogen\GitHub\slm\images_for_seminar\QLM_group B&W.jpg", cv2.IMREAD_COLOR)


    img = cv2.resize(img, (50,50))
    
    ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # converting to its binary form
    # bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # cv2.imshow("Binary", bw_img)
    # cv2.waitKey(0)
    # bw_img is an binary image made of pixels with either
    # 0 (black) or 255 (White)

    # get non transparent pixels (black)
    non_transparent = np.argwhere(bw_img == 0)

    trap_locations = non_transparent
    print(trap_locations)
    print(len(trap_locations))

    
  
    # trap_hologram = aags(traps=trap_locations,
    #                     iterations=2*len(trap_locations), #must be greater than no. of traps
    #                     beam_waist=None,
    #                     beam_center=(256,256),
    #                     shape=(512,512))

    # #grating and lens for sum
    # hor_grating_1 = hori_gradient(gradient=HORI_GRADIENT)
    # hor_grating_2 = hori(period = HORI_PERIOD)
    # ver_grating = vert_gradient(VERT_GRADIENT)
    # lens = focal_plane_shift(
    #                 shift=LENS_SHIFT,
    #                 x0=X0,
    #                 y0=Y0,
    #                 wavelength=WAVELENGTH,
    #                 pixel_size=SLM_PIXEL_SIZE,
    #                 shape=SLM_SHAPE) 


    # #sum holograms together
    # x = lens + hor_grating_1 + hor_grating_2 + trap_hologram + ver_grating

    #                 #apply circular aperture
    # holo = circ(x,
    #             x0=X0,
    #             y0=Y0,
    #             radius = APERTURE_RADIUS)


    
    # slm = SLM()


    # #apply hologram to SLM
    # slm.apply_hologram(holo)

    # #pause to allow for grating to loads
    # time.sleep(0.5)

    # #use camera class to take and save photos
    # camera.update_exposure(exposure)
    # image = camera.take_image()
    # save_image.save(image)
                
            

    # #Delete causes camera disconnect
    # del camera
    # del save_image

    # # Sleep for 1 second to allow for side-effects
    # # such as the camera releasing its handles
    # # to the OS
    # time.sleep(1)



            
if __name__ == "__main__":
    main()


