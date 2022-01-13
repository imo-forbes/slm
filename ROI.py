import cv2
import numpy as np
import os

amplitude_range= np.arange(-1,1,0.05)

for x in range(0,3*(len(amplitude_range))+1):
    img_raw = cv2.imread(r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\December\06\Measure 2" + "/{}.png".format(x))
    roi = (744, 397, 37, 37)
    roi_cropped = img_raw[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    path = r'C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\December\06\Measure 4\ROI'
    image_name = str(x) + '.png'
    (cv2.imwrite(os.path.join(path,image_name), roi_cropped))