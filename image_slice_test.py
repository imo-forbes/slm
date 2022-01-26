import numpy as np
import cv2
import os

X_start = [0,12,20]
X_end = [10,14,25]
Y = [0,14]

path = r"Z:\Tweezer\People\Imogen\GitHub\slm\images\2022\January\23\Measure 7"

#create directories 
for file in range(0,3,1):
    os.mkdir(path + "/cropped_images_beam" + str(file+1))

for image_no in range(1,121,1):
    image = cv2.imread(r"Z:\Tweezer\People\Imogen\GitHub\slm\images\2022\January\23\Measure 7\{}.png".format(image_no))

    for i in range(0,3,1):
        y=0
        x=X_start[i]
        h=14
        w=X_end[i]
        crop = image[y:y+h, x:x+w]
        cv2.imwrite(path + "/cropped_images_beam" + str(i+1) + "/" +str(image_no) + ".png", crop )

    

