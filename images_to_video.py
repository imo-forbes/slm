import imageio
import numpy as np
from datetime import datetime

amplitude_range= np.arange(-50, 50, 5)

images = []

for x in range(0,121,1):#(3*len(amplitude_range)+1),1):
    d = r"\\srsblue01.mds.ad.dur.ac.uk\CornishLabs\vfuser01-labs\Tweezer\People\Imogen\GitHub\slm\images\2022\January\23\Measure 23\cropped_images_beam2" + "/{}.png". format(x)
    images.append(imageio.imread(d))

#filename currently needs to be manually set
imageio.mimsave(r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\Animated Runs\15.11.21\Measure16.gif" , images)
