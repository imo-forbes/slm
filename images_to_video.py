import imageio
import numpy as np
from datetime import datetime

amplitude_range= np.arange(-50, 50, 5)

images = []

for x in range(0,19,1):#(3*len(amplitude_range)+1),1):
    d = r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\15\Measure 16" + "/{}.png". format(x)
    images.append(imageio.imread(d))

#filename currently needs to be manually set
imageio.mimsave(r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\Animated Runs\15.11.21\Measure16.gif" , images)
