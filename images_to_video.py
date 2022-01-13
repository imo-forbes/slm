import imageio
import numpy as np
from datetime import datetime

amplitude_range= np.arange(-1, 1, 0.1)

images = []

for x in range(0,(3*len(amplitude_range)+1),1):
    d = r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\29\Measure 6" + "/{}.png". format(x)
    images.append(imageio.imread(d))

#filename currently needs to be manually set
imageio.mimsave(r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\Animated Runs\29.11.21Measure6.gif" , images)
