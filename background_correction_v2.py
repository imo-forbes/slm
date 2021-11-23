# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:48:51 2021

@author: imoge
"""

import numpy as np
import matplotlib.pyplot as plt
from beam_fitting_code import profile

profile=profile(r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\08\Measure 3")
end = 23
d = r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\08\Measure 3\0.png"
z, wx, wy = profile.analyseBeamProfile(d)

w_x = wx - wx[0]
w_y = wx - wy[0]

plt.plot(z, w_x)

