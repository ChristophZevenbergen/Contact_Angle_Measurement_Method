#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 16:15:06 2025

@author: Christoph Indiana Zevenbergen
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import Lib_ContactAngle as esf



# Input file must be in the following format:
# 0 - SOLID
# 1 - FLUID 1
# 2 - FLUID 2

# Main function is: MeasureAngle
# The code will measure the contact angle distribuition on the image using the sphere regression method and return the following:
# A 2D NumPy array where each row corresponds to one detected contact point. Columns include:
    # - [0]  Angle (degrees)
    # - [1]  X coordinate of the contact point
    # - [2]  Y coordinate of the contact point
    # - [3]  Z coordinate of the contact point
    # - [4]  Local pore diameter
    # - [5]  Plane coefficient a (from ax + by + cz + d = 0)
    # - [6]  Plane coefficient b
    # - [7]  Plane coefficient c
    # - [8]  Plane coefficient d
    # - [9]  Sphere center X (C1)
    # - [10] Sphere center Y (C2)
    # - [11] Sphere center Z (C3)
    # - [12] Sphere radius R
    # - [13] Plane regression error (mean distance to plane)
    # - [14] Sphere regression error (mean distance to sphere)
    # - [15] Number of points for plane regression
    # - [16] Number of points for sphere regression


# Edit to folder of where the library file is located. 
file_dir = '/home/usuario/Documentos/GitHub/Sphere_regression'
#Input file .raw unisigned char
input_file = f'{file_dir}/Bentheimer_0125_A60.raw'

#Size of the image
size_x = 125
size_y = 125
size_z = 125

# Subvolume Radius for Measurements
# Default Values
R1 = 5 # Plane regression subvolume radius
alpha = 0.5 # Sphere regression subvbolume radius scaling factor


#Read image as numpy three dimanesional array
npimg = np.fromfile(input_file, dtype=np.uint8)
imageSize = (size_x, size_y, size_z)
npimg = npimg.reshape(imageSize)

# Plot a slice of the porous media
plt.imshow(npimg[:,:,50])


# Main Function Measure the contact Angle
Measurements = esf.MeasureAngle(npimg)


Reference_measurements = np.load(f'{file_dir}/measurements.npy')

if np.allclose(Measurements, Reference_measurements, rtol=0, atol=1e-14):
    print('Measurements taken successfully.')
else:
    raise ValueError('Measurements do not match the reference.')
