#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contact Angle Measurement Library for Porous Media (2025v3)
------------------------------------------------------------

This library provides a set of tools for estimating and analyzing contact angles in three-phase segmented images of porous media. It is specifically designed for 3D datasets where the phases include solid matrix (rock), and two immiscible fluids. The core functionality is based on detecting three-phase contact points and performing geometric regressions (plane and sphere fitting) to calculate the local contact angle formed at the intersection.

Main Features
-------------
- Fast interface tracking using voxel-wise segmentation
- Automatic detection of three-phase contact points
- Regression of a sphere to fluid-fluid interfaces and a plane to rock-fluid interfaces
- Contact angle estimation based on the intersection geometry
- Filtering based on regression error to ensure robust measurements
- Visualization of regression results with 3D rendering (via PyVista)
- Gaussian Mixture Model (GMM) fitting to statistically analyze contact angle distributions

Target Application
------------------
This library is intended for digital rock physics and porous media research, especially in applications involving wettability characterization, fluid-fluid displacement, and pore-scale analysis using micro-CT images or synthetic datasets.

Dependencies
------------
- numpy, scipy, matplotlib, porespy, pyvista, scikit-learn

Author
------
Christoph Zevenbergen, 2025
"""


import numpy as np
from matplotlib import pyplot as plt
from scipy import odr
from scipy.ndimage import label
import matplotlib.pyplot as plt
from collections import Counter
import porespy as ps
import time
import pyvista as pv
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

# RESOLVIDO: pontos de contato apenas nos voxels fluid, erro na função rastrear
# RESOLVIDO: search radius cubo ao invés do esférico
# RESOLVIDO: plane regression in function of z
# RESOLVIDO: plane regression first iteration guess all zero, the derivative of the level set function could be used
# RESOLVIDO: not shure if the label of the rock-fluid interface works well, maybe change it to both rock and fluid voxels, or do not label the rock_fluid interface (DOES NOT IMPROVE RESULTS)
# RESOLVIDO: sphere Sign verification only verifies for the fluid 2
# RESOLVIDO: sphere Sign verification may use fluid 2 voxels from other pores, changing the results
# RESOLVIDO: Why is the subvolume an even number?
# OBS: Contact points não são usados nas regressões
# OBS: testar função para resíduos da regressão

# Version 2025v1 
# Changes:
# Do not label the rock-fluid interface (may use interface points from other pores)
# Orthogonal plane regression (stoped using the scipy odr function)
# Add a filter based on the regression error

# Version 2025v2 
# Changes:
# No changes to the method. Measurements should be the same as version 2025v1
# Change the way the data is saved, instead of saving the information of the contact angles in the four-dimensional array, save it in a bidimensional array, with angle measured, position, plane equation, sphere equation, sphere regression error and plane regression error.
# Incorporate anguloesfera function into RegressãoEsfera function (anguloesfera function deleted)
# Eliminate vetorunitario function, not used in the code
# Eliminate dphi function, instead of computing the derivative of the level set function for the entire image, the derivative is only computed in the contact point (inside the RegressaoEsfera function)
# New function: plotregression. This function plots the sphere and plane regression as well as the points used for the regression

# Version 2025v3 
# Changes:
# Change in initial guess for sphere regression, removed a division by 2 from the radius initial guess
# Add function Gaussian_mixture. This function fits guassian distributions to the contact angle measurement and segments the measurements (The number of expected gaussian must be informed)
# Change name of functions to english, (Reg_esfera->MeasureAngle; Rastrear_fast -> ImgTracking; RegressãoEsfera -> SphereRegression; Encontrar_pontos_label_fluid -> FindPointsFluid; Encontrar_pontos_label_Rock -> FindPointsRock; Pontos_dentro_esfera -> SphereSign; plotrockwithregression -> ViewMeasurement, Gaussian_mixture -> ResultAnalysis) 
# Add description to functions: MeasureAngle, ViewMeasurement, ViewRegression, ResultAnalysis
# Add description to the library
# Removed function juntarmetrizesvetor
# Add the number of points used for regression, chnages in function 'MeasureAngle' and 

# Version 2025v4 
# Changes:
# Correct the residual of the sphere regression. The old version was calculating as the Square root of the residual error in the sphere model
#   Sphere model: (a-x)^2 + (b-y)^2 + (c-z)^2 - R^2 = 0 
#   Residual: sqrt( (a-x)^2 + (b-y)^2 + (c-z)^2 - R^2 )
# This way of calculating the regression residual does not represent the distance of the point to the sphere. The correct way to calculate the residual as the distance of the point to the sphere is:
#   sqrt( (a-x)^2 + (b-y)^2 + (c-z)^2 ) - R
# 
# Change the Rock array, the older version did not add the contact points to the rock interface (the rock voxels in contact with both fluids were labeled as contact point and were not included in the Rock-fluid interface)
# Changes in the ImgTracking and Interfaces functions
 

# GitHub Version 1 
# Fix error: Angle given in degrees (not radians)
# Filter all Nan measurements

############## Main Funcion ###################

def MeasureAngle(npimg, alpha = 0.5, R1 = 5, mpe = 3, mpp = 3, maxerr_e = 20, maxerr_p = 20):
    """
    MeasureAngle(npimg, alpha=0.5, R1=3, mpe=20, mpp=15, maxerr_e=3, maxerr_p=0.3)

    Estimates the contact angle distribution in a 3D segmented image.

    Parameters
    ----------
    npimg : ndarray
        A 3D numpy array representing a segmented image where:
            
        - 0 corresponds to rock
        - 1 to fluid 1
        - 2 to fluid 2

    alpha : float, optional
        Proportion of the pore size used to define the fluid-fluid search radius. Default is 0.8.

    R1 : int, optional
        Radius used for detecting rock-fluid interfaces. Default is 3 voxels.

    mpe : int, optional
        Minimum number of points required for sphere (fluid-fluid interface) regression. Default is 20.

    mpp : int, optional
        Minimum number of points required for plane (rock-fluid interface) regression. Default is 15.

    maxerr_e : float, optional
        Maximum mean regression error (in voxels) accepted for the sphere fit. Default is 3.

    maxerr_p : float, optional
        Maximum mean regression error (in voxels) accepted for the plane fit. Default is 0.3.

    Returns
    -------
    Contact_Angle : ndarray
        A 2D NumPy array where each row corresponds to one detected contact point. Columns include:

        - [0]  Angle (degrees)
        - [1]  X coordinate of the contact point
        - [2]  Y coordinate of the contact point
        - [3]  Z coordinate of the contact point
        - [4]  Local pore diameter
        - [5]  Plane coefficient a (from ax + by + cz + d = 0)
        - [6]  Plane coefficient b
        - [7]  Plane coefficient c
        - [8]  Plane coefficient d
        - [9]  Sphere center X (C1)
        - [10] Sphere center Y (C2)
        - [11] Sphere center Z (C3)
        - [12] Sphere radius R
        - [13] Plane regression error (mean distance to plane)
        - [14] Sphere regression error (mean distance to sphere)
        - [15] Number of points for plane regression
        - [16] Number of points for sphere regression

    Notes
    -----
    This function performs the following steps:
        
    1. Detects rock-fluid and fluid-fluid interfaces from the segmented image.
    2. Computes local pore sizes using a distance transform.
    3. For each contact point, fits a plane to the rock-fluid interface and a sphere to the fluid-fluid interface.
    4. Calculates the contact angle from the geometry of the fitted surfaces.

    Dependencies
    ------------
    This function relies on `ps.filters.local_thickness` from the `PoreSpy` library and `ndimage.label` from the `scipy` library.

    """
    
    start_time = time.time()
    print('Tracking interfaces')
    # Tracks the image for contact points, rock-fluid interface points and fluid-fliud interfafce points.
    P, Rock = ImgTracking(npimg)    
    # Label the conected interfaces
    Fluid = Interfaces(P)
    # Uses the Distance transform function to estimete pore sizes in the image
    print('Transform Ditance')
    Pore = np.where(npimg == 2, 1, npimg)
    Transf_Dist = ps.filters.local_thickness(Pore)   

    print('Plane and Sphere regression')
    # Fits a plane to the rock-fluid interface and a sphere to the fluid-fluid interface for each contact point
    Contact_Angle = SphereRegression(P, Rock, Fluid, Transf_Dist, npimg, R1, alpha, mpe, mpp, maxerr_e, maxerr_p)    

    print('Computing Contact Angle')    
    #Calculates the contact angle using the palne and sphere equation
    
    # Filter all Nan values.
    Contact_Angle = Contact_Angle[:, ~np.isnan(Contact_Angle[0])]

    end_time_part1 = time.time()
    print(f"Time taken for contact angle measurements: {end_time_part1 - start_time} seconds")

    return Contact_Angle

########### Angle Measurement #################

def ImgTracking(D):    
    #Tracks the interface by subtracting the offset of the image in each direction
    #The input three dimensional array must be given in the following format:
        # 0 - SOLID
        # 1 - FLUID 1
        # 2 - FLUID 2
    
    X, Y, Z = np.shape(D)

    #Creates new three dimensional array 'P' where the interface tracking will be saved
        # 0 - NO INTERFACE
        # 1 - SOLID-FLUID INTERFACE (Solid voxels in contact with fluid voxels)
        # 2 - FLUID-FLUID INTERFACE (Fluid 1 voxels in contact with fluid 2 voxels and vice versa)
        # 3 - CONTACT POINT (Fluid 1 or 2 voxel in contact with all three phases)
    P=np.zeros((X,Y,Z),dtype="uint8")

    #Creates new three dimensional array 'Rock' where: 
        # 0 - FLUIDS
        # 1 - ROCK
    #The subtraction of the offset of this image will track the SOLID-FLUID INTERFACE
    Rock = D.copy()
    Rock[D == 0] = 1
    Rock[(D == 1) | (D == 2)] = 0
    R_x1 = Rock - np.roll(Rock,  1, axis = 0)
    R_x2 = Rock - np.roll(Rock, -1, axis = 0)
    R_y1 = Rock - np.roll(Rock,  1, axis = 1) 
    R_y2 = Rock - np.roll(Rock, -1, axis = 1)
    R_z1 = Rock - np.roll(Rock,  1, axis = 2) 
    R_z2 = Rock - np.roll(Rock, -1, axis = 2) 
    R_x1[R_x1 == 255] = 0
    R_x2[R_x2 == 255] = 0
    R_y1[R_y1 == 255] = 0
    R_y2[R_y2 == 255] = 0
    R_z1[R_z1 == 255] = 0
    R_z2[R_z2 == 255] = 0
    # Resulting iamges R_x1, R_x2, Ry1... will containg 1 for all SOLID-FLUID INTERFACE and zero for the remaining of the image
    # only the rock voxels in contact with the fluid phase are used for the interface
    P = P + R_x1
    P = P + R_x2
    P = P + R_y1
    P = P + R_y2
    P = P + R_z1
    P = P + R_z2
    P[P!=0] = 1
    #The tracking is passed to the 'P' array   
    # Save the rock-fluid interface to the Rock array
    Rock = P.copy()
    
    
    #Creates new three dimensional array 'Fluids' where: 
        # 0 - ROCK
        # 1 - FLUID 1
        # 3 - FLUID 2
    #The subtraction of the offset of this image will track the FLUID-FLUID INTERFACE
    Fluids = D.copy()
    Fluids[Fluids == 2] = 3 
    #The subtraction of the offset of this image will track the FLUID-FLUID INTERFACE and CONTACT POINTS
    Cont_points = np.zeros((X,Y,Z),dtype="uint8")
    F_x1 = Fluids - np.roll(Fluids,  1, axis = 0)
    F_x2 = Fluids - np.roll(Fluids, -1, axis = 0) 
    F_y1 = Fluids - np.roll(Fluids,  1, axis = 1) 
    F_y2 = Fluids - np.roll(Fluids, -1, axis = 1) 
    F_z1 = Fluids - np.roll(Fluids,  1, axis = 2) 
    F_z2 = Fluids - np.roll(Fluids, -1, axis = 2)     
    #The resulting arrays are asingend as
        #2 - FLUID-FLUID INTERFACE
        #1 and 3 - Fluid voxels in contact with rock voxels 
    F_x1[F_x1 == 254] = 2
    F_x2[F_x2 == 254] = 2
    F_y1[F_y1 == 254] = 2
    F_y2[F_y2 == 254] = 2
    F_z1[F_z1 == 254] = 2
    F_z2[F_z2 == 254] = 2
    #saves information of the fluid-rock interface for tracking contact points
    Cont_points[F_x1 == 1] = 3
    Cont_points[F_x2 == 1] = 3
    Cont_points[F_y1 == 1] = 3
    Cont_points[F_y2 == 1] = 3
    Cont_points[F_z1 == 1] = 3
    Cont_points[F_z2 == 1] = 3
    Cont_points[F_x1 == 3] = 3
    Cont_points[F_x2 == 3] = 3
    Cont_points[F_y1 == 3] = 3
    Cont_points[F_y2 == 3] = 3
    Cont_points[F_z1 == 3] = 3
    Cont_points[F_z2 == 3] = 3
    F_x1[F_x1 != 2] = 0
    F_x2[F_x2 != 2] = 0
    F_y1[F_y1 != 2] = 0
    F_y2[F_y2 != 2] = 0
    F_z1[F_z1 != 2] = 0
    F_z2[F_z2 != 2] = 0
    P = P + F_x1    
    P = P + F_x2    
    P = P + F_y1    
    P = P + F_y2    
    P = P + F_z1    
    P = P + F_z2
    P[np.logical_and(P != 0, P != 1)] = 2
    #The tracking of the FLUID-FLUID INTERFACE is passed to the 'P' array   
    
    #Contact points are identified by the intersection of the fluid-rock interface and fluid-fluid interface
    P = P + Cont_points
    P[P == 3] = 0
    P[P == 5] = 3
 
    #Add contact points - Rock voxels conected with both fluids
    
    #Creates new three dimensional array 'Rock' where: 
        # 0 - FLUID1 AND ROCK
        # 1 - FLUID2
    #The subtraction of the offset of this image will track the SOLID-FLUID INTERFACE
    Fl2 = D.copy()
    Fl2[(D == 0) | (D == 1)] = 1
    Fl2[D == 2] = 0
    Fl2_x1 = Fl2 - np.roll(Fl2,  1, axis = 0)
    Fl2_x2 = Fl2 - np.roll(Fl2, -1, axis = 0)
    Fl2_y1 = Fl2 - np.roll(Fl2,  1, axis = 1) 
    Fl2_y2 = Fl2 - np.roll(Fl2, -1, axis = 1)
    Fl2_z1 = Fl2 - np.roll(Fl2,  1, axis = 2) 
    Fl2_z2 = Fl2 - np.roll(Fl2, -1, axis = 2) 
    Fl2_x1[Fl2_x1 == 255] = 0
    Fl2_x2[Fl2_x2 == 255] = 0
    Fl2_y1[Fl2_y1 == 255] = 0
    Fl2_y2[Fl2_y2 == 255] = 0
    Fl2_z1[Fl2_z1 == 255] = 0
    Fl2_z2[Fl2_z2 == 255] = 0   
    Fl2 = (Fl2_x1 + Fl2_x2 + Fl2_y1 + Fl2_y2 + Fl2_z1 + Fl2_z2)
    Fl2[Fl2 != 0] = 1

    #Creates new three dimensional array 'Rock' where: 
        # 0 - FLUID2 AND ROCK
        # 1 - FLUID1
    #The subtraction of the offset of this image will track the SOLID-FLUID INTERFACE
    Fl1 = D.copy()
    Fl1[(D == 0) | (D == 2)] = 1
    Fl1[D == 1] = 0
    Fl1_x1 = Fl1 - np.roll(Fl1,  1, axis=0)
    Fl1_x2 = Fl1 - np.roll(Fl1, -1, axis=0)
    Fl1_y1 = Fl1 - np.roll(Fl1,  1, axis=1) 
    Fl1_y2 = Fl1 - np.roll(Fl1, -1, axis=1)
    Fl1_z1 = Fl1 - np.roll(Fl1,  1, axis=2) 
    Fl1_z2 = Fl1 - np.roll(Fl1, -1, axis=2) 
    Fl1_x1[Fl1_x1 == 255] = 0
    Fl1_x2[Fl1_x2 == 255] = 0
    Fl1_y1[Fl1_y1 == 255] = 0
    Fl1_y2[Fl1_y2 == 255] = 0
    Fl1_z1[Fl1_z1 == 255] = 0
    Fl1_z2[Fl1_z2 == 255] = 0   
    Fl1 = (Fl1_x1 + Fl1_x2 + Fl1_y1 + Fl1_y2 + Fl1_z1 + Fl1_z2)
    Fl1[Fl1 != 0] = 1

    # Rock voxels in contact with both fluids
    F = Fl1 + Fl2
    F[F == 1] = 0
    
    P = P + F
    #The first and last value of array in each direction is discarted
    P[[0, -1], :, :] = 0
    P[:, [0, -1], :] = 0
    P[:, :, [0, -1]] = 0
    return P, Rock

def Interfaces(original_array):
    # Creates two new three dimensional arrays, one for the Rock interace and another for the Fluid interface
    # This arrays will contain zero for all voxels not belonging to the interface
    # In order to decrease computational cost all interface voxels conected are labeled as 1, 2, 3, 4, ...n, where n is the number of interfaces not conected to each other.
    # For example, in the resulting image all voxels with value x must be conected
    
    # Creates new array, with value 1 for rock-fluid interface and zero for the remaining of the image
    #REMOVED# Rock = np.where((original_array == 2) | (original_array == 3), 0, original_array)
     
    #REMOVED# Label the rock interface
    #REMOVED# Use scipy.ndimage.label to find connected components
    #REMOVED# Rock, num_features = label(Rock) #Do not label Rock-Flui interface, the entire rock-fluid is considered one interface coneceted
    #Do not label Rock-Flui interface, the entire rock-fluid is considered one interface coneceted
    
    # Creates new array, with value 2 for Fluid-fluid interface and zero for the remaining of the image
    Fluid = np.where((original_array == 1) | (original_array == 3), 0, original_array)
    # Label the rock interface
    # Use scipy.ndimage.label to find connected components
    Fluid, num_features = label(Fluid)

    return Fluid

def SphereRegression(P, Rock, Fluid, Transf_Dist, npimg, R1, alpha, mpe, mpp, maxerr_e, maxerr_p):
    # Fits a plane to the rock-fluid interface and a sphere to the fluid-fluid interface
    
    # Creates array containing the coordinates of every contact point
    Contact_Angle = np.where(P == 3)
    Contact_Angle = np.array(Contact_Angle)
    
    #Create Contact_Angle array, in this array all information needed for the measurement will be saved
    #Angle, X, Y, Z, Pore_Diameter, a, b, c, d, C1, C2, C3, R, err_p, err_e
    Contact_Angle = np.pad(Contact_Angle, ((1, 13), (0, 0)), mode='constant', constant_values=0).astype(float)
    a, b = np.shape(Contact_Angle)
    
    #Set all contact angles measured to nan (not a number)
    Contact_Angle[0,:] = np.nan

    # Creates Level set function, values are asigned as:
        # -1 - Fluid
        #  1 - Rock
    phi = np.where((npimg == 2) | (npimg == 1), -1, npimg)
    phi = np.where(phi == 0, 1, phi)
    
    # Loop through every contact point
    for pos in range(0, b):
        # Obtain coordinates of the contact point
        i, j, k = (int(x) for x in Contact_Angle[1:4, pos])
        
        # Obtain coordinates of the fluid-fluid interface inside the search radius
        # The search radius for the fluid-fluid interface is proportional to the pore size
        v1, R = FindPointsFluid(Fluid, (i,j,k), Transf_Dist, alpha)
        Contact_Angle[4,pos] = R/alpha
        # Obtain coordinates of the Rock-fluid interface inside the search radius
        v2 = FindPointsRock(Rock, (i,j,k), R=R1)

        # Filter cases where not enouth points are found        
        Contact_Angle[15,pos] = np.size(v2)/3
        Contact_Angle[16,pos] = np.size(v1)/3
        if np.size(v1) > mpe*3 and np.size(v2) > mpp*3:    

            # Sphere regression
            # First iteration for the sphere regression using initial guess: center is the mean of all points and radius is the distance half the distence between the two most distant points
            min_coords = np.min(v1, axis=0)
            max_coords = np.max(v1, axis=0)
            center_guess = (min_coords + max_coords) / 2.0
            radius_guess = np.linalg.norm(max_coords - center_guess) #REMOVED# / 2.0
            v1t = v1.transpose()                                                    
            data = odr.Data(v1t, y=1)
            # Sphere regression model uses the equation 
            # (x - a)² + (y - b)² + (z - c)² - R² = 0
            model = odr.Model(sphere_model, implicit=True)
            # Orthogonal Distance Regression 
            odr_solver = odr.ODR(data, model, beta0=[np.mean(v1t[0, :]), np.mean(v1t[1, :]), np.mean(v1t[2, :]), radius_guess])  
            result = odr_solver.run()
            # Save the results in the coeficients array
            #REMOVED# coefesfera[i,j,k,:4] = result.beta[:4]
            Contact_Angle[9:13, pos] = result.beta[:4]
            # Calculate the residuals
            residuals = sphere_model_residual(result.beta, v1t)
            # Save the error for the regression
            Contact_Angle[14, pos] = np.mean(np.abs(residuals))
            # If the error is higher than the maximum allowed, discard the sphere regression
            if Contact_Angle[14, pos] > maxerr_e:
                Contact_Angle[9:13, pos] = 0

            # Plane regression
            # Plane regression model uses the equation 
            # a*x + b*y + c*z + d = 0
            # Compute the mean point
            points = v2#.transpose()
            P_mean = points.mean(axis=0)
            # Center the points to it's mean
            centered = points - P_mean            
            # Compute the covariance matrix of the centered points
            C = np.dot(centered.T, centered) / len(points)            
            # Eigen decomposition of the matrix
            eigenvalues, eigenvectors = np.linalg.eigh(C)  # 'eigh' is for symmetric matrices            
            # Take the eigenvector with the smallest eigenvalue
            normal_vector = eigenvectors[:, np.argmin(eigenvalues)]            
            # Compute the plane offset d
            d = -np.dot(normal_vector, P_mean)
            a,b,c = normal_vector
            
            # Save the results in the coeficients array
            Contact_Angle[5:8, pos] = normal_vector
            Contact_Angle[8, pos] = d
            
            # Calculate the residuals
            residuals = plane_func(Contact_Angle[5:9, pos], points.T)
            # Save the error for the regression
            Contact_Angle[13, pos] = np.mean(np.abs(residuals))
            # If the error is higher than the maximum allowed, discard the plane regression
            if Contact_Angle[13, pos] > maxerr_p:
                Contact_Angle[5:9, pos] = 0
                
            
            # Verify the sense of the normal vector of the plane by comparing with the normal vector of using the level set function
            # Calculate the derivative of the level set function
            dphidx = (int(phi[i+1, j,k])-int(phi[i-1, j, k]))/2
            dphidy = (int(phi[i, j+1,k])-int(phi[i, j-1, k]))/2
            dphidz = (int(phi[i, j,k+1])-int(phi[i, j, k-1]))/2
            # Calculate the scalar product of the the vector obtain by the plane regression and the level set function                        
            b = Contact_Angle[5,pos]*dphidx + Contact_Angle[6,pos]*dphidy + Contact_Angle[7,pos]*dphidz
            # If the scalar product is negative, that means that the sense of the vector obtain by the plane regression must be inverted.
            if b < 0:
                Contact_Angle[5:8,pos] = -1*Contact_Angle[5:8,pos]
            # Verify for wich of the fluid the curvature is positive
            Sign = SphereSign(Contact_Angle[9:13, pos], (i,j,k), npimg, R)
            # Multiplies the radius by the Sign of the curvature
            Contact_Angle[5:8, pos] = Sign*Contact_Angle[5:8, pos]
            

            # If enouth points for regression and sphere regression diferent than the trivial solution
            if Contact_Angle[12,pos] != 0 and (Contact_Angle[5,pos] != 0 or Contact_Angle[6,pos] != 0 or Contact_Angle[7,pos] != 0):
                # Calculate distance from center of the sphere to the plane
                d = Contact_Angle[5,pos]*Contact_Angle[9,pos] + Contact_Angle[6,pos]*Contact_Angle[10,pos] + Contact_Angle[7,pos]*Contact_Angle[11,pos]
                # Calculates the cossine of the contact angle as the ratio between the distance of the center of the sphere to the plane and it's radius.
                cos = d/Contact_Angle[12,pos]
                # Due to numerical errors the cossine may obtain values higher than 1 or lower than -1
                # This represents a case in which the sphere does not intersect with the plane and should be expected to occur given the standart deviation of the measurements (Especialy for very wettable or non wettable fluids).
                # For this reason the cossine is corrected to fit in the range from -1 to 1
                if -1.01 < cos <= -1:
                    cos = -1
                elif 1.01 > cos >= 1:
                    cos = 1
                if cos >= -1 and cos <= 1:
                    # Calculates the contact angle as arccos of the cos estimated
                    # Saves the value measure in the Contact_Angle array together with it's coordinates.
                    Contact_Angle[0, pos] = (np.pi - np.arccos(cos))*180/np.pi   
            
    return Contact_Angle;

def sphere_model(B, x):
    return (x[0] - B[0])**2 + (x[1] - B[1])**2 + (x[2] - B[2])**2 - B[3]**2

def sphere_model_residual(B, x):
    return np.sqrt((x[0] - B[0])**2 + (x[1] - B[1])**2 + (x[2] - B[2])**2) - B[3]

def plane_func(beta, x):
    return beta[0]*x[0] + beta[1]*x[1]  + beta[2]*x[2] + beta[3]

def plane_func_Z(beta, x):
    return beta[0]*x[0] + beta[1]*x[1] + beta[2]

def ExtSubvolume(image, c_point, a):
    # a       - Size of the subvolume
    # c_point - center of the subvolume
    # image   - original image
    # For contact points near the edges of the image the subvolume is reduced in size to avoid exceeding the array boundaries and ensure the subvolume remains within the valid index range
    x_start = max(c_point[0] - a, 0)
    x_end = min(c_point[0] + a, image.shape[0]) + 1
    y_start = max(c_point[1] - a, 0)
    y_end = min(c_point[1] + a, image.shape[1]) + 1
    z_start = max(c_point[2] - a, 0)
    z_end = min(c_point[2] + a, image.shape[2]) + 1
    # Extract the subvolume and estimate the pore diameter as the maximum value within this region
    SubVolume = image[x_start:x_end, y_start:y_end, z_start:z_end]
    # Obtain the coordinate of the contact point for the subvolum    
    center_coords = np.array([(c_point[0]-x_start), (c_point[1]-y_start), (c_point[2]-z_start)])
    return SubVolume, center_coords

def FindPointsFluid(Interface, c_point, Transf_Dist, alpha):
    #Function extract all fluid-fluid interface points inside the search radius
    
    # Estimates the pore diameter near the contact point
    # Uses the maximal value of the Transform Function in the subvolume 7x7 around the contact point
    Diameter_subvolume, _ = ExtSubvolume(Transf_Dist, c_point, 3) 
    # Search Radius 'R' proportional to the pore diameter by the factor 'alpha'
    R = np.ceil(np.max(Diameter_subvolume)*alpha).astype(int)

    # Extract the subvolume of the search radius
    Sub, center_coords = ExtSubvolume(Interface, c_point, R)
    
    # Obain the label of the fluid-fluid interface conected to the contact point
    # Extract the subvolume (size 5x5) to obtain the label of the fluid-fluid interface
    c_Vol, _ = ExtSubvolume(Interface, c_point, 2)

    # Creates array of the labels of interfaces near the contact point, only one value expected
    Labels = np.unique(c_Vol[c_Vol != 0])
    pontos = np.zeros((0, 3))
    # Loop trhough labels to extract all interface points in the search radius subvolume
    for i in Labels:
        # Extract interface points and shift the coordinate system to the contact point (set as origin)
        p = np.array(np.where(Sub == i)).T
        p = p - center_coords
        pontos = np.vstack([pontos, p])
        
    # Discard all points that are not inside the search radius. (the subvolume is a cube)
    pontos = pontos[np.linalg.norm(pontos, axis = 1) <= R]
    return pontos, R

def FindPointsFluid_knowndiameter(Interface, c_point, Diameter_subvolume, alpha):
    #Function extract all fluid-fluid interface points inside the search radius

    # Search Radius 'R' proportional to the pore diameter by the factor 'alpha'
    R = np.ceil(np.max(Diameter_subvolume)*alpha).astype(int)

    # Extract the subvolume of the search radius
    Sub, center_coords = ExtSubvolume(Interface, c_point, R)
    
    # Obain the label of the fluid-fluid interface conected to the contact point
    # Extract the subvolume (size 5x5) to obtain the label of the fluid-fluid interface
    c_Vol, _ = ExtSubvolume(Interface, c_point, 2)

    # Creates array of the labels of interfaces near the contact point, only one value expected
    Labels = np.unique(c_Vol[c_Vol != 0])
    pontos = np.zeros((0, 3))
    # Loop trhough labels to extract all interface points in the search radius subvolume
    for i in Labels:
        # Extract interface points and shift the coordinate system to the contact point (set as origin)
        p = np.array(np.where(Sub == i)).T
        p = p - center_coords
        pontos = np.vstack([pontos, p])
        
    # Discard all points that are not inside the search radius. (the subvolume is a cube)
    pontos = pontos[np.linalg.norm(pontos, axis = 1) <= R]
    return pontos, R

def FindPointsRock(Interface, c_point, R):
    #Function extract all rock-fluid interface points inside the search radius

    # Extract the subvolume of the search radius
    # Obtain the coordinate of the contact point for the subvolume
    Sub, center_coords = ExtSubvolume(Interface, c_point, R)


    # Obain the label of the rock-fluid interface conected to the contact point
    # Extract the subvolume to obtain the label of the rock-fluid interface
    c_Vol, _ = ExtSubvolume(Interface, c_point, 2)
    
    # Creates array of the labels of interfaces near the contact point, only one value expected
    Labels = np.unique(c_Vol[c_Vol != 0])
    pontos = np.zeros((0, 3))
    #Loop trhough labels to extract all interface points in the search radius subvolume
    for i in Labels:
        # Extract interface points and shift the coordinate system to the contact point (set as origin)
        p = np.array(np.where(Sub == i)).T
        p = p - center_coords
        pontos = np.vstack([pontos, p])
        
    # Discard all points that are not inside the search radius. (the subvolume is a cube)
    pontos = pontos[np.linalg.norm(pontos, axis = 1) <= R]
    return pontos

def SphereSign(coefesfera, c_point, npimg, a):
    # Verify which fluid is inside the sphere, returns the Sign of the curvature
    
    # Extracts cubic subvolume with the size of 2*Radius+1
    # Obtain the coordinate of the contact point for the subvolume    
    Sub, center_coords = ExtSubvolume(npimg, c_point, a)
    
    Subf2 = np.where((Sub == 0) | (Sub == 1), 0, Sub)
    Subf2, num_features = label(Subf2)
    c_Vol, _ = ExtSubvolume(Subf2, center_coords, 1)
    Labels2 = np.unique(c_Vol[c_Vol != 0])

    Subf1 = np.where((Sub == 0) | (Sub == 2), 0, Sub)
    Subf1, num_features = label(Subf1)
    c_Vol, _ = ExtSubvolume(Subf1, center_coords, 1)
    Labels1 = np.unique(c_Vol[c_Vol != 0])
    
    # Sum the distance from each fluid 2 voxel inside the subvolume to surface of the sphere. If the Sign is negatice, the voxel is inside the sphere and if the Sign is positive, the voxel is outside the sphere.
    # This way we verify if the curvature should be positive or negative.
    # For contact points near the edges of the image the subvolume is reduced in size to avoid exceeding the array boundaries and ensure the subvolume remains within the valid index range
    
    # Obtain coordinates of every voxel of fluid 2 inside the subvolume
    pontos = np.array(np.where(Subf2 == Labels2)).T
    # Shift coordinate system to the center of the subvolume
    pontos = pontos - center_coords   
    # Calculate the distance of the points to the sphere surface
    dist = np.sqrt((pontos[:,0]-coefesfera[0])**2 + (pontos[:,1]-coefesfera[1])**2 + (pontos[:,2]-coefesfera[2])**2) - coefesfera[3]
    # Obtain the Sign of the curvature for the reference fluid
    Sign_fluid2 = np.sign(np.sum(dist))

    # Analog to fluid 1
    pontos = np.array(np.where(Subf1 == Labels1)).T
    pontos = pontos - center_coords   
    dist = np.sqrt((pontos[:,0]-coefesfera[0])**2 + (pontos[:,1]-coefesfera[1])**2 + (pontos[:,2]-coefesfera[2])**2) - coefesfera[3]
    Sign_fluid1 = np.sign(np.sum(dist))

    if Sign_fluid1*Sign_fluid2 == 1:
        return 0
    return Sign_fluid2

############ Result Analysis ##################

def ViewRegression(Contact_Angle, npimg, R1 = 3, alpha = 0.8):
    """
    Visualize the geometric regression of the contact angle using a fitted sphere and plane.
    
    This function generates a 3D visualization to inspect the result of the plane and sphere 
    regression. It displays the fitted sphere (representing a fluid interface) and the fitted plane (representing the solid-fluid interface). It also shows the segmented points for fluid, rock, and contact regions based on the input image.
    
    Parameters:
    -----------
    Contact_Angle : np.ndarray
        A 1D array encoding parameters related to the regression:
            - [1:4]  : Center of the measurement region (x, y, z)
            - [4]    : Pore diameter
            - [5:8]  : Normal vector of the estimated plane
            - [9:12] : Sphere center with origin in the contact point.
            - [12]   : Sphere radius.

    
    npimg : np.ndarray
        A 3D image array containing voxel labels (e.g., 0: rock, 1: fluid1, 2: fluid2).
    
    R1 : int, optional (default = 3)
        Radius used for detecting rock-fluid interfaces. Needs to be the same as the value used in the measurement. Default is 3 voxels.
        
    alpha : float, optional (default = 0.8)
        Proportion of the pore size used to define the fluid-fluid search radius. Needs to be the same as the value used in the measurement. Default is 0.5.
    
    Returns:
    --------
    None
        Displays an interactive 3D plot with:
            - A red semi-transparent sphere representing the fitted interface.
            - A blue semi-transparent plane showing the estimated contact angle.
            - An intersection line between the plane and the sphere.
            - Point clouds representing segmented fluid (red), rock (blue), and contact points (green).
            - A yellow point at the origin where the contact angle is measured.
    """
    
    R = Contact_Angle[12]
    Center = Contact_Angle[9:12]
    normal_vector = Contact_Angle[5:8]
    
    P, Rock = ImgTracking(npimg)    
    Fluid = Interfaces(P)
    
    fluidpoints, _ = FindPointsFluid_knowndiameter(Fluid, [int(Contact_Angle[1]), int(Contact_Angle[2]), int(Contact_Angle[3])], int(Contact_Angle[4]), alpha)
    rockpoints = FindPointsRock(Rock, [int(Contact_Angle[1]), int(Contact_Angle[2]), int(Contact_Angle[3])], R=R1)
    
    sphere0 = pv.Sphere(radius = R, center=Center, theta_resolution=120, phi_resolution=60)
    pl = pv.Plotter()
#    pl.add_mesh(sphere0, color='red', opacity = 0.8, show_edges=False)
    pl.add_silhouette(sphere0, color='black', opacity = 1, line_width=8)
#    pl.add_mesh(sphere0.outline(), color='black', line_width=2)

    
    plane = pv.Plane(center=(0.0, 0.0, 0.0), direction=(normal_vector), i_size=3*R, j_size=3*R)
#    pl.add_mesh(plane, color = 'blue', opacity = 0.8, show_edges=False)
    pl.add_silhouette(plane, color='black', opacity = 1, line_width=8)
    outline = plane.extract_feature_edges(boundary_edges=True)
    pl.add_mesh(outline, color='black', line_width=8)

#    pl.add_mesh(plane.outline(), color='black', line_width=1)                

    # Add the intersection line
    intersection = sphere0.slice(normal=normal_vector, origin=[0,0,0])
    pl.add_mesh(intersection, color='black', line_width=8)
    
    rock_points = pv.PolyData(rockpoints)
    pl.add_mesh(rock_points, color='blue', point_size=10, render_points_as_spheres=True)
    
    fluid_points = pv.PolyData(fluidpoints)
    pl.add_mesh(fluid_points, color='red', point_size=10, render_points_as_spheres=True)

    Sub, center_coords = ExtSubvolume(P,  [int(Contact_Angle[1]), int(Contact_Angle[2]), int(Contact_Angle[3])], 3)
    contactpoints = np.array(np.where(Sub == 3)).T - center_coords
    contactpoints = contactpoints #+ 0.5 #+ Contact_Angle[1:4] 
    
    contact_points = pv.PolyData(contactpoints)
    pl.add_mesh(contact_points, color='green', point_size=10, render_points_as_spheres=True)
    
    origin = pv.PolyData([0,0,0])
    pl.add_mesh(origin, color='yellow', point_size=10, render_points_as_spheres=True)
    
    pl.show()
    return

def ViewMeasurement(Contact_Angle, npimg, size, values = [0,1,2], opacity = [0.5,0.5,0.0], R1 = 3, alpha = 0.8):
    """
    Visualize the contact angle measurement region and surrounding features in 3D.

    This function creates a 3D visualization using PyVista to inspect segmented regions 
    of interest within a 3D image (`npimg`). It displays different material phases 
    (rock, fluid, contact points), a fitted sphere, and a plane representing the estimated 
    contact angle. The visualization helps verify the measurement region and interface 
    detection results.

    Parameters:
    -----------
    Contact_Angle : np.ndarray
        A 1D array encoding parameters related to the contact angle measurement:
            - [1:4]  : Center of the measurement region (x, y, z)
            - [4]    : Pore diameter
            - [5:8]  : Normal vector of the estimated plane
            - [9:12] : Sphere center with origin in the contact point.
            - [12]   : Sphere radius.

    npimg : np.ndarray
        A 3D image array containing voxel labels (e.g., 0: rock, 1: fluid1, 2: fluid2).

    size : int or list/tuple of int
        The dimensions of the uniform grid used to build the visualization. If a single integer 
        is given, it will be used for all three dimensions.

    values : list of int, optional (default = [0, 1, 2])
        The voxel values corresponding to the three materials or phases to highlight 
        (e.g., background, fluid, rock).

    opacity : list of float, optional (default = [0.5, 0.5, 0.0])
        The opacity levels for each of the three materials defined in `values`.

    R1 : int, optional (default = 3)
        Radius used for detecting rock-fluid interfaces. Needs to be the same as the value used in the measurement. Default is 3 voxels.
        
    alpha : float, optional (default = 0.8)
        Proportion of the pore size used to define the fluid-fluid search radius. Needs to be the same as the value used in the measurement. Default is 0.5.
        


    Returns:
    --------
    None
        Displays an interactive 3D visualization window with:
            - A clipped view of the 3D image.
            - Highlighted material regions.
            - A fitted sphere.
            - A contact angle plane.
            - Point clouds for detected rock, fluid, and contact interface points.
    """
    center = Contact_Angle[1:4]
    half_size = Contact_Angle[12]*2 
    bounds = (
        center[0] - half_size, center[0] + half_size,
        center[1] - half_size, center[1] + half_size,
        center[2] - half_size, center[2] + half_size
    )   
    
    center = np.array(center, dtype=int)
    half_size = int(half_size)
    
    # Calculate the bounding box indices
    xmin, xmax = center[0] - half_size, center[0] + half_size
    ymin, ymax = center[1] - half_size, center[1] + half_size
    zmin, zmax = center[2] - half_size, center[2] + half_size
    
    # Create a mask for values OUTSIDE the bounding box, and set them to 2
    masked_npimg = np.copy(npimg)
    masked_npimg[:xmin, :, :] = 2
    masked_npimg[xmax:, :, :] = 2
    masked_npimg[:, :ymin, :] = 2
    masked_npimg[:, ymax:, :] = 2
    masked_npimg[:, :, :zmin] = 2
    masked_npimg[:, :, zmax:] = 2
    # Create the grid
    if not isinstance(size, (list, tuple)): 
        size = [size] * 3
    
    if not isinstance(size, (list, tuple)): size = [size]*3
    # Create a uniform grid
    grid = pv.UniformGrid()
    grid.dimensions = (size[0]+1),(size[1]+1),(size[2]+1)
    grid.origin = (0, 0, 0)  
    grid.spacing = (1, 1, 1)  
    grid.cell_data["npimg"] = masked_npimg.ravel(order="F")  

    # Extract the subvolume using clip_box
#    grid = grid.clip_box(bounds, invert=False)

    # Apply thresholding
    threshed_1 = grid.threshold([values[0]-0.1, values[0]+0.1])
    threshed_2 = grid.threshold([values[1]-0.1, values[1]+0.1])  # Adjust to highlight different regions
    threshed_3 = grid.threshold([values[2]-0.1, values[2]+0.1])


    # Plot the 3D visualization
    pl = pv.Plotter()
    
    pl.set_background("white")
    # First layer: Lower opacity (more transparent)
    pl.add_mesh(
        threshed_1, 
        show_scalar_bar=False, 
        show_edges=True, 
        cmap="binary", 
        opacity=opacity[0]  # More transparent
    )

    # Second layer: Higher opacity (less transparent)
    pl.add_mesh(
        threshed_2, 
        show_scalar_bar=False, 
        show_edges=True, 
        cmap="autumn",  # Different colormap for contrast
        opacity=opacity[1]  # Less transparent
    )
#    pl.add_mesh(
#        threshed_3, 
#        show_scalar_bar=False, 
#        show_edges=False, 
#        cmap="winter",  # Different colormap for contrast
#        opacity=opacity[2]  # Less transparent
#    )
    
    
    R = Contact_Angle[12]
    Center = Contact_Angle[9:12] + Contact_Angle[1:4]
    normal_vector = Contact_Angle[5:8]
    
    P, Rock = ImgTracking(npimg)    
    Fluid = Interfaces(P)
    
    fluidpoints, _ = FindPointsFluid_knowndiameter(Fluid, [int(Contact_Angle[1]), int(Contact_Angle[2]), int(Contact_Angle[3])], int(Contact_Angle[4]), alpha)
    rockpoints = FindPointsRock(Rock, [int(Contact_Angle[1]), int(Contact_Angle[2]), int(Contact_Angle[3])], R=R1)


    Sub, center_coords = ExtSubvolume(P,  [int(Contact_Angle[1]), int(Contact_Angle[2]), int(Contact_Angle[3])], 3)
    contactpoints = np.array(np.where(Sub == 3)).T - center_coords
    contactpoints = contactpoints + Contact_Angle[1:4] + 0.5

    fluidpoints = fluidpoints + Contact_Angle[1:4] + 0.5
    rockpoints = rockpoints + Contact_Angle[1:4] + 0.5

    
    
    
    sphere0 = pv.Sphere(radius = R, center=Center, theta_resolution=120, phi_resolution=60)
    pl.add_mesh(sphere0, color='red', opacity = 1, show_edges=False)
    
    plane = pv.Plane(center=Contact_Angle[1:4], direction=(normal_vector), i_size=3*R, j_size=3*R)
    pl.add_mesh(plane, color = 'blue', opacity = 1, show_edges=False)
        
    rock_points = pv.PolyData(rockpoints)
    pl.add_mesh(rock_points, color='blue', point_size=10, render_points_as_spheres=True)

    contact_points = pv.PolyData(contactpoints)
    pl.add_mesh(contact_points, color='green', point_size=10, render_points_as_spheres=True)
    
    fluid_points = pv.PolyData(fluidpoints)
    pl.add_mesh(fluid_points, color='red', point_size=10, render_points_as_spheres=True)

#    origin = pv.PolyData(Contact_Angle[1:4])
#    pl.add_mesh(origin, color='yellow', point_size=10, render_points_as_spheres=True)
    
    pl.show()
    return

def ResultAnalysis(Contact_Angle, num_components=1, expected_angle = [-1]):
    """
    Analyze contact angle measurements by fitting a Gaussian Mixture Model (GMM).

    This function fits one or more Gaussian distributions to the input contact angle measurements 
    using a Gaussian Mixture Model. Each individual measurement is assigned to the most probable 
    Gaussian component and replaced with that component's mean. The function also generates a 
    plot showing the original histogram, the fitted Gaussian components, their means, and 
    (optionally) any expected contact angle values.

    Parameters:
    -----------
    Contact_Angle : np.ndarray
        A 2D array where the contact angle data is stored in the first row (shape: [4, N]).
    num_components : int, optional (default=1)
        The number of Gaussian components to fit in the mixture model.
    expected_angle : list of float, optional (default=[-1])
        A list of known or expected contact angle values to be optionally marked on the plot.
        Use -1 to skip plotting expected values.

    Returns:
    --------
    gmm_means : np.ndarray
        The mean of each Gaussian component fitted by the GMM.
    gmm_weights : np.ndarray
        The weight (importance) of each Gaussian component.
    gmm_stds : np.ndarray
        The standard deviation of each Gaussian component.
    Mean_Angles : np.ndarray
        A modified copy of the input array `Contact_Angle`, where the first row contains 
        the most probable mean values from the GMM assigned to each measurement.
    """
    
    data = np.copy(Contact_Angle[0,:])
    
    # Fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=num_components, random_state=42)
    gmm.fit(data.reshape(-1, 1))

    gmm_means = gmm.means_.flatten()
    gmm_weights = gmm.weights_
    gmm_stds = np.sqrt(gmm.covariances_).flatten()

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=int((np.max(data) - np.min(data))/3), density=True, alpha=0.5, label='Histogram')

    # Optional: Real known values (adjust or remove if not needed)
    for ang in expected_angle:        
        if ang != -1:
            plt.axvline(x=ang, color='black', linestyle='--', label=f'Expected: {ang}')

    x_vals = np.linspace(data.min(), data.max(), 1000)
    total_pdf = np.zeros_like(x_vals)

    colors = plt.cm.Set1(np.linspace(0, 1, num_components+1))  # Get distinct colors
    gmm_pdf = np.zeros((num_components+1, data.size))
    pdf_dis = np.zeros(data.size)
    for i in range(num_components):
        mean = gmm_means[i]
        std = gmm_stds[i]
        weight = gmm_weights[i]

        # Plot mean line
        plt.axvline(x=mean, color=colors[i], linestyle='--', label=f'GMM Mean {i+1}: {mean:.2f} \nStDev {i+1}: {std:.2f}')

        # Plot individual Gaussian component
        gmm_pdf[i,:] = gmm_weights[i] * norm.pdf(data, gmm_means[i], gmm_stds[i])
        component_pdf = weight * norm.pdf(x_vals, mean, std)
        plt.plot(x_vals, component_pdf, color=colors[i], linestyle='dashed')
        total_pdf += component_pdf
        
    for i in range(num_components):
        pdf_dis = np.where((gmm_pdf[i, :] > np.max(np.delete(gmm_pdf, i, axis=0), axis=0)), gmm_means[i], pdf_dis)

    plt.xlim(0,180)
    # Plot total mixture model
    plt.plot(x_vals, total_pdf, color='black', linestyle='solid', label='GMM Fit')

    plt.legend()
    plt.title('Estimated PDF and Peaks')
    plt.show()
    
    Mean_Angles = np.copy(Contact_Angle[0:4,:])
    Mean_Angles[0,:] = pdf_dis 
    return gmm_means, gmm_weights, gmm_stds, Mean_Angles


def FilterMeasurements(Contact_Angle, R1 = 5, mpe = 20, mpp = 0.5, maxerr_e = 0.5, maxerr_p = 0.4):
    mask = (Contact_Angle[13, :] < maxerr_p) & \
        (Contact_Angle[14, :] < maxerr_e) & \
        (Contact_Angle[15, :] > np.pi*(R1-mpp)**2) & \
        (Contact_Angle[16, :] > mpe)
    Contact_Angle = Contact_Angle[:, mask]
    return Contact_Angle


