
# coding: utf-8

# In[1]:

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpimg
import numpy as np
from IPython.display import HTML
import os, sys
import glob
import moviepy
from moviepy.editor import VideoFileClip
from moviepy.editor import * 
from IPython import display
from IPython.core.display import display
from IPython.display import Image
import pylab
import scipy.misc



# In[2]:

def region_of_interest(img):
    mask = np.zeros(img.shape, dtype=np.uint8) #mask image
    roi_corners = np.array([[(200,675), (1200,675), (700,430),(500,430)]], 
                           dtype=np.int32) # vertisies seted to form trapezoidal scene
    channel_count = 1#img.shape[2]  # image channels
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)   
    masked_image = cv2.bitwise_and(img, mask)
 
    return masked_image



# In[3]:

def CTHR(img):               # Threshold Yellow anf White Colos from RGB, HSV, HLS color spaces
    
    HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # For yellow
    yellow = cv2.inRange(HSV, (20, 100, 100), (50, 255, 255))

    # For white
    sensitivity_1 = 68
    white = cv2.inRange(HSV, (0,0,255-sensitivity_1), (255,20,255))

    sensitivity_2 = 60
    HSL = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    white_2 = cv2.inRange(HSL, (0,255-sensitivity_2,0), (255,255,sensitivity_2))
    white_3 = cv2.inRange(img, (200,200,200), (255,255,255))

    bit_layer = yellow | white | white_2 | white_3
    
    return  bit_layer



# In[4]:

from skimage import morphology

def SobelThr(img):                     # Sobel edge detection extraction
    gray=img
    
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=15)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=15)
    
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
   
    
    binary_outputabsx = np.zeros_like(scaled_sobelx)
    binary_outputabsx[(scaled_sobelx >= 20) & (scaled_sobelx <= 255)] = 1
   
    
    
    binary_outputabsy = np.zeros_like(scaled_sobely)
    binary_outputabsy[(scaled_sobely >= 100) & (scaled_sobely <= 150)] = 1

    
    mag_thresh=(100, 200)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
   

    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    binary_outputmag = np.zeros_like(gradmag)
    binary_outputmag[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    combinedS = np.zeros_like(binary_outputabsx)
    combinedS[(((binary_outputabsx == 1) | (binary_outputabsy == 1))|(binary_outputmag==1)) ] = 1
   
    return combinedS



# In[5]:

def combinI(b1,b2):     ##Combine color threshold + Sobel edge detection

    combined = np.zeros_like(b2)
    combined=b1|b2

    
    return combined


# In[6]:

def prespectI(img):               # Calculate the prespective transform and warp the Image to the eye bird view
    
 

    src=np.float32([[728,475],
                  [1058,690],
                  [242,690],
                  [565,475]])
    
    dst=np.float32([[1058,20],
                  [1058,700],
                  [242,700],
                  [242,20]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (1280,720), flags=cv2.INTER_LINEAR)
    
    return (warped, M)


# In[7]:

def undistorT(imgorg):              # Calculate Undistortion coefficients


    nx =9
    ny = 6
    objpoints = []
    imgpoints = []
    objp=np.zeros((6*9,3),np.float32)
    objp[:,:2]=np.mgrid[0:6,0:9].T.reshape(-1,2)


    images=glob.glob('./camera_cal/calibration*.jpg')
    for fname in images:                 # find corner points and Make a list of calibration images
            img = cv2.imread(fname)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (6,9),None)

            # If found, draw corners
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)
                # Draw and display the corners
                #cv2.drawChessboardCorners(img, (nx, ny), corners, ret)    


    return cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)


    


# In[8]:

def undistresult(img, mtx,dist):     #    undistort frame
    undist= cv2.undistort(img, mtx, dist, None, mtx)
    
    return undist



# In[9]:

def LineFitting(wimgun):                  #Fit Lane Lines

    # Set minimum number of pixels found to recenter window
    minpix = 20
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
        

    histogram = np.sum(wimgun[350:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((wimgun, wimgun, wimgun))


    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])

    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 9

    # Set height of windows
    window_height = np.int(wimgun.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = wimgun.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin =80




    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = wimgun.shape[0] - (window+1)*window_height
        win_y_high = wimgun.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, wimgun.shape[0]-1, wimgun.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
#     out_img = np.dstack((wimgun, wimgun, wimgun))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    
#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
#     plt.xlim(0, 1280)
#     plt.ylim(720, 0)
#     plt.imshow(out_img)
# #     plt.savefig("./output_images/Window Image"+str(n)+".png")
#     plt.show()

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
#     plt.title("r")

#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
#     plt.xlim(0, 1280)
#     plt.ylim(720, 0)
#     plt.imshow(result)
# #     plt.savefig("./output_images/Line Image"+str(n)+".png")
#     plt.show()


    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    #print(left_curverad, right_curverad)
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
#     y_eval = np.max(ploty)

# # Calculate the new radias of curvature
   
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
#    # left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
#    # right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    
#     camera_center=wimgun.shape[0]/2
#     #lane_center = (right_fitx[719] + left_fitx[719])/2    
    lane_offset = (1280/2 - (left_fitx[-1]+right_fitx[-1])/2)*xm_per_pix
#     print(left_curverad1, right_curverad1, lane_offset)

    return (left_fit, ploty,right_fit,left_curverad, right_curverad,lane_offset)

   

 # Create an image to draw the lines on
def unwrappedframe(img,pm, Minv, left_fit,ploty,right_fit):
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    
    warp_zero = np.zeros_like(pm).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    
    # Combine the result with the original image

    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    
    

