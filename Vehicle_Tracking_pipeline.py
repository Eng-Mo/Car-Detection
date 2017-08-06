
# coding: utf-8

# In[ ]:

import math as Math
class Car(object):
    def __init__(self, detection=False,detect=1 ,x1=0,y1=0,x2=0,y2=0,width=0,hight=0):
        self.detected=detection
        self.n_detections =detect
        self.x11=x1
        self.x12=x2
        self.y11=y1
        self.y12=y2
        self.width=width
        self.hight=hight
#         self.n_nondetections=0
#         self.xpixles=None
#         selfyxpixles=None
#         self.recent_xfitted=[]
#         self.bestx=None
#         self.recent_yfitted=[]
#         self.besty=None
#         self.recent_wfitted=[]
#         self.bestw=None
#         self.recent_hfitted=[]
#         self.besth=None

    def calcOverlap(self, pts):
        x21=pts[0][0]
        y21=pts[0][1]
        x22=pts[1][0]
        y22=pts[1][1]
        x_overlap = max(0, min(self.x12,x22) - max(self.x11,x21))
        y_overlap = max(0, min(self.y12,y22) - max(self.y11,y21))
        return x_overlap * y_overlap;
    
    
    def updateDetection(self):
        self.n_detections+=1
        
        
    def updateCar(self, coordinates):
        self.x11= int((self.x11+coordinates[0][0])/2)
        self.x12=int((self.x12+coordinates[1][0])/2)
        self.y11=int((self.y11+coordinates[0][1])/2)
        self.y12=int((self.y12+coordinates[1][1])/2)
        self.width=self.x11+self.x12
        self.hight=self.y11+self.y12
        
    
    def drawCar(self,image):
        cv2.rectangle(image, (self.x11,self.y11), (self.x12,self.y12), (255, 0, 0), 2)
        return image
        
        


# In[ ]:

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img



# In[ ]:

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


# In[ ]:

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


# In[ ]:

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


# In[ ]:

def findcont(img,f):   #find contours in heat map and dilate them to group closed high heats
    
    windows=[]       
    img=img.astype(np.uint8)   #conver Image to int numpy
#     plt.imshow(img,cmap='gray')
#     plt.show()

    
    ret,thresh1 = cv2.threshold(img,5,255,cv2.THRESH_BINARY)
#     plt.title('threshold')
#     plt.imshow(thresh1, cmap='gray')
#     plt.show()
#     kernel1 = np.ones((5,5), np.uint8)
    kernel2 = np.ones((15,15), np.uint8)
    img_erod = cv2.erode(thresh1, kernel2, iterations=1) 
#     plt.title('erode')
#     plt.imshow(img_erod, cmap='gray')
#     plt.show()

#     plt.show()
    
    im2, cnts, hierarchy = cv2.findContours(img_erod.copy(), 
                                            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #find contours
#     cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:3]
    cari = None
    
    for c in cnts:              #obtain four point window to be drawn       
        (x,y,w,h) = cv2.boundingRect(c)
#         cv2.rectangle(f, (x,y), (x+w,y+h), (255, 0, 0), 2)
        w=((x,y),(x+w,y+h))
        windows.append(w)

    return f ,windows   


# In[ ]:

from LaneDetect import *     # Lane detection piplne

def laneDetection(img):
    global font
    
    
    gray =cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobel_edge= SobelThr(gray) 
    color_threshld= CTHR(img)
    comI=combinI(sobel_edge,color_threshld) 
    roib=region_of_interest(comI)
    undistI=undistresult(roib, mtx,dist)
    pI, pM=prespectI(undistI)  
    pI = cv2.inRange(pI, 10, 255)
    Minv = np.linalg.inv(pM)
    [left_fit, ploty,right_fit,lc, rc, offset]= LineFitting(pI)
    uW=unwrappedframe(img,pI,Minv,left_fit, ploty,right_fit)
    uW=cv2.putText(uW,'Curvature left: %.1f m'%lc,(50,50), 
                    font, 1,(255,255,255),2,cv2.LINE_AA)
    uW=cv2.putText(uW,'Curvature right: %.1f m'%rc,(50,100),
                   font, 1,(255,255,255),2,cv2.LINE_AA)
    uW=cv2.putText(uW,'Center car offset: %.1f m'%offset,(50,150),
                   font, 1,(255,255,255),2,cv2.LINE_AA)
    return uW
    


# In[ ]:

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
# import cv2
from sklearn.preprocessing import StandardScaler

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale):
    
    with open('clf.pkl', 'rb') as f:
        data =pickle.load(f)
        
        
    svc_p = data['svc'] 
    X_scaler_p = data['X_scaler']

    spatial_size = data['spatial_size']
    hist_bins = data['hist_bins']
    orient = data['orient']
    pix_per_cell = data['pix_per_cell']
    cell_per_block = data ['cell_per_block']
    n_channel = data['hog_channel']



    

    draw_img = np.copy(img)
#     img = img.astype(np.float32)/255
    
    
    
    img_tosearch = img[ystart:ystop,300:1280,:]
    gray_i=cv2.cvtColor(img_tosearch,cv2.COLOR_RGB2GRAY)
    
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        gray_i=cv2.resize(gray_i, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = gray_i#ctrans_tosearch[:,:,0]
#     ch2 = ctrans_tosearch[:,:,1]
#     ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step =1 # Instead of overlap, define how many cells to step
    nxsteps = ((nxblocks) - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(gray_i, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False,ch=n_channel)
#     hog2 = get_hog_features_sc(ch2, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
#     hog3 = get_hog_features_sc(ch3, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
    
    
    window_list=[]
    hog_features=[]

    
#     plt.imshow(ch1)
#     plt.show()
   
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window]
#             hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window]
#             hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window]
            
# #             hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
#             hog_features.append(hog_feat1)
#             hog_features.append(hog_feat2)
#             hog_features.append(hog_feat3)
#             hog_features=np.ravel(hog_feat1)
            hog_features=np.reshape(hog_feat1,-1)      
            

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = HistofClolor(subimg,nbins=32, bins_range=(0, 255))
#             hist_features=np.array(hist_features)
#             hist_features=hist_features.reshape(-1)
#             print(hist_features.shape)
#             print(spatial_features.shape)
#             print(hog_features.shape)
            
#             hist_features=hist_features.astype(np.float64)
#             spatial_features=spatial_features.astype(np.float64)
#             print(hist_features.dtype)
#             print(spatial_features.dtype)
#             print(hog_features.dtype)
            X_Test=np.concatenate((spatial_features, 
                                      hist_features,hog_features),0)
           
           
            
#             X_scaler = StandardScaler(copy = False).fit(X_Test)
            
         

            # Scale features and make a prediction
            test_features = X_scaler_p.transform(X_Test) 
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc_p.decision_function(test_features)
            
#             print(test_prediction)
            if test_prediction >0.03:
#                 print(test_prediction)
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left+300, ytop_draw+ystart),
                                  (xbox_left+win_draw+300, ytop_draw+win_draw+ystart),(0,0,255),6)
                sx=xbox_left+300
                sy=ytop_draw+ystart
                ex=xbox_left+win_draw+300
                ey= ytop_draw+win_draw+ystart
#                 print(sx)
#                 print(sy)
#                 print(ex)
#                 print(ey)
                window_list.append(((sx,sy),(ex,ey)))
            hog_features=[]
    
    
                

#     plt.imshow(draw_img)
#     plt.show()

                
    return window_list
    


# In[ ]:

from moviepy.editor import *
from moviepy.editor import VideoFileClip
from moviepy.video.VideoClip import VideoClip
import moviepy
import matplotlib as mpimg
import time
from moviepy.Clip import Clip
from scipy.ndimage.measurements import label
import scipy.misc
from Classifier import *
# from LaneDetect import *

def processImage (frame):
    ystart = 400
    ystop = 600
    scale = 1.6
    global cars
    
    global mtx
    global dist
 
    
    
    heat = np.zeros_like(frame[:,:,0]).astype(np.float) # create heat map
        
    
    t=time.time()
    

    hot_windows=find_cars(frame, 400, ystop, 2.2)
    hot_windows.extend(find_cars(frame, 400, 500, 1.1))
    heat = add_heat(heat,hot_windows)
    #         Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    fimage, windows=findcont(heatmap,frame)     #group closed hot peaks
    I=frame
    if windows:
            if not cars:

                for coordinates in windows:

                    width=coordinates[1][0]-coordinates[0][0]
                    hight=coordinates[1][1]-coordinates[0][1]
                    car=Car(True,1,coordinates[0][0],coordinates[0][1],coordinates[1][0],
                            coordinates[1][1],width,hight)
                    cars.append(car)
            else:
                for coordinates in windows:
                    width=coordinates[1][0]-coordinates[0][0]
                    hight=coordinates[1][1]-coordinates[0][1]
                    detection_flag=0
                    
                    for car in cars:
                        overlap_area=car.calcOverlap(coordinates)
#                         print("overlap= ",overlap_area)
                        
                        if overlap_area>1000:
                            if car.n_detections>2:
                                car.updateCar(coordinates)
                                I= car.drawCar(frame)
                                detection_flag=1
                            else:
                                car.updateDetection()
                                detection_flag=1
                    if detection_flag==0:
                        cars.append(Car(True,1,coordinates[0][0],coordinates[0][1],coordinates[1][0],
                        coordinates[1][1],width,hight))
#     labels = label(heatmap)
#     draw_img=draw_labeled_bboxes(np.copy(frame),labels)    #draw windows


#     final_image=laneDetection(out_img,frame)   # detect lanes
    t2 = time.time()
    tf=t2-t
#     print(tf)
    result= laneDetection(I)
    final_image=cv2.putText(result,'Car detection Time process per frame= %.2f ms'%tf,(10,200),
                   font, 1,(255,255,255),2,cv2.LINE_AA)
#     plt.imshow(final_image)
#     plt.show()
    return final_image
    
    
    



# In[ ]:

from moviepy.editor import VideoFileClip
from IPython.display import HTML
cars=[]


font = cv2.FONT_HERSHEY_SIMPLEX
test_ouput='result_output.mp4'
clip = VideoFileClip('project_video.mp4')

frameClibration= clip.get_frame(0)
[ret, mtx, dist, rvecs,tvecs] =undistorT(frameClibration)

test_clip= clip.fl_image(processImage)

test_clip.write_videofile(test_ouput,audio=0)

