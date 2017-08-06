
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
import glob
import cv2
import pdb
import sklearn
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn import svm, grid_search, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import time


# In[2]:

#Load data function used combination data from GTI and KITTI
def loadData():
    images=glob.glob('./dataset/vehicles/GTI_Far/image*.png')
    train_car=[]    
    train_not_car=[]
#     all data was read using cv2.imread() to avoid diffrent imge format

    for fname in images:
        img=cv2.imread(fname)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)    
        train_car.append(img)
        
    images=glob.glob('./dataset/vehicles/GTI_Left/image*.png')
    for fname in images:
        img=cv2.imread(fname)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  
        train_car.append(img)
        
    images=glob.glob('./dataset/vehicles/GTI_Right/image*.png')

    for fname in images:
        img=cv2.imread(fname)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        train_car.append(img)
        

    images=glob.glob('./dataset/vehicles/MiddleClose/image*.png')
    for fname in images:
        img=cv2.imread(fname)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        train_car.append(img)
        

    images=glob.glob('./dataset/vehicles/KITTI_extracted/*.png')
    for fname in images:
        img=cv2.imread(fname)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        train_car.append(img)
        


    images=glob.glob('./dataset/non-vehicles/GTI/image*.png')
    for fname in images:
        img=cv2.imread(fname)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        train_not_car.append(img)
        

    images=glob.glob('./dataset/non-vehicles/Extras/extra*.png')
    for fname in images:
        img=cv2.imread(fname)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        train_not_car.append(img)
      

   
    return (train_car,train_not_car)


# In[3]:

from mpl_toolkits.mplot3d import Axes3D
# this function to draw color space in 3D
def plot3d(pixels, colors_rgb,
        axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulatio


# In[4]:

def bin_spatial(img, size=(32,32)):
    # binning spatial feature with size (16X16) to reduce computation
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    
    return np.hstack((color1, color2, color3))


# In[5]:


def HistofClolor(img,nbins=32, bins_range=(0, 255)):
#     Color feature histogram
#     hist_features=[]
    yhist=np.histogram(img[:,:,0], bins=32, range=(0, 255))
    crhist=np.histogram(img[:,:,1], bins=32, range=(0, 255))
    cbhist=np.histogram(img[:,:,2], bins=32, range=(0, 255))
    hist_features= np.concatenate((yhist[0], crhist[0], cbhist[0])) #concatinate color features    
    return hist_features 


# In[6]:

def get_hog_features(img_channel, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False,ch=3):
    
#     extracting Hog feature from gray scal image (for single channel) to reduce computation
# the commetted lines for extracting Hog for 3 three channels
    features_data = []
    if  ch==3:
#         img=cv2.cvtColor(imgo,cv2.COLOR_RGB2GRAY)  #conver to gray
        ch_0=imgo[:,:,0]
        ch_1=imgo[:,:,1]
        ch_2=imgo[:,:,2]
        
#         plt.imshow(img)
#         plt.show()

        # Use skimage.hog() to get both features and a visualization
        features_0 = hog(ch_0, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell)
                                  , cells_per_block=(cell_per_block, cell_per_block),
                                  visualise=vis, feature_vector=feature_vec)
#         plt.title('sample HOG')
#         plt.imshow(imagehog)
#         plt.savefig('./output_images/hog_sample.png')
#         plt.show()


        features_1 = hog(ch_1, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell)
                                  , cells_per_block=(cell_per_block, cell_per_block),
                                  visualise=vis, feature_vector=feature_vec)
#         plt.imshow(imagehog)
#         plt.show()
# #         

        features_2 = hog(ch_2, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell)
                                  , cells_per_block=(cell_per_block, cell_per_block),
                                  visualise=vis, feature_vector=feature_vec)
#         plt.imshow(imagehog)
#         plt.show()
        
#         features_0=features_0.reshape(-1)  # reshape feature vector
#         features_1=features_1.reshape(-1)
#         features_2=features_2.reshape(-1)
        features_data.append(features_0)
        features_data.append(features_1)
        features_data.append(features_2)
        

#         features_data.append(np.hstack((features_0,features_1,features_2)))
#         features_data = np.ravel(features_data) 
#         features_data.append(features_0)  # appending Hog features

        
        return features_data


    else:
        
        features_0= hog(img_channel, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell)
                                  , cells_per_block=(cell_per_block, cell_per_block),
                                  visualise=vis, feature_vector=feature_vec)
     

        # Use skimage.hog() to get features only
                
        return features_0



# In[7]:

def extractFeature(data, spatial_size,hist_bins,orient,pix_per_cell,cell_per_block,hog_ch):


    
    
    file_features=[]  # concatinated features per image
  
    features=[]   # all features data
    feature_hog=[]  #Hog features
#     HSV_data=[]
    count=0
    for i in data:
#         plt.imshow(i)
#         plt.show()
       
        
              
        
        img_YCrCb = cv2.cvtColor(i, cv2.COLOR_RGB2YCrCb)  #convert image to YCrCb
        
        
 #         pdb.set_trace()
        spatial_features=bin_spatial(img_YCrCb, size=spatial_size)  #extract spatial features
#         pdb.set_trace()
        hist_features=HistofClolor(img_YCrCb,nbins=hist_bins, bins_range=(0, 255))  #extract histogram features
        if hog_ch==3:
            
            feature_hog0, feature_hog1, feature_hog2=get_hog_features(img_YCrCb, orient,                             #extract Hog features
                            pix_per_cell, cell_per_block, 
                            vis=False, feature_vec=False,ch=hog_ch)
        else:
            gray=cv2.cvtColor(i,cv2.COLOR_RGB2GRAY)
            feature_hog=get_hog_features(gray, orient,                             #extract Hog features
                            pix_per_cell, cell_per_block, 
                            vis=False, feature_vec=False,ch=hog_ch)
            
        
        feature_hog=np.reshape(feature_hog,-1)             #reshape Hog features
#         print(spatial_features.shape)
#         print(feature_hog.shape)
#         print(hist_features.shape)
        file_features=np.concatenate((spatial_features, 
                                      hist_features,feature_hog),0)  #concatenate features image
        
        
        features.append(file_features)  #append features image
        count+=1
    
    
    return features


# In[8]:

def featuresCalc(car, not_car,spatial_size,hist_bins,orient,pix_per_cell,cell_per_block,hog_ch):
    
    car_features=extractFeature( car,spatial_size,hist_bins,orient,pix_per_cell,cell_per_block,hog_ch)
        #pdb.set_trace()
    not_car_features=extractFeature(not_car,spatial_size,hist_bins,orient,pix_per_cell,cell_per_block,hog_ch)
    
    return car_features,not_car_features


# In[9]:

def trainClf(features_c,features_nc,spatial_size,hist_bins,orient,pix_per_cell,cell_per_block,hog_ch):
  

    # Create an array stack, NOTE: StandardScaler() expects np.float64

    X = np.vstack((features_c, features_nc)).astype(np.float64)  # stack features 

    from sklearn.preprocessing import  StandardScaler
    # Fit a per-column scaler
    X_scaler = StandardScaler(copy = False).fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)    #Normalise data
    spatial=32
    histbin=32
    y = np.hstack((np.ones(len(features_c)), np.zeros(len(features_nc))))   #create lable array
    rand_state = np.random.randint(0, 1000)
    X_train, X_test, y_train, y_test = train_test_split(                           #split data to train and test 40%
        scaled_X, y, test_size=0.3, random_state=rand_state)
    X_train, y_train = shuffle(X_train, y_train)

    print('Using spatial binning of:',spatial,
        'and', histbin,'histogram bins')
    print('Feature vector length:', len(X_train[0]))

    parameters = {'kernel':['linear'], 'C':[.001]}
    # svc = svm.SVC()
    # gs = GridSearchCV(svc, parameters,cv=2,n_jobs=4,verbose=3,)         #GridsearchCV was very slow
    # gs.fit(X_train, y_train)

    # aGrid = aML_GS.GridSearchCV( aClassifierOBJECT, param_grid = aGrid_of_parameters,
    #                             cv = cv, n_jobs = n_JobsOnMultiCpuCores, verbose = 5 )


    # # Use a linear SVC 
    svc = LinearSVC(C=.001,dual=False)     #create classifier
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)     #fit train data
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))  #predict test data
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = len(X_test)
    # print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    # print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    import pickle
    # now you can save it to a file

    pickle_file = 'clf.pkl'
    print('Saving data to pickle file...')
    with open(pickle_file, 'wb') as f:
        pickle.dump(
            {
                'svc':svc, 
                'X_scaler': X_scaler,
                'spatial_size': spatial_size,
                'hist_bins': hist_bins,
                'orient': orient,
                'pix_per_cell': pix_per_cell,
                'cell_per_block': cell_per_block,
                'hog_channel': hog_ch,
            },

             f, pickle.HIGHEST_PROTOCOL)


# In[10]:

def processData():
    
    #     Features parameter

    spatial_size=(32,32)  #spatial binned size
    hist_bins=32    
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_ch = 1


    car_, not_car_=loadData()
    
    carfeatures, notcar_features=featuresCalc(car_,not_car_,spatial_size,hist_bins,orient,pix_per_cell
                                              ,cell_per_block, hog_ch)
    trainClf(carfeatures,notcar_features,spatial_size,hist_bins,orient,pix_per_cell,cell_per_block,hog_ch)


# In[11]:



