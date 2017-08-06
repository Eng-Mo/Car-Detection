
# Vehicle Detection Project

[//]: # (Image References)
[image1]: ./output_images/sample_0.png
[image2]: ./output_images/sample_1.png
[image3]: ./output_images/sample_2.png
[image4]: ./output_images/sample_3.png
[image5]: ./output_images/sample_4.png
[image6]: ./output_images/sample_5.png
[image7]: ./output_images/result_0.png
[image8]: ./output_images/result_1.png
[image9]: ./output_images/result_2.png
[image10]: ./output_images/result_3.png
[image11]: ./output_images/result_4.png
[image12]: ./output_images/result_5.png

[video1]: ./out_video2.avi


---
This project is an implementation of the Histogram of Gradient (HOG) and Support vector machine (SVM) approach for vehicle detection and tracking. the followings are the documentation of the software python code and description report for the pipeline and the issues regarding the research topic.

---
### Project Goals
This project aims to investigate the computer vision techniques and machine learning to detect multiple vehicles and the approach that used for the implementation as following:
1. collecting labeled dataset `vehicle` and `non-vehicle`.
2. Extracting Histogram of Gradient features.
3. extract histogram of color and binned color features.
4. concatenate HOG feature vector with color features.
5. Train data using Linear SVM.
6. Detecting Vehicles by extracting HOG features from the frame and predict multi scale window search.
7. Creating a heatmap for removing false detections.
8. Draw a bounding box for each detected vehicle.

### Project Files
1. `Classifier.ipynb` contains the pipeline for training dataset and `Classifier.py` is the python script that produce pickle file `clf.pkl` that contains all the training parameters.

2. `Vehicle_Tracking_pipeline.ipynb` contains the pipeline for the vehicle detection in video and `Vehicle_Tracking_pipeline.py` is the python script.
To run the code you need to import `Classifier.py` and call `processData()` for loading and training data. Then Run `Vehicle Tracking pipeline.ipynb`

---

# Project Pipline
This section contains the training classifier and video analysis examinations.

## Featurs extraction and training pipline 


### Loading data
I started by reading in all `vehicle` and `non-vehicle` data form GTI and KITTI dataset in function `loadData()`in `Classifier.py`. the balance between number of vehicle and non-vehicle was considered. 8000 data sample for vehicle and non-vehicle images were selected and processed. In next step from code below the features extracted from all data by calling function `featuresCalc()`.


### Feature extraction `extractFeature()`:
```python
def featuresCalc(car, not_car,spatial_size,hist_bins,orient,pix_per_cell,cell_per_block,hog_ch):
    
    car_features=extractFeature( car,spatial_size,hist_bins,orient,pix_per_cell,cell_per_block,hog_ch)
    not_car_features=extractFeature(not_car,spatial_size,hist_bins,orient,pix_per_cell,cell_per_block,hog_ch)
    
    return car_features,not_car_features
```

The feature extraction starts with converting the image to `YCrCb` color space as after exploration I found a good representation for grouping each channel value and after trying other color spaces I found `YCrCb` is the best color space that detected black and white color.

### Spatial binning `bin_spatial()`:
The color spatial feature was extracted by resizing the image to 32x32, and this size is good for detection and for reducing the processing time. 

### Histogram features `HistofClolor()`:
Histogram features were extracted from the 3-channels and concatenated in variable `hist_features`

### Histogram of Oriented Gradients `get_hog_features()`

The code for this step is contained in function `get_hog_features()`. The HOG extraction processed on single gray image for reducing the computation. the following are the parameters of hog function and the result showed below.

```python
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
```
I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). The issue in this step is compromising between extracting good features and computation time. After exploration the gray scale was chosen and `hog()` parameter were tuned to produce a general shape for cars and the parameters above produced the following result for vehicle and non-vehicle features. 
![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
Higher parameter will produce big blocks of Hog features which car shape can be generalized to other shape. lower parameters produce much better results but it increases computation.

### Classifier SVM ` trainClf()`

After creating features array I combined the vehicle and non-vehicle features in `X` and labeled the data in `y`. The data was spitted into train data `X_train, y_train` and test data `X_test, y_test` then I shuffled the data. 
I trained a linear SVM using same number of combined `car` and `non-car` data. the data are scaled with `StandardScaler()`  and produced `scaled_X = X_scaler.transform(X)`. the classifier parameter `C=.001` to make the training more robust against to the false positives and it produced accuracy `SVC =  0.98`

## Tracking  Pipline `Vehicle Tracking pipeline.ipynb`

The vehicle detection starts with `find_cars()` function that return the predicted vehicles’ window list.

### Window Search 

To make more efficient sliding window I used Hog Sub-sampling technique to reduce the computation by extracting the Hog features in different two window scales `(2.2,1.2)` from the frame once and then sub-sampled to get all of its overlaying windows. and then run SVC prediction on each window. the reason of chosing the 2.2 and 1.2 scale is that to detect the vehicle in diffrent scale as 2.2 when the vehicle close and 1.2 when the vehicle further far. after I run the SVC prediction on each windows by using `SVC.decision_function()` to set threshold `.03` to reduce false detection. 

### Removing false detection

Every positive detection was recorded in each frame of the video and from the positive detections I created a heatmap and then thresholded that map to identify vehicle positions by value `2 window`. more flase detection removal technique was used in the tracking system.
### Grouping Windows ` findcont()`

`findcont()` function analyze the heat map and turn it to thresholded Image and this function is implemented because cars can be detected in multiple windows and it might some of them doesn't pass the heat map threshold so the result two bounding boxes for the same car. therefore this function group the closed boxes and then use find `findContours()` to find contours and return the coordinates of each vehicle. and the following are the results of the test images.

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

### Vehicle Tracking

I implemented `Class Car` to store the coordinates of each detected car and in list of car `cars=[]`. In order to remove all the false the detection I made condition that the vehicle has tobe detected in 3 sequenced frames to be considered as a car. the following code to keep track of the detected vehicles and update its bounding box by taking the average box of the current frame and previous frame. the tracking is done by calculating the rectangle overlapping of the current rectangles and the Cars' rectangle. if the overlapping >1000 in three sequenced frames so this rectangle represent a vehicle and then call `car.drawCar(frame)` to draw the bounding box. 

```python
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
```
### Result

Here's a [link to my video result](https://youtu.be/kjDp3Gl1GDo) or open `result_output.mp4`

---

### Discussion
The main issue in using SVM is the computation time as I achieved `3.2s` when I used normal multi scale sliding windows. In this update I achieved .6 ms frame per second processing time by using HOG Sub-sampling and integrating the Lane detection project running on GPU on AWS. It still far away to be real time, however the proposed solution has proved high reliability in detecting the cars and ignore the false detections and the followings are the recommended further works.

1. Using Deep learning is highly recommended duet to its reliability and high processing speed. Here I preferred to enhance the SVM for learning purpose.

2. To Implement real time detection using SVM , Parallel Processing will speed up the processing time but it needs to be implemented with C++.

3. reducing the search area can make the classifier focus on particular objects in the Image rather than consider the specified area. to Achieve that I Think Global background segmentations and segments the moving objects in foreground then we run the classifier only on the foregrond objects. Back ground segmentation can be challenging on moving cameras and it can achieved by using 3D cameras or segmentation using deep learning techniques.

5. Other tracking approach can be investigate like camshift, particle filter to track car rectangles motion and run the classifier every 10 frame will speed up the time processing. 