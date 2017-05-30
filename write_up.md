
# Vehicle Detection Project

[//]: # (Image References)
[image1]: ./output_images/sample_car.png
[image2]: ./output_images/sample_not_car.png
[image3]: ./output_images/hog_sample.png
[image4]: ./output_images/Detected_cars_1.png
[image5]: ./output_images/Detected_cars_2.png
[image6]: ./output_images/Detected_cars_3.png
[image7]: ./output_images/Detected_cars_4.png
[image8]: ./output_images/Detected_cars_5.png
[image9]: ./output_images/Detected_cars_6.png
[image10]: ./output_images/result.png
[video1]: ./out_video2.avi


---
This project is an implementation for vehicle detection and tracking in video. the following is explanation of the pipeline and the issues around the topic.

---
### Loading data
I started by reading in all the `vehicle` and `non-vehicle` images in function `loadData()`. the balance between number of vehicle and non-vehicle was considered. After, 4000 data sample from both 8000 car and 8000 non-car images were selected and processed for avoiding memory error. In next step from code below the features extraction for both data by calling function `extractFeature()`. 

![alt text][image1]
![alt text][image2]

```python
car_features=extractFeature(                     #extract car features
    sc, color_space='YCrCb', size=(32, 32))
#pdb.set_trace()
not_car_features=extractFeature(                 #extract not car features
    snc, color_space='YCrCb', size=(32, 32))
```

## Feature extraction `extractFeature()`:

The feature extraction starts with converting the image to `YCrCb` color space as after exploration this color space has good representation for grouping each channel value and it detects very well colors like black or white.

### Spatial binning `bin_spatial()`:
The color spatial feature was extracted by resizing the image to 16x16 , and this size is good for detection and for reducing the processing time. 

### Histogram features `HistofClolor()`:
Histogram features were extracted for the 3-channels and concatenated in variable `hist_features`

### Histogram of Oriented Gradients `get_hog_features()`

The code for this step is contained in the code cell 6 of the IPython notebook in function `get_hog_features()`. The HOG extraction processed on single gray image for reducing the computation. the following are the parameters of hog function and the result showed below.

```python
    orient = 9
    pix_per_cell = 8
    cell_per_block = 4
```
I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). The issue in this step is compromising between extracting good features and computation. After exploration the gray scale was chosen and `hog()` parameter were tuned to produce a general shape for cars and the parameters above produced the following result. 
![alt text][image3]
Higher parameter will produce big blocks of Hog features which car shape can be generalized to other shape. lower parameters doesn't produce much better results but it increase computation.

#### Classifier SVM

This part implemented in  cell [9] starts at line `13`. I trained a linear SVM using same number of combined `car` and `non-car` data. the data are scaled with `StandardScaler(copy = False)`  and produced `scaled_X = X_scaler.transform(X)`. the classifier parameter `kernel` used `linear` and `c= 100` to produce accuracy `SVC =  0.977`

#### Sliding Window Search

Two different scale window were chosen after exploration smaller sizes and overlap. Since as number of windows increase and overlap scale, the running process increases up to `11s`. the following are the two scaled window chosen for this project based on detection accuracy and time computation.

```python
        windows=windowex(imageX,x_start_stop=[800, None], y_start_stop=[400, 550]  #extract windows 
                        , xy_window=(90, 90), xy_overlap=(0.8,0.8))

        hot_windows=searchwindow(imageX,windows)

        windows=windowex(imageX,x_start_stop=[650, None], y_start_stop=[400, 680]   #extract windows
                                , xy_window=(128, 128), xy_overlap=(0.2,0.2))

        hot_windows.extend(searchwindow(imageX,windows))

```

By using window size `(69,69)` instead of `(90,90)` and thersholding the heat mape By thresholding the `heatma`p to `8` it get better detiction but process time per frame `7s` as showed below.
![alt text][image10]

#### Removing false detection

The window search focused in particular region in frame by taking the following points for small window`(90,90)`,`(x_start_stop=[800, None], y_start_stop=[400, 550])` and for `(128,128)` ,`(x_start_stop=[650, None], y_start_stop=[400, 680])`.These points were chosen to reduce the false detection and com[computation time. Every positive detection was recorded in each frame of the video and from the positive detections I created a heatmap and then thresholded that map to identify vehicle positions by value `4`.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected and the following are the results for test images. 

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]


#### Video Implementation

Here's a [link to my video result](./out_video2.avi)

---

### Discussion
The main issue I encountered is the computation time as I achieved `3.2s` which is not suitable for real time. In `CarDetection_Contours.ipynb` in cell [17] I tried to implement another approach instead of window search as it is slow and theres other tracking approach such as template matching, camshift or optical flow for tracking detected object rather than detect the entire each frame. The idea is to subtract background and predict the the foreground segmented objects. since the camera is moving, subtraction global motion from local motion should be implemented but I didn't as it is wide topic. I tierd to do trick by subtracting the `image_test2.png` from the frame as this image doesn't has any cars so I could use it as a background. Then after filtering and by using `cv2.findcontours()` the moving objects thanks to the camera will be detected. But this approach failed because of other objects(contours) in thresholded are detected as a car.

Here's a [link to my video result](./out_video_c.avi)

I think for better performance subtracting background from moving camera should be investigated and by extracting the other moving object and run classifier on each one. Then for true objects(contours) it should be taken as template and track this templates across the frames' contours. for contours in sequenced frames that doesn't has template it should be predicted with the classifier. And the templates that didn't match with any contour should be removed.

Also for grouping closed heatmap, finction `findecont()` dilates the heatmap image and group the closed detected window in one object. Then use `cv2.findcontour()` and `cv2.rectcontour()` to obtain one window for all detected windows. This approach produced better window drawn but it is committed to use `scipy.ndimage.measurements.label()` as mentioned in rubric points.  

Note: this method taken from pedestrian detection and tracking from fixed camera which I implemented previously.