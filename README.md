# Introduction

As a baseline for this project I took the pipeline from the previous Advanced Lane Finding project as it already included some of the processing, line image distortion and the end result would be to merge both of those functions in single video.
As for the vehicle detection pipeline, the classes were quite detailed and worked well. Doing all the individual chapters allowed to take that code and insert it in the new pipeline. Most of playing with parameters was done already in the class stage and did not require much additional effort.
 
Idea was to follow the class recommendations on using HOG, however already initially I planned to have Neural network classifier replacing HOG as it should perform better, because HOG uses parameters tuned by hand, but Neural network architectures could learn which features are more useful and optimize parameters.
Due to this there are two pipelines, one with HOG and Sliding windows and the second one using Neural Network.
HOG detection was used to tune pipeline and to create better heat maps, because it gave more false positives (due to not being parameter optimized), but gave good overall pipeline. Moreover, HOG was abandoned, because its processing performance was quite slow. Even with approach taken from updated lectures with whole image HOG calculation, it was still performing at least 5x worse than YOLO classifier.

[//]: # (Image References)
[image1]: ./output_images/sliding_windows_test5.jpg "All overlapping sliding windows with different sizes displayed"
[image2]: ./output_images/sliding_windows_single_test5.jpg "Example windows for the size and top vertical position"
[image3]: ./output_images/hot_windows_test5.jpg "Windows with positive detections"
[image4]: ./output_images/frame_heat_map_test5.jpg "Binary heatmap for one frame"

 
## Histogram of Oriented Gradients (HOG)
Training image classification is done in the UdacityClass_classifier notebook. Function definitions (as in course) are in cell 1 and the actual process is in the cell 2.
Colorspace: Training is done in the YUV color-space, which performed better than RGB and marginally better than HSV in my tests. 
HOG Orientations: During classes I tried to minimize number of orientations (less features) without losing accuracy. If it was reduced below 8, then accuracy started to drop, 8 or more orientations did not give significant changes in accuracy.
HOG Color channels: All channels were used, because if some channels were omitted, then there was loss of accuracy.
Also both spatial (image resized to less number of pixels 16x16 and flattened) and color histogram with 16 bins were used.
 
Before passing to classifier, the features has been normalized by using StandardScaler from sklearn.preprocessing.
As for the classifier, the SVM was used, because SVM works well when there are less number of classes and in our case those are only 2 (cars and non-cars). Decision Trees would be more prone to overfitting on specific features, but we do not features that are clearly different, so Decision trees might not be best solution. Bayes classifier would work best if there would be more features and they would not be so dependant on each other, but for example car shape is quite distinct set of features.

After classifier has been trained, it is saved to the file in order to use it for the video pipeline.
 

## Sliding Window Search
I have implemented sliding window search in the Project5.ipynb notebook code cells: 
I used 3 sizes of windows:
Pixel size, line span, overlap (vert, hor)
96x96 pixels, y lines from 380 to 480, 50% overlap on X axis and 75% on Y axis
128x128 pixels, y lines from 400 to 560, 50% overlap on X axis and 75% on Y axis
164x164 pixels, y lines from 440 to image height, 50% overlap on X axis and 75% on Y axis
![alt text][image1]
Lines were chosen by looking at the different images and approximately framing the car at various distances.
![alt text][image2]
 
## Improvements to the classifier
After implementing first version of the classifier, it has been updated with following items:
There were too few positive detections, overlap was increased from TODO to TODO percentage.
Performance was very slow, whole image HOG processing was implemented, but as HOG is not scale invariant it required multiple calculations for different sliding windows, that did not provide significant performance gains.
Mainly due to the 2nd item, it was decided to implement heatmaps using this sub-optimal parameters and performance and afterwards change this to different classifier (Neural networks).
Benefit of using sub-optimal detection parameters is that better tuning can be done for heatmap processing, because if there are only few false positives or non-detected frames, then it is not enough data to clean-up with heatmaps.
![alt text][image3]
 
## Heatmap false positive removal
Code is located in cell 12 CarHeat class.
From the all bounding boxes within frame, one heatmap is calculated, which is added to the heatmap stack. Single frame heatmap is binary, so that there are either detected on not-detected pixels.
![alt text][image4]
Output is created summing last 10 heatmaps and thresholding it to at least 5 detections.
That way in each frame bounding boxes are combined. And if there are false positives in less than 5 frames, then those detections are removed.
For the frames above threshold, the scipy.ndimage label method is launched to label the windows and determine their bounding boxes.

Non optimized HOG based test video is located here.
Note, this was not optimized on purpose, as the processing times was around 1 frame par second and was looking for better performing alternatives.
[HOG classifier not optimized](https://youtu.be/OyFpzPOSr4E)
 
## YOLO classifier
Due to slow performance I checked the alternatives for the vehicle detection and one of the currently best performing algorithms is YOLO.
While looking for YOLO python usage, I found that member of Udacity previous cohort has done great job transferring and describing YOLO tiny model for our purpose.
I used the implementation “https://github.com/xslittlegrass/CarND-Vehicle-Detection” to test how well the YOLO architecture could perform for such task.
YOLO detects center pixels of the object as well as its height and width. It also predicts class probability and in this case if it is car class above threshold then it is treated as positive detection.
[YOLO detection only without aggregating](https://youtu.be/Mbuymi-Wd4g)
This give very good results, but detection frames are quite jitter'y and also there is one place with false positive detection.

Therefore, I did not feed YOLO results directly to output as it is done by “github.com/xslittlegrass”, but used it to feed heatmap calculation, essentially just replacing Sliding Windows search with HOG+Color detection and instead using YOLO implementation for that.
The predicted windows above threshold is then combined and threshold to give smoother output. It required smaller number of frames and lower threshold values then HOG detection.

Final submission video is using YOLO + Heatmap thresholding and it is here:
[YOLO detection with aggregation](https://youtu.be/YKQIL3Gi8Rs)

 
## Discussion

HOG performance: I was surprised that HOG calculation performance was really poor. This was quite limiting factor in terms of writing good code (long video processing times) and also it would be very limiting in real world applications, because then much better hardware would be required. Some ways to improve the performance would be:
*) Process HOG windows in parallel threads
*) Create separate server like process for HOG window processing (e.g. similar to behavior simulator) and scale it accordingly.

YOLO: Advances in AI has led to many improvements in terms of accuracy and performance, also as I tested YOLO implementation works very well and also is much better performing than OpenCV and search window based approach.

YOLO performance could be still improved as there are currently many sub-optimal and debugging information left inside the code. Also moving to C/C++ most of the pipeline (Keras is using TF backend which anyway uses C++/CUDA) should help with performance improvements.

## Summary:
Overall this was my favorite projects, because approaches that I learned during this project and related classes are applicable in many computer vision areas.
