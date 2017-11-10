# Road-detection-and-tracking
The project is an implementation of the paper ‘Efficient Road detection and tracking’ by Zhou et al. The algorithm has been implemented in C++ using the OpenCV library setup in CodeBlocks developement environment.

### Prerequisites

For running the project, OpenCV needs to be setup. There are many blogs and resources available for setting up OpenCV either in Windows or Linux OS. I personally prefer Linux as everything is much easier to set up.

## General Methodology

### Road detection Methodology

The algorithmic flow for Road detection can be described as follows:


![alt text](grabCut/screenshots/detection.png)

### Road detection Demo

Step 1: Get an Initial Road segmentation by selecting a ROI. The segmentation method used is the GrabCut segmentation


![alt text](grabCut/screenshots/roi.png)

After selecting the ROI we get an initial segmentation. Note that the segmentation is not proper.


![alt text](grabCut/screenshots/init_seg.png)


Step 2: Mask some pixels as background to get a proper segmentation


![alt text](grabCut/screenshots/mask.png)

After masking the initial segmentation we get a refined segmentation


![alt text](grabCut/screenshots/masked_seg.png)

Step 4: Apply post processing steps like Connected Component Analysis, Hole filling and Otsu’s Thresholding to get the final image segmentation


![alt text](grabCut/screenshots/cc.png "Connected Component Analysis of the segmented road region\n")


![alt text](grabCut/screenshots/final_seg.png "Final road segmentation after post processing\n")

### Road tracking methodology

The algorithmic flow for Road detection can be described as follows:


![alt text](grabCut/screenshots/tracking.png "Road tracking methodology")

### Road tracking demo:

Step 1: Locate the FAST features in two subsequent frames.Compute the features in one frame and track them in the next frame using Optical Flow or the KL tracker(This is used to compute the Homography matrix between those two frames using the RANSAC method)

Fast features in the present frame
![alt text](grabCut/screenshots/fast.png)


![alt text](grabCut/screenshots/fast_predicted.png "Features in the next frame using the Optical flow method")

Step 2: Track the roads in the previous frame to the new frame-starting from the final road segmentation-by projecting the detected road pixels in the first frame to the next frame.


![alt text](grabCut/screenshots/tracked_road.png "Projected road in the present and the next kth frame using Homography
transformation")


Step 3: Detect the new road in the present frame(using skeleton analysis and center of skeleton methodolgies. Implementation in progress!!!).

## Conclusion
Refer to the paper for more details into the algorithm
