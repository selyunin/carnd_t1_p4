## Advanced Lane Finding Project

### The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/pic_1.png "Camera calibration images"
[image2]: ./output_images/pic_2.png "Chessboard Corners"
[image3]: ./output_images/pic_3.png "Undistorted Images"
[image4]: ./output_images/pic_4.png "Undistorted Road Images"
[image5]: ./output_images/pic_5.png "Thresholded images"
[image6]: ./output_images/pic_6.png "Birds-eye-view"
[image7]: ./output_images/pic_7.png "Finding peaks"
[image8]: ./output_images/pic_8.png "Finding lanes"
[image9]: ./output_images/pic_9.png "Image pipeline results"
[video1]: ./project_video.mp4 "Video"

The code for the project: is presented in (1) Jupyter notebook file 
[`Advanced Lane Finding
Project.ipynb`](https://github.com/selyunin/carnd_t1_p4/blob/master/Advanced%20Lane%20Finding%20Project.ipynb) 
(with the corresponding
[HTML](https://github.com/selyunin/carnd_t1_p4/blob/master/Advanced%2BLane%2BFinding%2BProject.html))
and as (1) a standalone command-line project, which include the files:
* [Camera.py](https://github.com/selyunin/carnd_t1_p4/blob/master/Camera.py)
* [ImageHandler.py](https://github.com/selyunin/carnd_t1_p4/blob/master/ImageHandler.py)
* [Line.py](https://github.com/selyunin/carnd_t1_p4/blob/master/Line.py)
* [VideoHandler.py](https://github.com/selyunin/carnd_t1_p4/blob/master/VideoHandler.py)
* [video_processing_system.py](https://github.com/selyunin/carnd_t1_p4/blob/master/video_processing_system.py)

Below, I will go over rubric points, which are the easiest to observe in the
Jupyter notebook.


### Camera Calibration

#### 1. 

First, we use OpenCV to `findChessboardCorners` and identify
corresponding `objpoints` (ideal coordinates of a chessboard plane in
a 3D space, e.g. `[2., 0., 0.]`, `z`-coordinate always `0`) and
`imgpoints` (real 2D-pixel coordinates of chessboard corners on a
photo, e.g. `[265.0, 631.7]`). We then use  `calibrateCamera` to find
camera matrix and distortion coefficients.

The code for this step is contained in the section 1 (cells 1-10) 
of the Jupyter notebook.

We use the following images to calibrate the camera: 

![Camera calibration images 1][image1]

We then search for chessboard corners and visualize results below. 
Chessboard corners are found not in every image. 
If not found, we display a black square.

![Camera calibration images 2][image2]

In the next step we find camera matrix and distortion coefficients and undistort
camera calibration images:

![Camera calibration images 3][image3]

We define `Camera` class and save the results of camera calibration in
the pickle file.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate distortion correction on the road images, I read every test image and undistort it, applying parameters found in the previous step, the result is shown below:

![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Given various colors of the lane lines, various lightning condition,
we apply Sobel and colorspace transform and thresholding to identify
lane lines on different roads.
We first compute (i) absolute value of Sobel filter in `x` and `y`
direction; (ii) gradient magnitude of the Sobel filter (l2-norm);
(iii) directional threshold where we take `arctan` `y` / `x` for
directional Sobel filters. We then combine all these thresholds in
`threshold_image` method. We also use HLS colorspace and find a color
threshold in `hls_threshold`. Color thresholds and Sobel thresholds
are combined in the `apply_thresholds` method, which takes an
undistorted image and returnes a thresholded binary image.

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The next step is to perform perspective transformation, and obtain the
so-called "bird's-eye-view" of the road. We identify source and
destinations polyhedra for image transformation. We use straight line
image `straight_lines1.jpg` to find a polyhedron and transform it to
rectangular. Method `vertices_img_pipeline` returns source and
destination vertices. Method `region_of_interest` (from the first
project) zeros out the pixels that are outside of the road segment.


![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Given a bird's-eye-view binary image of the road, we now identify
bottom positions of the lines and detect pixels that belong to the
lanes. We use `gaussian_filter1d` to smoothen the pixel distributions,
and `find_peaks_cwt` to identify maximum probable locations of the
lane in an image.

![alt text][image7]

We then define a method `get_lane`, which performs initial scanning and
detection of the lane lines. When we have initial polynomial
coefficients detected, we use `get_successive_poly_fit` to perform
search in the region of previously detected polynomial. We then fit
second order polynomial to our lane line in a method
`get_lane_poly_fit`.

Finally, we visualize lane detection: we color pixels of the left lane in
red, right lane in blue, and the corresponding fitted polynomial in
yellow. 

![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Given polynomial fits for each line, we detect radius of curvature
(`radius_of_curvature`) and lane offset (`get_lane_offset`).
We use implement formulae presented in the project description.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The method `pipeline_v_0_3` takes original image and performs all the steps of
the image pipeline:
1. Undistort an image
2. Apply thresholds (colorspace and Sobel)
3. Zero out pixels that are outside of the predefined mask of the road
4. Change perspective
5. Detect right and left lanes
6. Fit polynomials for right and left lanes
7. Calculate curvature an offset
8. Project result on undistored image
9. Put the text messages

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The results of the video pipeline can be seen in the [repo](./project_video_out.mp4) or on
[youtube](https://www.youtube.com/watch?v=gvNrxtdHSds&feature=youtu.be).


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline performs relatively well and robust on the project video. 
To make it more efficient, we can improve detecting lanes algorithm and probably
drop processing of some frames. The method with a convolution did not
perform well on the road images, but it is interested to explore this 
possibility further and tweaking convolution to detecting the lanes.
Another possibility to explore, is to use wavelet transform on a
binary images to detect the lane lines.
