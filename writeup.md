## Advanced Lane Finding Project Writeup

### 2017/12/31.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_img/undistorted.png "Undistorted"
[image2]: ./test_images/test5.jpg "Road Transformed"
[image3]: ./writeup_img/binary_combo.png "Binary Output"
[image4]: ./writeup_img/trans_after.png "Warp Image"
[image5]: ./writeup_img/searching.png "Fit Visual"
[image6]: ./writeup_img/one_output.png "Output"
[image7]: ./writeup_img/trans_before.png "Origin Straight"
[image8]: ./writeup_img/skip.png "Skip Binary"
[video1]: ./output_videos/normal.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 3 code cell of the IPython notebook located in "./line.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (cell 4 in `line.ipynb` and a function called "final_filter()").  Here's an example of my output for this step. 

For yellow line, I created a Lab color space filter with threshold 170 - 255. For white line, I created a LUV color space filter with threshold 215 - 255. Then I combined these two filters using 'or' operation and then combined gradient filter using 'and'.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `p_trans()`, which appears in cell 9 in `line.ipynb` The `p_trans()` function takes as inputs an image (`img`) and M matrix(`M`), `M` was calulated by a function called `cv2.getPerspectiveTransform`. it takes source (`src`) and destination (`dst`) points as input.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[261, 680], [1041,680], [752, 492], [533, 492]])
dst = np.float32([[350, 680], [900,680],[900, 492], [350, 492]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 261, 680      | 350, 680      | 
| 1041, 680     | 900, 680      |
| 752, 492      | 900, 492      |
| 533, 492      | 350, 492      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Origin][image7]
![Warped][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I sliding window and fit my lane lines with a 2nd order polynomial like this:

![alt text][image5]

And tried skip binary search with a margin of 100. Visualization like this.

![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in cell 15, 16 - measuring curvature in my code in `line.ipynb` for curvature, and a function called `cal_offset` in cell 19 which wrapped all the code into functions.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 17 - Drawing in `line.ipynb` including a function called `cv2.fillPoly()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

As I stated before, I wrapped all the code into functions and use them for my video pipeline called `process_video` in `line.ipynb`

the pipeline contains the 6 steps same as processing 1 single image. I also added 3 sanity check here to check is the line detection is correct or not.

check 1: check 2 lines have roughly same curvature

```python
def check_curvature(left_curverad, right_curverad, thresh = 300):
    if (abs(left_curverad - right_curverad) < thresh):
        return True
    else:
        return False
```

check 2: check 2 lines are seperate by roughly same distance

```python
def check_dist(left_fitx, right_fitx, ploty, thresh=(2.5, 3.5), percent=0.8):
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    total = len(ploty)
    valid = 0
    #print(left_fitx[0]*xm_per_pix, right_fitx[0]*xm_per_pix)
    for x1, x2, y in zip(left_fitx, right_fitx, ploty):
        dist = xm_per_pix*(x2 - x1)
        if (dist > thresh[0] and dist < thresh[1]):
            valid += 1
    if valid/total > percent:
        return True
    else:
        return False
```

check 3: check 2 lines are roughly parallel

```python
# stol: 2nd order parameter tolerance, ftol: 1st order parameter tolerance, 
def check_parallel(left_fitx, right_fitx, ploty, stol=1e-03, ftol=0.5):
    left_fit = np.polyfit(ploty, left_fitx, 2)
    right_fit = np.polyfit(ploty, right_fitx, 2)
    second_order = np.isclose(left_fit[0], right_fit[0], rtol=stol, atol=stol)
    first_order = np.isclose(left_fit[1], right_fit[1], rtol=ftol, atol=ftol)
    return second_order and first_order
```

If my sanity checks reveal that the lane lines detected are problematic, I assume it was a bad or difficult frame of video, then retain the previous positions from the frame prior and step to the next frame to search again. If you lose the lines for 3 frames in a row, I search from scratch using a histogram and sliding window.


I also used a class called `Line()` to track all parameters and smoothing the drawing. And here it is.

```python
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #if it is the first frame
        self.first_frame = True 
        self.max_len = 3
        self.bad_counter = 0
    def empty(self):
        while self.recent_xfitted:
            self.recent_xfitted.pop()
        return self.recent_xfitted
    def put(self, item):
        self.recent_xfitted.insert(0, item)
    def get(self):
        self.recent_xfitted.pop()
    def size(self):
        return len(self.recent_xfitted)
    def exceed_size(self):
        return len(self.recent_xfitted) > self.max_len
    def avg(self):
        return int(np.mean(self.recent_xfitted))
    def isEmpty(self):
        if not self.recent_xfitted:
            return True
        else:
            return False
    def save_offset(self, offset):
        self.line_base_pos = offset
    def line_detected(self):
        self.detected = True
    def not_detected(self):
        self.detected = False
    def set_curve(self, curve):
        self.radius_of_curvature = curve
    def update_best_fit(self):
        self.best_fit = np.mean(self.recent_xfitted, axis=0)
    def update_bestx(self, x):
        self.bestx = x
```



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline works well on the normal video but does not work on the challenge video and harder challenge video. It will fall when there is something very dark on the road. I problably should create an L threshold to illiminate that.

#### 2. Future work

Continue working on challenge video and harder challenge video.
