Udacity self driving car ND - P! - Finding lanes
=======================================
Pipeline consists of 5 steps.

1. Used Gaussian blur on the image to smooth out noisy parts of he image
2. Converted the image to grayscale which acts as an input to canny transform to detect edges. The image is later masked outside region of interest
3. Now lines are detected by passing the output of the canny transform to hough transform for marking lanes
4. All the collected lines from the hough transform are being used to come up with a left line and right line by means of averaging of slopes of longest lines ang extrapolation
    - Eliminated almost horizontal lines i.e. slope < 0.5
    - Used top 5 longest lines for slope average calculation
5. Lastly, the image having lane markings are overlapped onto initial image