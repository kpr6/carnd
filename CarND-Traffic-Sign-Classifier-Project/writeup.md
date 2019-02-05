# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34899
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...
1. At least one image corresponding to all unique classes in the dataset has been plotted to get an understanding of images as well as labels

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

My preprocessing step includes 
1. normalising the images
2. conversion to grayscale images to reduce the no. of operations in the network


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32 	|
|		RELU			|												|
| 		Max pooling	   	| 2x2 stride,  outputs 5x5x32 					|
| Fully connected		| size 800x512 									|
|		RELU			|												|
|		dropout			|												|
| Fully connected		| size 512x256 									|
|		RELU			|												|
|		dropout			|												|
| Fully connected		| size 256x43 									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used below 
1. adam optimiser
2. batch size 128
3. number of epochs 20
4. LR 0.001
5. 12 filters of size 5x5x3 each for 1st conv layer

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 94.7
* validation set accuracy of 94.7
* test set accuracy of 93.8

Approach:
1. Initially LeNet was chosen as it was a solid start for image classification
2. It had low depth filters which wasn't enough to grab the characteristics of the image while training. So I increased it
3. I also added dropout layers for fully connected layers to avoid overfitting
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

8 new images can be found in the notebook among which most of em are wrongly classfied for some reason although the test accuracy is above 93. I think the reason is coz the training dataset wasn't as diverse.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| bicycle crossing 		| priority road   								| 
| pedestrians  			| Bumpy road 									|
| Road Work				| Road Work										|
| Narrow road	   		| Roundabout mandatory			 				|
| speed limit 50kmph	| Keep right         							|
| No overtaking			| speed limit 60 kmph  							|
| speed limit 70 kmph	| speed limit 20 kmph  							|
| Slippery road			| Slippery Road      							|


The model was able to correctly guess 2 of the 8 traffic signs, which gives an accuracy of 25%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

For the first image, 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .59         			| Keep right   									| 
| .36     				| Right-of-way at the next intersection			|
| .16					| Yield											|
| .15	      			| Children crossing					 			|
| .12				    | Beware of ice/snow      						|


For the second image,


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .51         			| Bumpy road   									| 
| .32     				| Traffic signals 								|
| .14					| Road work										|
| .0026	      			| Bicycles crossing								|
| .002				    | Road narrows on the right      				|

For the third image,

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .55         			| Speed limit (60km/h)   						| 
| .30     				| Children crossing 							|
| .1					| Priority road									|
| .017	      			| Roundabout mandatory							|
| .015				    | Keep left         							|

And so on ...
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



