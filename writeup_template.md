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


[//]: # (Image References)

[image1]: https://github.com/vivek09pathak/traffic-sign-classifier/blob/master/Histogram_Image.JPG
[image10]: https://github.com/vivek09pathak/traffic-sign-classifier/blob/master/Histogram%20valid%20data.JPG
[image9]: https://github.com/vivek09pathak/traffic-sign-classifier/blob/master/Histogram%20test%20data.JPG
[image2]: https://github.com/vivek09pathak/traffic-sign-classifier/blob/master/Data%20Exploration.JPG
[image4]: https://github.com/vivek09pathak/traffic-sign-classifier/blob/master/images/image_no_passing.png
[image5]: https://github.com/vivek09pathak/traffic-sign-classifier/blob/master/images/image_speed_50.png
[image6]: https://github.com/vivek09pathak/traffic-sign-classifier/blob/master/images/image_speed_80.png
[image7]: https://github.com/vivek09pathak/traffic-sign-classifier/blob/master/images/image_turn_left_ahead.png
[image8]: https://github.com/vivek09pathak/traffic-sign-classifier/blob/master/images/image_turn_right_ahead.png
[image11]: https://github.com/vivek09pathak/traffic-sign-classifier/blob/master/Augmented%20image.JPG
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/vivek09pathak/traffic-sign-classifier)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data set varies over the labels as histogram has been plotted for Train,Valid and Test data and random images with the labels given below

train data![alt text][image1] 

valid data![alt text][image9] 
  
test data ![alt text][image10]

random images ![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because my Images were in 3-channel which I converted the images into grayscale by use of numpy.mean() function for all data sets i.e. train,test and validation and saved them again in same parameters.

Here is an example of a traffic sign image before and after grayscaling and normalization.

Before 
![alt text][image2]

After  
![alt text][image11]


As a last step, I normalized the image data because to ensure that all the features are in same scale so that no particular features influence the neural network to do that I have taken data to zero mean and unit covariances. 



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray scale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  28x28x6 in, 14x14x6 out 				|
| Convolution 5x5	   | 2x2 stride,  14x14x6 in, 10x10x16 out						|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  10x10x16 in, 5x5x16 out 				|
| Convolution 5x5	   | 1x1 stride,   5x5x16 in, 1x1x420 out						|
| RELU					|												|
|	Fully connected					|				1x1x420 in, 420	out							|
|	Flatten					|												|
|	Fully connected					|				420 in, 120	out							|
| RELU					|												|
| DROPOUT					|							prob=0.5					|
|	Fully connected					|				120 in, 84	out							|
| RELU					|												|
|	Fully connected					|				84 in, 43	out							|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer to train my model to find minima of the cross entropy batch size i have given as 127 with number of epochs as 30 and other hyperparameter as mean=0 and standard deviation=0.1 with learning rate as 0.0005

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.935
* test set accuracy of 0.931

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture chosen was LENET as data used to train MINIST dataset is for german traffic sign and dataset for the project provided was same to MINIST dataset.
* What were some problems with the initial architecture?
The problems with the architecture was the dropout layer and the architecture had lesser covolution layer for to maintain a higher accuracy for the architecture
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
The architecture was adjusted with convolution layer added to after second convolution layer as hidden layer with 5x5x16x420 and extra relu function was added to it after the first fully connected layer the dropout layer was added to decrease the over fitting of the data
* Which parameters were tuned? How were they adjusted and why?
The Epochs size,Batch Size,Learning and Keep Probs were tuned.The batch size was 127 and epoch given as 30 with learning rate as 0.0005 and keep prob for Train set was 0.5 and valid was 1
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Droput layer was added because the overfitting with the dataset because the our dataset has biased towards certain traffic signs.Building upon lenet architecture it was observed that it could reach better accuracy by adding more layers as with LENET architecture a certain accuracy came after which it git maxed out.

If a well known architecture was chosen:
* What architecture was chosen?LENET
* Why did you believe it would be relevant to the traffic sign application?Because MNIST data and Traffic classifier data are have similar classifcation problems
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 Becuase of higher accuracy on Test data which shown to network for first time shows the increase in stability of data model of higher accuracy

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


