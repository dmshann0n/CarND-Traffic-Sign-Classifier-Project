# **Traffic Sign Recognition**  

## Writeup Template

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

[image0]: ./examples/processed.png "Sample results"
[image1]: ./examples/dataset_explore.png "Exploration of datasets"
[image2]: ./examples/model.png "Model architecture"
[image3]: ./examples/processed.png "Processed results"
[image4]: ./web_samples/1.jpg "Sample: 30 km/h"
[image5]: ./web_samples/13.jpg "Sample: Yield"
[image6]: ./web_samples/14.jpg "Sample: Stop"
[image7]: ./web_samples/38.jpg "Sample: Stay right"
[image8]: ./web_samples/4.jpg "Sample: 70 km/h"
[image9]: ./web_samples/7.jpg "Sample: 100 km/h"
[image10]: ./examples/training_summary.png "Summary of training data"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/dmshann0n/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used Python (with some help from Numpy) to get some basic details of the training set.

* The size of training set is: 34,799
* The size of the validation set is: 4,410
* The size of test set is: 12,630
* The shape of a traffic sign image is: 32x32x3
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I showed samples from the dataset for each label (with the friendly names of the signs displayed). I also charted the frequency of each class in the various data sets.

![image1]
![image10]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For the original dataset (and for normalizing the other sets) I cropped the image to get rid of border noise and normalized using the CLAHE - histogram equalization. This was an attempt to equalize the contrast in the images. I was able to get to 89%-ish accuracy with this method. I tested grayscale and other normalization techniques. I didn't see improvement in my early tests, but given more time would play with other normalization techniques. 

Using the process outlined in the Pierre Sermanet and Yann LeCun paper, I also wrote a function (jitter_img) to introduce random changes of translation, rotation, and scale. Initially I generated 5 "jittered" images for each original in the training set. The final accuracy of the model depended heavily on the random inputs, so I experimented with generating new images per epoch and found that didn't create significantly different outcomes. Instead I settled on increasing the generated samples to 10x per original. (I didn't normalize the number of samples per class through this method, but that might be something to attempt.)

Here's how one image was modified and a random jitter in the "augmented" case:
![image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model followed the sample LeNet architecture very closesly with the addition of a dropout layer.

I'm posting the code here because I think that's easiest way to visualize this (and show how close it was to the LeNet starting point!) --

![image10]

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the LeNet sample from the previous lessons with the Adam optimizer, batch size of 128, 15 epochs, and a reduced learning rate of 0.0005.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.954 
* test set accuracy of 0.942

Model questions: 
* What architecture was chosen? LeNet

* Why did you believe it would be relevant to the traffic sign application? It was used in the handwriting sample and seemed to handle general computer vision challenges like this well.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? Given this well cleaned toy dataset, the model identified images with a high degree of accuracy. I wouldn't trust my self-driving car if it could only identify 94% of signs though. There are also plenty of real world factors that would impact this model -- inclement weather, lighting conditions, angle, identifying the outline of the sign correctly -- that would need to be figured out for real world applications.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are SIX German traffic signs that I found on the web:

(Also I learned that the American stop sign is somewhat standard internationally!)

![image4] ![image5] ![image6] 
![image7] ![image8] ![image9]

I selected these images from stock photos. I expected the "Keep Right" sign to cause the most issues because it is taken from a skewed perspective.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![image0]

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in cells 17 and 18 of my notebook.

For most of these samples, the model was predictive with a high level of certainty with high certainty on every image except the 100 km/h sign. On this sign, it wasn't able to differentiate between 80 km/h and 100 km/h. In other runs of the model I was able to get 100% accuracy for these samples depending on the random jitter added to the training dataset. Obviously this doesn't mean it would perform better in the real world.




