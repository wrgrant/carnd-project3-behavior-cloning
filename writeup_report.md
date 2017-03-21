#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./images/before.jpg "Original image"
[image2]: ./images/after.jpg "After cropping"



---


My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results


Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

    python drive.py model.h5

<br>

###Model Architecture and Training Strategy


The overall strategy I ended up with for deriving a model architecture was to use the absolute simplest architecture I could manage while keeping the car on the road. I wanted to test the limit of a basic model to see if it could pass the test with appropriate training data.

My first step was actually to use a complicated convolution neural network model from the comma.ai [github site](https://github.com/commaai/research/blob/master/train_steering_model.py). I figured that the more complex the better, and that with a very deep neural network I should be able to just drive a lap or two around the track and have an easy time with the project. I did a full lap at the center of the lane and a 'recovery' lap, as the lab instructions suggested and I couldn't believe all that data and a big network didn't result in better performance. The car drove very smoothly, but would routinely drive straight off the track at corners. 

I modified drive.py and setup a pipeline such that calling one python file would build my model, move the model.h5 file to the correct place, startup the code in drive.py, and feed a unique directory based upon the timestamp, so I could quickly iterate on model ideas.

I really wanted to develop a feel for what it would take in terms of network complexity to get the car to drive successfully on the track, so I stripped it down to a single convolution and a ```Dense(1)``` layer. This proved interesting in that it would stay on the track for a tiny bit but then become confused and wander off. I added a small hidden fully-connected layer of 20 neurons and immediately performance got much better. The car would stay on the track, but it would do things like center the car over the yellow lines for a whole section of track. As if the model over-fit to be activated only by yellow lines. A dropout layer seemed to prevent this behavior.

At this point I took a look again at the training data. I came across some posts in the slack channel about actually using smaller amounts of data to train with and using less small-angle training data, and it dawned upon me that this made perfect sense in my situation. Since I was experimenting now with a minimalist model, I may as well experiment with minimalist training data! So I went back and trained with very short clips, only one of each type (yellow lines, bridge, striped lines, brown side for pit-area). Immediately the car almost made it around the whole track! Amazing!! With such limited training data and limited model architecture!

Thinking about it, a simple model is not capable of complex or deep analysis. It's like training a child to stay away from certain things. The child doesn't know why something is dangerous, nor does she need to understand the greater patterns of vision and analysis, just stay away from those things which I tell you to! And on the other hand, trying to explain the complexities of thermodynamics and joule heating of a stove element is just going to confuse her. She doesn't even understand words yet! But she can successfully copy certain behaviors like playing with simple toys. Thanks to observing my own daughter's development, I was starting to get a physical feel for the interplay between model complexity and training data--they have to be at appropriate levels to each other.

I then implemented some code that would skip some small-angle training data, as I didn't want my model to know much about 'cruising' down a lane, but rather, to simply avoid the side boundaries.

At this point, my model would successfully drive the car around the track. It's very rough, I admit, but that was my goal! It is amazing to me that such a simple network architecture can successfully navigate the car around the track. I do wonder what the lower limit of training data would be. There ended up being around 1000 images in my set (center, left, right, center_reverse), so that's really around 250 unique images. What would be the lower bound of training data to make a successful model? What if I only took one picture for each situation?  


<br>

#### Final Model Architecture

The final model architecture (model.py lines 65-74) consisted of a convolution neural network with the following layers:

1. A normalization lambda layer
1. Cropping
1. 5x5 convolutional layer with filter_depth=6 
1. RELU non-linear activation
1. MaxPooling with 2x2 for downsampling
1. Dropout of 20%
1. Flatten to 1D
1. Fully-connected with 20 neurons
1. Single-value output of steering angle


<br>

#### Creation of the Training Set

To augment my data set, I flipped images and angles. Without doing so, the car would favor left-ish angles. I also added normalization and cropping layers so that the training data concentrated on only the features I wanted the model to learn. I enabled random shuffling of the data set and split 20% of the data for validation. 


<br>

#####Example images:

Raw image from simulator:

![][image1]

Cropped image concentrating on just the road:

![][image2]
