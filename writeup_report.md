﻿# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./writeup/image1.png "Model architecture"
[image2]: ./writeup/image2.png "History without dropout"
[image3]: ./writeup/image3.png "History with dropout"
[image4]: ./writeup/image4.png "Original distribution"
[image5]: ./writeup/image5.png "Redistributed data"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing (on Windows Anaconda Prompt)

```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

In the instructions for the project I've read that the NVIDIA model is very effective in the field of behavior cloning. You can see the model architecture on the following figure:

![alt text][image1]
< source: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf>

First time I tried to use the `RELU` activation function. It works well on image classification, but it has a disadvantage in the regression calculation. It can't handle negative values. I tried the `ELU` and `linear` activation functions. Finally I used the `ELU` for the convolutional layers and `linear` for the fully connected layers. These activations gave much better results. 



#### 2. Attempts to reduce overfitting in the model

I used the same augmetation process as in the last project. On the basis some random numbers I used a perspective transformation, some rotation, some shifts, sharpening and blur. I augmented the original images before cropping and resizing them. The change of the borders was no as radical in this case.

I tried a model with a dropout layer and without that. You can see the training history in the following charts before and after the putting the dropout layer into the model. 
![alt text][image2]![alt text][image3]

The validation loss is significantly lower than training loss. There were some epochs where it was higher, but it doesn't show an overfitting. The dropout layer gave a higher loss, so I didn't used it.

This is not an absolute regular function, so it is impossible to reach 100 % accuracy. In my opinion a loss less than 2.5% is quite a good result.


#### 3. Model parameter tuning

During the training process I used the following parameters:

Optimizer: Adam

Learning rate: 0.0005

Batch size: 32

Epochs: 30

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I recorded some cases when I came back from the edge of the road to the middle. I recorded the data on the first track. I recorded one lap in each direction.

The model originally was created for (66,200,3) shape images. In the first step I  applied some augmentation on the images. After that I cropped the top and the bottom part of the image, finally I resized it to (66,200,3) shape.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy was using the NVIDIA model. Build a new approach can take weeks, and it's very hard to make a better model.

#### 2. Final Model Architecture

The final model was the original NVIDIA model. I didn't need any changes. The model was not overfitted, so I didn't need dropout layers. The only change was a Lambda layer, where I normalized the data.



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track using center lane driving. I recorded some cases when I came back from the edge of the road to the middle.

I recorded altogether 16 820 images. I started with central images. A generated the distribution of steering angles. The original distribution can be seen on the left side, the redistributed data on the right side.

![alt text][image4] ![alt text][image5]

To decrease the number of pictures with zero angles to the second highest histogram bar resulted a nervous behavior on the road. I needed a lot of zero angles images for a realistic behavior so I introduced a new parameter:

    zero_rate=10

With tis parameter I easily can change the amount of the pictures on the straight road. Te original value is more than 20. I used the value of 10.

Firstly I augmented all the images. I used the data generator which was introduced in the lesson. I had tu change the batch sizes because I generated 4 images inside the generator. I used the left and right images and the mirrored central image. The correction rate for steering was 0.18.

Finally I reached loss about 3,5 %. With the central data only it's possible to reach the loss about 2%, but the side images gives more features for the model. 

After the model was generated I had to change the drive.py file. It was necessary to reshape the images from the camera in the same way as it was before the training. The drive.py used different library than me in the training process. The two algorithms are the following.

`
def prepr_img(img):
​    cropped_img = img[60:140,:]
​    resized_img = cv2.resize(cropped_img, (200,66), interpolation = cv2.INTER_AREA)
​    return resized_img
`



`
def prepr_img(img):
    cropped_img = img.crop((0, 60, 319, 140))
    resized_img = cropped_img.resize((200,66))
    return resized_img
`

Finally a followed the instructions and made my video file.

