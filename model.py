import csv
import cv2
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Lambda, Conv2D, Flatten, Dense
import math
import matplotlib.image as mpimg
import numpy as np
import os
from random import shuffle
import sklearn

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

def redistribute():
    # zero_rate shows how many times can be the number of zero data then the second higher bar
    zero_rate=10
    
    #I shuffle the sample data for the best distribution of dropping lines
    shuffle(samples)
    
    #We need some data format manipulation for the histogram
    ar_samples = np.array(samples)
    ar_steering = ar_samples[:,3]
    ar_steering = ar_steering.astype(np.float)

    # Now we are redi for preparing the histogram
    hist,bins = np.histogram(ar_steering,21)
    
    # We will need the second higher hist value for the redistribution
    hist.sort()
    bin_limit = hist[-2]
    zero_counter =0
    new_samples = []
    # I generate a new list by adding the images. 
    # If we reach the zero_rates we don't add any more zeroes to samples.
    for line in samples:

        if line[3]!='0':
            new_samples.append(line)
        else:
            zero_counter+=1
            if zero_counter<=bin_limit*zero_rate:
                new_samples.append(line)
    return new_samples

# Commenting the next line we can avoid the redistribution of data and work with the ori
samples = redistribute()

# I splitted the data ino train and validation splits. The rate is 0.3
# because in the generator I'll use more images
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.3)

# Let's create some augmentation functions. 
# These worked well in the last project
def persp_img(img):
    src = np.float32([[0,0],[319,0],[319,159],[0,159]])
    #src = np.float32([[0 + np.random.randint(4),0 + np.random.randint(4)],
    #       [31 - np.random.randint(4),0 + np.random.randint(4)],
    #       [31 - np.random.randint(4),31 - np.random.randint(4)],
    #       [0 + np.random.randint(4),31 - np.random.randint(4)]])
    dst = np.float32([[0 + np.random.randint(30),0 + np.random.randint(30)],
           [319 - np.random.randint(30),0 + np.random.randint(30)],
           [319 - np.random.randint(30),159 - np.random.randint(30)],
           [0 + np.random.randint(30),159 - np.random.randint(30)]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (320,160))
    return warped

def rotate_img(img):
    rows,cols = img.shape[:2]

    M = cv2.getRotationMatrix2D((cols/2,rows/2),5-np.random.randint(11),1)
    img = cv2.warpAffine(img,M,(cols,rows))
    return img

def sharpen_img(img):
    sh_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, sh_kernel) 
    return sharpened

def blur_img(img):
    kernel = np.array([[0.25,0.25], 
                   [0.25, 0.25,]])
    blured = cv2.filter2D(img, -1, kernel) 
    return blured

def shift_img(img):
    rows,cols = img.shape[:2]

    M = np.float32([[1,0,2-np.random.randint(5)],[0,1,2-np.random.randint(5)]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst


# Some augmentation
def augment_img(image):
    
    if np.random.randint(2)==1:
        image=persp_img(image)
    
    if np.random.randint(2)==1:
        image=rotate_img(image)
    
    if np.random.randint(2)==1:
        image=blur_img(image)
    
    if np.random.randint(2)==1:
        image=shift_img(image)
        
    if np.random.randint(2)==1:
        image=sharpen_img(image)   
   
    return image
    

def prepr_img(img):
    # I crop the top and bottom areas
    cropped_img = img[60:140,:]
    # I resize the image to the input of CNN
    resized_img = cv2.resize(cropped_img, (200,66), interpolation = cv2.INTER_AREA)
    return resized_img

def mirror_img(img):
    img = np.fliplr(img)
    return img

def generator(samples, batch_size=32, istraining=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                head, tail = os.path.split(batch_sample[0])
                fname = 'data\\IMG\\' + tail
                center_image = mpimg.imread(fname)
                # We need augmentation only in training method
                if istraining:
                    center_image = augment_img(center_image)
                center_image = prepr_img(center_image)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                
                # In case of training we will use the side cameras and 
                # add the mirrored images as well
                if istraining:
                    # Set the correction
                    correction = 0.18
                    
                    # Add left camera image to the dataset
                    head, tail = os.path.split(batch_sample[1])
                    fname = 'data\\IMG\\' + tail
                    left_image = mpimg.imread(fname)
                    left_image = prepr_img(left_image)
                    images.append(left_image)
                    angles.append(center_angle+correction)
                    
                    # Add right camera image to the dataset
                    head, tail = os.path.split(batch_sample[2])
                    fname = 'data\\IMG\\' + tail
                    right_image = mpimg.imread(fname)
                    right_image = prepr_img(right_image)
                    images.append(right_image)
                    angles.append(center_angle-correction)
                            
                    # Adding mirrored center images
                    center_image = mirror_img(center_image)
                    center_angle = - center_angle
                    images.append(center_image)
                    angles.append(center_angle)

            # Generate X_train and y_train
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size//4, istraining=True)
validation_generator = generator(validation_samples, batch_size=batch_size, istraining=False)

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66,200,3)))
model.add(Conv2D(24, (5,5), activation = "elu", strides=(2, 2)))
model.add(Conv2D(36, (5,5), activation = "elu", strides=(2, 2)))
model.add(Conv2D(48, (5,5), activation = "elu", strides=(2, 2)))
model.add(Conv2D(64, (3,3), activation = "elu" ))
model.add(Conv2D(64, (3,3), activation = "elu" ))
model.add(Flatten())
model.add(Dense(100, activation = "linear"))
model.add(Dense(50, activation = "linear"))
model.add(Dense(10, activation = "linear"))
model.add(Dense(1, activation = "linear"))

optimizer= Adam(lr=0.00005)
model.compile(loss="mse", optimizer = optimizer)

history = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)*4/batch_size), 
                    validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size),
                    epochs=30, verbose=1)
model_file="model.h5"
model.save(model_file)
print('Model saved as ',model_file)