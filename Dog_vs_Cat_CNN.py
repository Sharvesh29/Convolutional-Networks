#Deep learning 
#convolutional neural network

#Creating our CNN

#IMPORTING OUR LIBRARIES

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#intializing the CNN

classifier = Sequential()

#Step Wise 
#Step1- Convolution

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation ='relu'))

#Step-2 Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))

### Inorder to Obtain better accuracy add another Convolution layer along with the pooling 
## input_shape function is optional in the @nd Convoution layer as we have already defined the shape in previous layer 
#But we can optimize the image pixel value from 64 to 128 or 256 inthe Convolution layer for better accuracy

# classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation ='relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step-3 Flatten

classifier.add(Flatten())

#Step-4 Full Connection

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling CNN and Optimizing our results using adam and loss via cross entropy an dour metrics inthe form of accuracy

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

##Fitting the CNN to the Images
#Importing ImageDataGenerator for Image Augumentation.

from keras.preprocessing.image import ImageDataGenerator

#Image Augmenation for TRAINING AND TEST SET

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#Imporing our train and test sets for CNN process

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size = (64, 64), batch_size=32, class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

#Fitting our Training set and Test set into CNN for Results

classifier.fit_generator(training_set, steps_per_epoch=8000, epochs=25, validation_data=test_set, validation_steps=2000)