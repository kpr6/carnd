import os, csv
import sklearn
import numpy as np
from scipy import ndimage
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Dense, Conv2D, MaxPooling2D, Flatten, Dropout

samples = []
with open('/opt/carnd_p3/sim_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    skip = True
    for line in reader:
        if skip == True:
            skip = False
            continue
        # Dropping steering angles as they're skewing the data
        if float(line[3]) > 0.05 or float(line[3]) < -0.05:
            samples.append(line)

# print(len(samples))
# name = '/opt/carnd_p3/data/'+samples[0][0]
# center_image = ndimage.imread(name)
# print(center_image.shape)

from sklearn.model_selection import train_test_split
training_samples, validation_samples = train_test_split(samples, test_size = 0.2)

def generator(samples, batch_size=64):
    num_samples = len(samples)
    while 1:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images, angles = [], []
            for batch_sample in batch_samples:
                name = batch_sample[0]
#                 name = '/opt/carnd_p3/data/'+batch_sample[0]
                center_image = ndimage.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# compile and train the model using the generator function
train_generator = generator(training_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

model = Sequential()
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320, 3)))
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(80, 320, 3), output_shape=(80, 320, 3)))
model.add(Conv2D(6, (5,5), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (5,5), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(84, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
# model.add(Dense(1024, activation = 'relu'))
# model.add(Dropout(0.6))
# model.add(Dense(512, activation = 'relu'))
# model.add(Dropout(0.6))
# model.add(Dense(128, activation = 'relu'))
# model.add(Dropout(0.6))
# model.add(Dense(1))
print(model.summary())

# freeze_flag = True  # `True` to freeze layers, `False` for full training
# weights_flag = 'imagenet' # 'imagenet' or None
# preprocess_flag = True # Should be true for ImageNet pre-trained typically

# # Loads in InceptionV3
# from keras.applications.inception_v3 import InceptionV3

# # Using Inception with ImageNet pre-trained weights
# inception = InceptionV3(weights=weights_flag, include_top=False,
#                         input_shape=(139,320,3))
# if freeze_flag == True:
#     ## TODO: Iterate through the layers of the Inception model
#     ##       loaded above and set all of them to have trainable = False
#     for layer in inception.layers:
#         layer.trainable = False

# print(inception.summary())

# model.add(inception)
# model.add(GlobalAveragePooling2D())
# model.add(Dense(512, activation = 'relu'))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(training_samples), 
        validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=2, verbose=1)
model.save('model.h5')