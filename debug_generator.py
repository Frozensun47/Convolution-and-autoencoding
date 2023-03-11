from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
input_dir = 'Images'
batch_size = 1

# Create a data generator for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # normalize pixel values to [0,1]
    shear_range=0.2,  # apply random shear augmentation
    zoom_range=0.2,  # apply random zoom augmentation
    horizontal_flip=True  # flip images horizontally
)

train_generator = train_datagen.flow_from_directory(
    input_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='input',
    shuffle=True
)
import os
print(os.getcwd())
# Print some information about the data generator
print("Number of training samples:", train_generator.samples)
print("Number of classes:", train_generator.num_classes)
print("Input shape:", train_generator.image_shape)

# Get the next batch of images and labels from the generator
x_batch,y_batch= next(train_generator)

# Get the first image from the batch
img = x_batch[0]

# Display the image
plt.imshow(img)
plt.show()

# Get the first image from the batch
img = y_batch[0]

# Display the image
plt.imshow(img)
plt.show()
