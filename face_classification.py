from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Model
from utils import read_config
import numpy as np
import os

# Read configs
config = read_config()

# Define constant variables
IMG_HEIGHT, IMG_WIDTH = config["img_height"], config["img_width"]
SAVE_MODEL_PATH = config["save_model_path"]
NUM_CLASSES = config["num_classes"]
TRAIN_PATH = config["train_path"]
BATCH_SIZE = config["batch_size"]
EPOCHS = config["epochs"]

# Create dataset with help of keras dataset creator
train_ds = image_dataset_from_directory(
    TRAIN_PATH,  # directory to data
    seed=11,  # seed for randomizing
    color_mode="rgb",  # save all images as with rgb channels
    interpolation="nearest",  # how to resize images
    image_size=(IMG_HEIGHT, IMG_WIDTH),  # size of images
    label_mode='binary',  # how to set labels, binary is 0 and 1
    batch_size=BATCH_SIZE,  # size of the batch size
    shuffle=True  # shuffle data
)

# Define our base model which is small mobilenet with imagenet weights.
# in this model we remove last layer of model to add customize linear
# layer which can classify the images custom faces.
base_model = MobileNetV3Small(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
)

# Freeze all layers in mobilenet because of the computational cost
for layer in base_model.layers:
    layer.trainable = False

# Add customize linear layer at the top of the mobilenet
# First use global average pooling for flattening output features
global_avg_pooling = GlobalAveragePooling2D()(base_model.output)
# Add linear layer with 1 output as 0 or 1 with sigmoid activation
output = Dense(NUM_CLASSES, activation='sigmoid')(global_avg_pooling)

# Define our final model
model = Model(inputs=base_model.input,
              outputs=output,
              name='MobileNet')

# Choose binary cross entropy as loss function because output is
# binary and choose adam as optimizere.
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )

# Train model for a few epochs
history = model.fit(train_ds, epochs=EPOCHS)

# save model
model.save(SAVE_MODEL_PATH)
