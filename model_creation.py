import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_generator = ImageDataGenerator(
    rotation_range=35,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

train_generator = image_generator.flow_from_directory("./melanoma_cancer_dataset/train/", target_size=(256, 256), class_mode="binary", batch_size=16)

test_generator = image_generator.flow_from_directory("./melanoma_cancer_dataset/test/", target_size=(256, 256), class_mode="binary", batch_size=16)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(2, 2), input_shape=(256, 256, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(256, 256, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

model.add(Conv2D(filters=32, kernel_size=(2, 2), input_shape=(256, 256, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))


model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))

model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy")

early_stop = EarlyStopping(monitor="val_loss", patience=25, verbose=3, mode="min")

results = model.fit(train_generator, validation_data=(test_generator), epochs=125, verbose=1, steps_per_epoch=100, validation_steps=12, callbacks=[early_stop])

model.save("melanoma_binary_classifier.keras")


#EVALUATING THE MODEL BELOW
'''
import os

benign_path = "./melanoma_cancer_dataset/test/benign/"
malignant_path = "./melanoma_cancer_dataset/test/malignant/"


benign_melanoma_test_predictions = []
for benign_melanoma in os.listdir(benign_path):
    benign_melanoma_image = image.load_img(f"{benign_path}/{benign_melanoma}", target_size=(256, 256))
    img_array = image.img_to_array(benign_melanoma_image)
    img_array = img_array.reshape(1, 256, 256, 3)
    img_array = img_array/img_array.max()
    
    single_prediction = (model.predict(img_array)>0.5).astype("int32")
    benign_melanoma_test_predictions.append(single_prediction[0][0])
    
malignant_melanoma_test_predictions = []
for malignant_melanoma in os.listdir(malignant_path):
    malignant_melanoma_image = image.load_img(f"{malignant_path}/{malignant_melanoma}", target_size=(256, 256))
    img_array = image.img_to_array(malignant_melanoma_image)
    img_array = img_array.reshape(1, 256, 256, 3)
    img_array = img_array/img_array.max()
    
    single_prediction = (model.predict(img_array)>0.5).astype("int32")
    malignant_melanoma_test_predictions.append(single_prediction[0][0])
    

false_predictions_benign = 0
for prediction in benign_melanoma_test_predictions:
    if prediction == 1:
        false_predictions_benign +=1

        
false_predictions_melanoma = 0
for prediction in malignant_melanoma_test_predictions:
    if prediction == 0:
        false_predictions_melanoma +=1


'''
#DIVIDE NUMBER OF FALSE PREDICTIONS BY LENGTH (500), so if we get 21 false predictions for benign melanoma,
#divide 21/500 to get 0.042, or a 96% success rate (SHOULD RECEIVE AROUND 90% ACCURACY COMBINING MALIGNANT AND BENIGN PREDICTION SUCCESS RATES)

