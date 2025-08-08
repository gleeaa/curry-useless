import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    """Loads and augments data using TF's ImageDataGenerator."""
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='training'
    )
    
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )
    
    return train_generator, val_generator