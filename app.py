import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def build_model(num_classes=3):
    """Uses EfficientNetB0 (pretrained on ImageNet) for transfer learning."""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base layers
    base_model.trainable = False
    print("hello")
    # Add custom head
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=base_model.input, outputs=predictions)