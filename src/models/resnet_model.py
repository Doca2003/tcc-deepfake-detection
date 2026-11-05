import tensorflow as tf
from tensorflow.keras import layers, models

#config. A e B

def build_resnet(input_shape=(224,224,3), weights="imagenet"):
    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(tf.keras.applications.resnet50.preprocess_input)(inputs)

    base = tf.keras.applications.ResNet50(include_top=False, weights=weights)
    x=base(x)

    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name="resnet50_detector")
