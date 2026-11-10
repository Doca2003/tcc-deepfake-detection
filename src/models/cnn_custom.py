from tensorflow.keras import layers, models

#config. C

def build_custom_cnn(input_shape=(224,224,3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)

    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs=layers.Dense(1,activation="sigmoid")(x)
        return models.Model(inputs, outputs, name="custom_cnn")

