from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    BatchNormalization,
)

def build_model(input_shape, num_classes):
    model = Sequential([

        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(input_shape)
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
        ), 
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu",
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        GlobalAveragePooling2D(),

        Dense(
            units=128,
            activation="relu",
        ),

        Dropout(rate=0.5),

        Dense(
            units=num_classes,
            activation="softmax",#تحول مخرجات النموذج إلى احتمالات
        ),
    ])
    return model