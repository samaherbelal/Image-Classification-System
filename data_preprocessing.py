import tensorflow as tf
from pathlib import Path

DATASET_PATH = "C:/Users/USER/jupyter project/ML/Deep Learning/Image Classification System/Military vehicles"
Train_DIR = "C:/Users/USER/jupyter project/ML/Deep Learning/Image Classification System/Military vehicles/train"


IMG_size = (224,224)
BATCH_SIZE = 32


train_dataset = tf.keras.utils.image_dataset_from_directory(
    Train_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_size,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
num_classes = len(class_names)

print("Classes: ", class_names)
print("Number of classes: ", num_classes)

