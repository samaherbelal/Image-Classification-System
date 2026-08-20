import tensorflow as tf
from pathlib import Path

DATASET_PATH = "C:/Users/USER/jupyter project/ML/Deep Learning/Image Classification System/Military vehicles"

Train_Path = "C:/Users/USER/jupyter project/ML/Deep Learning/Image Classification System/Military vehicles/train"
Test_Path = "C:/Users/USER/jupyter project/ML/Deep Learning/Image Classification System/Military vehicles/test"
Validation_Path = "C:/Users/USER/jupyter project/ML/Deep Learning/Image Classification System/Military vehicles/validation"

IMG_size = (224,224)
BATCH_SIZE = 32
SEED = 42 # لتحكم في العشوائية

# tf.keras.utils.image_dataset_from_directory()
#وظيفة هذه الدالة تقرأ الصور من المجلدات وتعرف عدد الفئات من اسماء المجلدات
#وتحول الصور الى بيانات تستطيع الخوارزمية التعامل معها 

train_dataset = tf.keras.utils.image_dataset_from_directory(
    Train_Path,
    seed=SEED,
    image_size=IMG_size,
    batch_size=BATCH_SIZE,
    shuffle = True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    Validation_Path,
    seed=SEED,
    image_size=IMG_size,
    batch_size=BATCH_SIZE,
    shuffle = False #لأننا لا نحتاج خلط بيانات التقييم

)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    Test_Path,
    seed=SEED,
    image_size=IMG_size,
    batch_size=BATCH_SIZE,
    shuffle = False 
)

class_names = train_dataset.class_names

print("Classes: ", class_names)
print("Number of classes: ", len(class_names))

