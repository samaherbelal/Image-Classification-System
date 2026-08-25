import tensorflow as tf

#DATASET_PATH = "C:/Users/USER/jupyter project/ML/Deep Learning/Image Classification System/Military vehicles"

Train_Path = "C:/Users/USER/jupyter project/ML/Deep Learning/Image Classification System/Military vehicles/train"
Test_Path = "C:/Users/USER/jupyter project/ML/Deep Learning/Image Classification System/Military vehicles/test"
Validation_Path = "C:/Users/USER/jupyter project/ML/Deep Learning/Image Classification System/Military vehicles/validation"

IMAGE_SIZE = (224,224)# الحجم الموحد الذي سيتم تحويل جميع الصور إليه
BATCH_SIZE = 32
SEED = 42 # لتحكم في العشوائية

#دالة عرض معلومات
def print_dataset_summary(dataset, class_names):
    print("Classes names: ", class_names)
    print("Number of classes: ", len(class_names))

#طبقة لتحويل قيم البكسل الى نطاق من 0 الى 1
def normalize_images(images, labels):
    normalization_layer = tf.keras.layers.Rescaling(scale=1.0 / 255.0)
    images = normalization_layer(images) #على الصور Normalization نطبق عملية
    return images, labels

#بدل أن يرى النموذج الصور نفسها دائما، نقوم بعمل تغييرات بسيطة عليها وقت التدريب
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"), #يقلب الصورة أفقيا بشكل عشوائي
    tf.keras.layers.RandomRotation(factor=0.1), #تعمل دورانا عشوائيا بسيطا للصورة.
    tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1), #تعمل زووم عشوائي للصورة.
])

def augment_images(images, labels):
    images = data_augmentation(images, training=True)#من ان يبين لطبقات أننا في مرحلة التدريب
    return images, labels

def prepare_datasets():

    # tf.keras.utils.image_dataset_from_directory()
    #وظيفة هذه الدالة تقرأ الصور من المجلدات وتعرف عدد الفئات من اسماء المجلدات
    #وتحول الصور الى بيانات تستطيع الخوارزمية التعامل معها 
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=Train_Path,
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle = True
    )

    class_names = train_dataset.class_names #تحفظ اسماء مجلدات الصور


    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=Validation_Path,
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle = False #لأننا لا نحتاج خلط بيانات التقييم
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=Test_Path,
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle = False 
    )

    train_data = train_dataset.map(normalize_images)
    validation_data = validation_dataset.map(normalize_images)
    test_data = test_dataset.map(normalize_images)


    train_data = train_data.map(augment_images)

    
    print_dataset_summary(train_data, class_names)

    return train_data, validation_data, test_data, class_names