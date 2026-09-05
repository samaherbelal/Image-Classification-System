import os
import numpy as np
import tensorflow as tf

# استدعاء أبعاد الصور ودالة التجهيز من ملف
from data_preprocessing import IMAGE_SIZE, prepare_datasets



def load_trained_model(model_path= "trained_model.h5"):
    """
    هنا بنحمل موديل الـ CNN اللي دربناه وحفظناه قبل كدا[cite: 5]
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Please run train.py first."
        )

    print(f"--- Loading model from: {model_path} ---")
    model = tf.keras.models.load_model(model_path)
    return model


def preprocess_single_image(image_path):
    """
    بناخد صورة واحدة وبنضبطها بنفس الخطوات اللي عملناها وقت التدريب بالضبط[cite: 1]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # بنقرأ الصورة وبنخلي مقاسها (224, 224) زي ما الموديل متعود[cite: 1]
    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)

    # بنحول الصورة مصفوفة أرقام (Numpy Array)
    img_array = tf.keras.utils.img_to_array(img)

    #  بنقسم على 255 عشان نخلي قيم البكسلات من 0 لـ 1 (Normalization)[cite: 1]
    img_array = img_array / 255.0

    #  بنزود بعد جديد (Batch Dimension) عشان تبقى (1, 224, 224, 3) والموديل يقبلها
    img_batch = np.expand_dims(img_array, axis=0)

    return img_batch


def predict_image(image_path, model_path="trained_model.h5"):
    """
    الدالة الرئيسية اللي بتتوقع الفئة وبتحسب الموديل واثق من إجابته بنسبة كام[cite: 4]
    """

    # بنجيب أسماء الكلاسات من دالة تجهيز البيانات[cite: 1]
    _, _, _, class_names = prepare_datasets()

    # بنحمل الموديل ونجهز الصورة
    model = load_trained_model(model_path)
    processed_image = preprocess_single_image(image_path)

    # الموديل بيعمل التنبؤ هنا
    predictions = model.predict(processed_image, verbose=0)

    # بنجيب أعلى احتمال ونشوف الكلاس بتاعه ونحسب النسبة المئوية[cite: 4]
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100


    print("\n=================== Prediction Results ===================")
    print(f"Predicted Class : {predicted_class_name}")
    print(f"Confidence Level: {confidence:.2f}%")
    print("=========================================================\n")

    return predicted_class_name, confidence


if __name__ == "__main__":
    # حط هنا مسار أي صورة عايز تجرب عليها للتنبؤ
    test_image_path = r"C:\Users\USER\jupyter project\ML\Deep Learning\Image Classification System\Military vehicles\test\Armored personnel carriers\Armored personnel carriers_0_2.jpeg"

    # اختار الموديل الأصلي أو الموديل بعد الـ Fine-Tuning
    model_to_use = "trained_model.h5"  # أو "fine_tuned_model.keras"

    try:
        predict_image(test_image_path, model_path=model_to_use)
    except Exception as e:
        print(f"Error during prediction: {e}")