import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# علشان نربطه مع ملف البيانات 
from data_preprocessing import prepare_datasets

def evaluate_and_fine_tune():
    _, validation_data, test_data, class_names = prepare_datasets()

    # تحميل النموذج الذي تم حفظه عند تشغيل
    print("\n--- جاري تحميل النموذج المدرب (trained_model.keras) ---")
    model = tf.keras.models.load_model("trained_model.keras")

    # حساب التنبؤات على مجموعة الاختبار test_data
    print("\n--- جاري تقييم النموذج على مجموعة الاختبار (Test Set) ---")
    y_true = []
    y_pred_probs = []

    for images, labels in test_data:
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds)
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.argmax(np.array(y_pred_probs), axis=1)

    # 4. حساب استخراج Accuracy, Precision, Recall, و F1-Score
    print("\n=================== (Metrics Report) ===================")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # هنا نرسم ونحفظ مصفوفة الارتباك 
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\n Saved done: 'confusion_matrix.png'")

    # Fine-Tuning
    print("\n--- بدء مرحلة الضبط الدقيق (Fine-Tuning) ---")
    
    # خفض معدل التعلم لتدريب آمن بدون هدم الأوزان التي تعلمها النموذج
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    FINE_TUNE_EPOCHS = 5
    model.fit(
        validation_data,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=test_data
    )

    # حفظ النموذج المحسّن
    model.save("fine_tuned_model.keras")
    print("\nتم الانتهاء من Fine-Tuning وحفظ النموذج النهائي في 'fine_tuned_model.keras'.")

if __name__ == "__main__":
    evaluate_and_fine_tune()