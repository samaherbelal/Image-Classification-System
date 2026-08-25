from data_preprocessing import prepare_datasets, IMAGE_SIZE
from model import build_model

train_data, validation_data, test_data, class_names = prepare_datasets()

num_classes = len(class_names)

input_shape = IMAGE_SIZE + (3,)#3 = RGB

#نبني النموذج
model = build_model(input_shape=input_shape, num_classes=num_classes)

#تعلم النموذج كيف يتعلم
model.compile(
    optimizer="adam",#تحديث أوزان النموذج أثناء التدريب
    loss="sparse_categorical_crossentropy",#تقول كم كان النموذج مخطئا في توقعه
    metrics=["accuracy"],# للجودة
)

#تجيب ملخص لبنية النموذج
model.summary()

EPOCHS = 20 #مرور النموذج على كامل بيانات التدريب مرة واحدة

# تدريب النموذج
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
)

model.save("trained_model.keras")


print("Training complete. Model saved to 'trained_model.keras'.")
