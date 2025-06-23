import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# === 1. Define paths ===
train_dir = 'dataset/chest_xray/train'
val_dir = 'dataset/chest_xray/val'
test_dir = 'dataset/chest_xray/test'

# === 2. Image size and batch ===
img_height, img_width = 150, 150
batch_size = 32

# === 3. Data generators ===
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary'
)

# === 4. Build model ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# === 5. Train model ===
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=5,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# === 6. Save model ===
#model.save('pneumonia_model.h5')
model.save("pneumonia_model", save_format="tf")  # ✅ saves folder format, not .h5

print("✅ Model saved as pneumonia_model.h5")

# === 7. Evaluate on test set ===
test_loss, test_acc = model.evaluate(test_generator)
print(f"📊 Test Accuracy: {test_acc:.4f}")
print(f"📉 Test Loss: {test_loss:.4f}")

# === 8. Plot accuracy and loss ===
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history.get('accuracy'), label='Train Accuracy')
plt.plot(history.history.get('val_accuracy'), label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history.get('loss'), label='Train Loss')
plt.plot(history.history.get('val_loss'), label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
