import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

# Parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
DATASET_PATH = "dataset_small/"
MODEL_PATH = "model/alz_model_basic.h5"
CLASS_NAMES = ['AD', 'CN', 'EMCI', 'LMCI']  # Based on your class folders

# --- TRAINING BLOCK ---
def train_model():
    # Data Generator
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Model
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='softmax')  # 4 classes
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

    # Save model
    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"✅ Model saved at: {MODEL_PATH}")

# --- PREDICTION FUNCTION ---
def predict_stage(img_path):
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found. Train the model first.")
        return None

    model = models.load_model(MODEL_PATH)

    try:
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

    prediction = model.predict(img_array)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    print(f"🧠 Predicted Stage: {CLASS_NAMES[predicted_class]} ({confidence*100:.2f}% confidence)")
    return predicted_class

# --- RUN ---
if __name__ == "__main__":
    # Uncomment below line to train first time
    train_model()

    # Predict from a test image
    test_img_path = "C:\\Users\\Mayur\\Desktop\\alzemers classification\\sample_image.jpg.jpg"  # Make sure this image exists in your folder
    predict_stage(test_img_path)
