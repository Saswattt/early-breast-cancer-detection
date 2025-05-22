import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np
import os

# Set random seed for reproducibility
tf.random.set_seed(42)

# Data paths
BASE_DIR = '/Users/saswatkumarsahoo/Desktop/breast-cancer-detection'
DATA_DIR = os.path.join(BASE_DIR, 'data/processed/MIAS')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
VAL_CSV = os.path.join(DATA_DIR, 'val.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
IMG_SIZE = 224  # Standard size for most CNN architectures

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Prepend the full path to the image paths
    df['image_path'] = df['image_path'].apply(lambda x: os.path.join(DATA_DIR, x))
    return df['image_path'].values, df['label'].values

def preprocess_image(image_path):
    try:
        # Read image
        img = tf.io.read_file(image_path)
        # Try both PNG and PGM formats
        try:
            img = tf.image.decode_png(img, channels=3)
        except:
            img = tf.image.decode_image(img, channels=3)
        # Resize
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        # Normalize
        img = img / 255.0
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        # Return a blank image in case of error
        return tf.zeros([IMG_SIZE, IMG_SIZE, 3])

def create_dataset(image_paths, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: (preprocess_image(x), y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')  # 3 classes: normal, benign, malignant
    ])
    return model

def train_model():
    # Load data
    train_paths, train_labels = load_data(TRAIN_CSV)
    val_paths, val_labels = load_data(VAL_CSV)
    test_paths, test_labels = load_data(TEST_CSV)

    # Create datasets
    train_dataset = create_dataset(train_paths, train_labels)
    val_dataset = create_dataset(val_paths, val_labels)
    test_dataset = create_dataset(test_paths, test_labels)

    # Build and compile model
    model = build_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Create models directory if it doesn't exist
    model_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"\nTest accuracy: {test_accuracy:.4f}")

    # Save model
    model.save(os.path.join(model_dir, 'mias_model.h5'))

    return model, history

if __name__ == "__main__":
    model, history = train_model()