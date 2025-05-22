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
DATA_DIR = os.path.join(BASE_DIR, 'data/processed/mini-DDSM')
TRAIN_CSV = os.path.join(DATA_DIR, 'splits/train.csv')
VAL_CSV = os.path.join(DATA_DIR, 'splits/val.csv')
TEST_CSV = os.path.join(DATA_DIR, 'splits/test.csv')
IMG_SIZE = 224  # Standard size for most CNN architectures

def load_data(csv_path):
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} samples in {csv_path}")
    print("Available columns:", df.columns.tolist())
    
    # Map the actual column names to expected names
    df = df.rename(columns={'filepath': 'image_path', 'class': 'label'})
    
    # Map text labels to numeric values
    label_map = {'normal': 0, 'benign': 1, 'malignant': 2}
    df['label'] = df['label'].map(label_map)
    
    # Prepend the full path to the image paths and fix path separators
    df['image_path'] = df['image_path'].apply(lambda x: os.path.join(DATA_DIR, x.replace('\\', '/')))
    
    return df['image_path'].values, df['label'].values

def preprocess_image(image_path):
    try:
        # Read image
        img = tf.io.read_file(image_path)
        # Decode PNG format
        img = tf.image.decode_png(img, channels=3)
        # Resize
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        # Normalize
        img = img / 255.0
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return tf.zeros([IMG_SIZE, IMG_SIZE, 3])

def create_dataset(image_paths, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: (preprocess_image(x), y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_model():
    model = models.Sequential([
        layers.Conv2D(64, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')  # 3 classes: normal, benign, malignant
    ])
    return model

def train_model():
    print("Starting model training...")
    
    # Load data
    print("\nLoading datasets...")
    train_paths, train_labels = load_data(TRAIN_CSV)
    val_paths, val_labels = load_data(VAL_CSV)
    test_paths, test_labels = load_data(TEST_CSV)
    
    print("\nCreating data generators...")
    # Create datasets
    train_dataset = create_dataset(train_paths, train_labels)
    val_dataset = create_dataset(val_paths, val_labels)
    test_dataset = create_dataset(test_paths, test_labels)
    
    print("\nBuilding model...")
    # Build and compile model
    model = build_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Create models directory if it doesn't exist
    model_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    print("\nStarting training...")
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
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(model_dir, 'mini_ddsm_model_best.h5'),
                save_best_only=True,
                monitor='val_accuracy'
            )
        ],
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Save final model
    model.save(os.path.join(model_dir, 'mini_ddsm_model_final.h5'))
    
    return model, history

if __name__ == "__main__":
    train_model()