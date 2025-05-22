import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ..models.hybrid_model import build_hybrid_model

def train_mias_hybrid_model():
    # Set paths
    base_path = '/Users/saswatkumarsahoo/Desktop/breast-cancer-detection/data/processed/MIAS'
    train_csv = os.path.join(base_path, 'train.csv')
    val_csv = os.path.join(base_path, 'val.csv')
    test_csv = os.path.join(base_path, 'test.csv')

    # Load CSV files and convert labels to strings
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    # Convert labels to strings
    train_df['label'] = train_df['label'].astype(str)
    val_df['label'] = val_df['label'].astype(str)
    test_df['label'] = test_df['label'].astype(str)

    # Image data generator with augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation and test data generators (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Set image parameters
    img_height = 224
    img_width = 224
    batch_size = 32

    # Create data generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=base_path,
        x_col="image_path",
        y_col="label",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=base_path,
        x_col="image_path",
        y_col="label",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=base_path,
        x_col="image_path",
        y_col="label",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Build the hybrid model
    num_classes = 3  # normal (0), benign (1), malignant (2)
    model = build_hybrid_model(input_shape=(img_height, img_width, 3), num_classes=num_classes, training=True)  # Added training parameter

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/mias_hybrid_model_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Train the model
    epochs = 50
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )

    # Evaluate the model on test set
    test_results = model.evaluate(test_generator)
    print(f"Test accuracy: {test_results[1]:.4f}")

    # Save the final model
    model.save('models/mias_hybrid_model_final.h5')

    return history, test_results

if __name__ == "__main__":
    history, test_results = train_mias_hybrid_model()