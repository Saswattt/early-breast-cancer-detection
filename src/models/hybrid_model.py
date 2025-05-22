import tensorflow as tf
from tensorflow.keras import layers, models
from .transformer_model import TransformerBlock, positional_encoding

def build_hybrid_model(input_shape=(224, 224, 3), num_classes=3, training=True):
    # CNN part
    inputs = layers.Input(shape=input_shape)
    
    # CNN Feature Extraction
    x = layers.Conv2D(64, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    
    # Prepare for Transformer
    patch_size = x.shape[1]
    projection_dim = 256
    num_patches = (patch_size * patch_size)
    
    # Reshape CNN output for Transformer
    x = layers.Reshape((num_patches, projection_dim))(x)
    
    # Add positional encoding
    positions = positional_encoding(num_patches, projection_dim)
    x += positions
    
    # Transformer blocks
    x = TransformerBlock(projection_dim, num_heads=8, ff_dim=512)(x, training=training)
    x = TransformerBlock(projection_dim, num_heads=8, ff_dim=512)(x, training=training)  # Added training parameter
    
    # Classification head
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)