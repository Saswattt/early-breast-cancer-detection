import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_model(input_shape=(224, 224, 3), patch_size=16, num_classes=3):
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    projection_dim = 128  # Increased from 64
    num_heads = 8  # Increased from 4
    transformer_units = [
        projection_dim * 4,  # Increased multiplier
        projection_dim,
    ]
    transformer_layers = 6  # Increased from 4
    mlp_head_units = [2048, 1024, 512]  # Added another layer

    inputs = layers.Input(shape=input_shape)
    
    # Create patches with layer normalization
    x = layers.Conv2D(projection_dim, patch_size, patch_size, padding="valid")(inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    patches = layers.Reshape((num_patches, projection_dim))(x)
    
    # Add positional embeddings
    positions = positional_encoding(num_patches, projection_dim)
    patches += positions
    
    # Add dropout after positional embeddings
    patches = layers.Dropout(0.1)(patches)
    
    # Create transformer blocks with residual connections
    for _ in range(transformer_layers):
        # Layer normalization and residual connection
        x = layers.LayerNormalization(epsilon=1e-6)(patches)
        x = TransformerBlock(projection_dim, num_heads, transformer_units[0])(x, training=True)
        patches = layers.Add()([patches, x])
    
    # Create MLP head with stronger regularization
    representation = layers.LayerNormalization(epsilon=1e-6)(patches)
    representation = layers.GlobalAveragePooling1D()(representation)  # Changed from Flatten
    
    for units in mlp_head_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(representation)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)  # Increased dropout
        representation = x
    
    outputs = layers.Dense(num_classes, activation="softmax")(representation)
    
    return models.Model(inputs=inputs, outputs=outputs)