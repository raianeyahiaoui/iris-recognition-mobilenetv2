# src/model.py

import tensorflow as tf

def build_model(input_shape, n_classes, dropout_rate=0.3):
    """
    Builds an iris recognition model using MobileNetV2 for transfer learning.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        n_classes (int): The total number of unique classes (individuals).
        dropout_rate (float): The dropout rate for regularization.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    # Define the input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Preprocessing layer for MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Load the base MobileNetV2 model with pre-trained ImageNet weights
    # We exclude the top classification layer to add our own
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model layers so they are not updated during training
    base_model.trainable = False

    # Pass the preprocessed input through the base model
    x = base_model(x, training=False)

    # Add custom layers for classification
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(n_classes)(x) # Output layer with logits

    # Create the final model
    model = tf.keras.Model(inputs, outputs)

    return model
