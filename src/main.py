# src/main.py

import os
import tensorflow as tf
from model import build_model
from utils import get_dataset, plot_history

# --- Configuration & Hyperparameters ---
# Dataset Parameters
TRAIN_DIR = "../data/Train_Dataset"
TEST_DIR = "../data/Test_Dataset" # This will be used as the validation set in this script

# Image Parameters
N_CLASSES = 50      # IMPORTANT: Adjust this to the number of classes in your dataset
IMG_HEIGHT = 400
IMG_WIDTH = 300
CHANNELS = 3

# Training Parameters
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 64

# Model Parameters
DROPOUT_RATE = 0.3
MODEL_SAVE_PATH = 'MobilenetV2'

def main():
    """Main function to run the iris recognition training pipeline."""
    # Ensure dataset paths exist
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print("Error: Dataset directories not found.")
        print("Please place your training and testing data in the 'data/Train_Dataset' and 'data/Test_Dataset' folders.")
        return

    # 1. Load Datasets
    # Note: We use the 'TEST_DIR' as the validation data for the .fit() method
    # An alternative is to use a single directory and a validation_split as in utils.py
    train_ds, _ = get_dataset(TRAIN_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, validation_split=0.01) # Small split to satisfy function
    val_ds, _ = get_dataset(TEST_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, validation_split=0.01)

    # 2. Build the Model
    input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    model = build_model(input_shape, N_CLASSES, DROPOUT_RATE)

    # 3. Compile the Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()

    # 4. Train the Model
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds
    )
    print("--- Model Training Finished ---\n")

    # 5. Save the Model
    print(f"Saving model to '{MODEL_SAVE_PATH}'...")
    model.save(MODEL_SAVE_PATH)
    print("Model saved successfully.")

    # 6. Visualize Results
    plot_history(history)


if __name__ == '__main__':
    main()
