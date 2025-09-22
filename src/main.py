# src/main.py

import os
import tensorflow as tf
from model import build_model
# ADD save_sample_images to this import
from utils import get_dataset, plot_history, save_sample_images 

# --- Configuration & Hyperparameters ---
# (Keep all your existing configurations the same)
TRAIN_DIR = "../data/Train_Dataset"
TEST_DIR = "../data/Test_Dataset"
N_CLASSES = 50
IMG_HEIGHT = 400
IMG_WIDTH = 300
CHANNELS = 3
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 64
DROPOUT_RATE = 0.3
MODEL_SAVE_PATH = 'MobilenetV2'

def main():
    """Main function to run the iris recognition training pipeline."""
    if not os.path.exists(TRAIN_DIR):
        print("Error: Training directory not found.")
        return

    # 1. Load Datasets
    train_ds, _ = get_dataset(TRAIN_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, validation_split=0.01)
    val_ds, _ = get_dataset(TEST_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, validation_split=0.01)

    # --- NEW PART: SAVE THE BANNER IMAGE ---
    # Get the class names from the dataset
    class_names = train_ds.class_names
    # Define the save path for the banner
    banner_path = '../docs/images/project_banner.png'
    # Call the function to save the image
    save_sample_images(train_ds, class_names, banner_path)
    # --- END OF NEW PART ---

    # 2. Build the Model
    # (The rest of your main function stays exactly the same)
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
