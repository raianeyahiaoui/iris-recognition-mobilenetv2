# src/utils.py

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def get_dataset(data_dir, batch_size, img_height, img_width, validation_split=0.2):
    """
    Loads training and validation datasets from a directory.
    """
    print(f"Loading data from: {data_dir}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    return train_ds, val_ds
def save_sample_images(dataset, class_names, save_path):
    """Creates a 3x3 grid of sample images and saves it to a file."""
    import os
    import matplotlib.pyplot as plt

    # Create the directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    
    # Save the figure
    plt.savefig(save_path)
    plt.close() # Close the plot to free up memory
    print(f"Project banner saved to: {save_path}")
def plot_history(history):
    """
    Visualizes the training and validation history and saves the plots to files.
    """
    # --- Create a directory to save results ---
    save_dir = 'docs/images' # We will save them directly to the docs folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- Plot and Save Accuracy ---
    plt.figure(figsize=(8, 6))
    sns.set_style("darkgrid")
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    accuracy_plot_path = os.path.join(save_dir, 'accuracy_plot.png')
    plt.savefig(accuracy_plot_path)
    print(f"Accuracy plot saved to: {accuracy_plot_path}")
    
    # --- Plot and Save Loss ---
    plt.figure(figsize=(8, 6))
    sns.set_style("darkgrid")
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    loss_plot_path = os.path.join(save_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to: {loss_plot_path}")

    # --- Show the plots on screen ---
    plt.show()
