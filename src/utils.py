# src/utils.py

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def get_dataset(data_dir, batch_size, img_height, img_width, validation_split=0.2):
    """
    Loads training and validation datasets from a directory.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): The size of the batches of data.
        img_height (int): The height to resize images to.
        img_width (int): The width to resize images to.
        validation_split (float): Fraction of data to reserve for validation.

    Returns:
        tuple: A tuple containing the training and validation datasets.
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

def plot_history(history):
    """
    Visualizes the training and validation history for accuracy and loss.

    Args:
        history (tf.keras.callbacks.History): The history object returned by model.fit().
    """
    epochs = range(1, len(history.history['accuracy']) + 1)

    # Create a DataFrame for easy plotting
    results_df = pd.DataFrame({
        "epoch": epochs,
        "accuracy": history.history["accuracy"],
        "val_accuracy": history.history["val_accuracy"],
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"]
    }).set_index("epoch")

    # Plotting
    sns.set_style("darkgrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Accuracy
    sns.lineplot(data=results_df[["accuracy", "val_accuracy"]], dashes=False, ax=ax1)
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')

    # Plot Loss
    sns.lineplot(data=results_df[["loss", "val_loss"]], dashes=False, ax=ax2)
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')

    plt.suptitle('Model Training History', fontsize=16)
    plt.show()
