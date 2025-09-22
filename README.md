# Iris Recognition using Transfer Learning with MobileNetV2

![Project Banner](https://user-images.githubusercontent.com/26465436/205796030-51a8767e-a0e2-45e5-a386-a6c38b2537c3.png)

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced iris recognition system implemented in Python using TensorFlow and Keras. This project leverages the power of **transfer learning** with the **MobileNetV2** architecture to build an efficient and accurate classifier for biometric identification.



## 📋 Table of Contents
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Dataset Requirements](#-dataset-requirements)
- [Installation Guide](#-installation-guide)
- [How to Use](#-how-to-use)
- [Results](#-results)
- [Model Architecture](#-model-architecture)
- [License](#-license)
- [Contact](#-contact)

---

## ✨ Key Features

-   **High Accuracy:** Utilizes a state-of-the-art pre-trained MobileNetV2 model to extract robust and discriminative features from iris images.
-   **Efficient Training:** Employs transfer learning to significantly reduce training time and computational cost.
-   **Modular Codebase:** The source code is cleanly structured into modules for model definition, data handling, and training logic.
-   **Data Augmentation:** Easily extensible with TensorFlow's data augmentation layers to improve model generalization.
-   **Performance Visualization:** Automatically generates and displays plots of training/validation accuracy and loss to monitor performance.

---

---

## 📂 Project Structure

The project is organized in a clear and maintainable structure:

```
IrisRecog-MobileNetV2/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── main.py
    ├── model.py
    └── utils.py
```

---

## 🗂️ Dataset Requirements

This model requires a dataset of iris images organized into separate subdirectories for each class (i.e., each individual). The directory structure should be as follows:

```
data/
├── Train_Dataset/
│   ├── class_001/
│   │   ├── image_01.bmp
│   │   └── image_02.bmp
│   ├── class_002/
│   │   ├── image_03.bmp
│   │   └── ...
│   └── ...
└── Test_Dataset/
    ├── class_001/
    │   ├── image_99.bmp
    │   └── ...
    └── ...
```

**Note:** The dataset itself is not included in this repository and must be provided by the user. Place your `Train_Dataset` and `Test_Dataset` folders inside the `data/` directory.


## ⚙️ Installation Guide

Follow these steps to set up the project environment. A local Python 3.8+ installation is required.

**1. Clone the repository:**
``bash
git clone https://github.com/your-username/IrisRecog-MobileNetV2.git
cd IrisRecog-MobileNetV2

**2. Create a virtual environment:**
This helps manage project dependencies without affecting your global Python installation.
``bash
python -m venv venv
source venv/bin/activate  
# On Windows, use `venv\Scripts\activate`


**3. Install the required packages:**
`bash
pip install -r requirements.txt



## 🚀 How to Use

Once the environment is set up and the dataset is in place, you can run the project.

**1. Configure the model:**
Before running, open  `src/main.py` and adjust the `N_CLASSES` variable to match the number of unique individuals in your dataset.

**2. Run the training script:**
Navigate to the source directory and execute the main script.```bash
cd src
python main.py


The script will:
1.  Load the training and testing datasets.
2.  Build the MobileNetV2-based model.
3.  Compile and train the model for the specified number of epochs.
4.  Save the trained model to a new directory named `MobilenetV2`.
5.  Display the performance plots.


## 📊 Results

After training, the script will output plots visualizing the model's accuracy and loss over each epoch. This helps in assessing overfitting and overall performance.

*(Here you can add a sample image of your results plot after you run it)*

![Sample Results Plot](https-placeholder-for-your-results.png)



## 🏗️ Model Architecture

The model is constructed using the following layers:

1.  **Input Layer:** Accepts images of size `(400, 300, 3)`.
2.  **Preprocessing Layer:** Normalizes pixel values to the range expected by MobileNetV2.
3.  **MobileNetV2 Base:** The convolutional base of MobileNetV2 with frozen, pre-trained ImageNet weights acts as the primary feature extractor.
4.  **Global Average Pooling:** Flattens the feature maps into a single vector per image.
5.  **Dropout Layer:** A regularization layer to prevent overfitting.
6.  **Dense Output Layer:** A fully connected layer that produces the final classification logits for each class.


## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


## 📞 Contact

Feel free to reach out with any questions or for collaboration opportunities.

**Yahiaoui Raiane**
-   **Email:** `ikba.king2015@gmail.com`
-   **LinkedIn:** [linkedin.com/in/yahiaoui-raiane-253911262](https://www.linkedin.com/in/yahiaoui-raiane-253911262/)
