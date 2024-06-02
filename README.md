# Digit Recognition Using MNIST Dataset

This project demonstrates a digit recognition system using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The project is implemented in Python using TensorFlow and Keras.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup and Installation](#setup-and-installation)
- [Running the Code](#running-the-code)
- [Model Evaluation](#model-evaluation)
- [Next Steps](#next-steps)
- [License](#license)

## Introduction

The goal of this project is to build a neural network that can accurately classify handwritten digits (0-9) from the MNIST dataset. The MNIST dataset is a benchmark dataset in the field of machine learning and consists of 70,000 28x28 grayscale images of handwritten digits.

## Dataset

The MNIST dataset is available in TensorFlow and can be easily loaded using the `keras.datasets` module. It consists of 60,000 training images and 10,000 test images.

## Model Architecture

The model used in this project is a Convolutional Neural Network (CNN) with the following architecture:
- Two Convolutional layers with ReLU activation and MaxPooling
- Flatten layer to convert 2D data to 1D
- Dense layer with 128 units and ReLU activation
- Output Dense layer with 10 units (one for each digit) and softmax activation

## Setup and Installation

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/digit-recognition-mnist.git
    cd digit-recognition-mnist
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Code

1. Run the script:
    ```bash
    python digit_recognition.py
    ```

This will load the MNIST dataset, normalize the data, build and train the CNN model, and evaluate its performance on the test set.

## Model Evaluation

The script will print the model summary, training history, and the test accuracy. It will also display plots of the training and validation accuracy and loss over epochs. Additionally, it will show sample test images with predicted and true labels.

## Next Steps

Here are a few suggestions to take this project to the next level:
- Experiment with different model architectures, such as adding more layers or changing activation functions.
- Implement data augmentation to artificially expand the training dataset.
- Tune hyperparameters like learning rate, batch size, and number of epochs.
- Try different optimizers like SGD or RMSprop.
- Evaluate the model on different datasets or real-world data.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
