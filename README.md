# Overview

## How a neural network works
A neural network is a computational model inspired by the human brain, consisting of layers of interconnected nodes (neurons). Each neuron receives inputs, applies a linear transformation (weighted sum plus bias), and passes the result through a nonlinear activation function. Layers are stacked so that the output of one layer becomes the input to the next.

### Forward Pass
During the forward pass, input data is propagated through the network layer by layer:
1. **Linear Transformation:** Each layer computes a weighted sum of its inputs and adds a bias.
2. **Activation Function:** The result is passed through a nonlinear function (e.g., tanh or softmax).
3. **Output:** The final layer produces predictions (e.g., class probabilities).

### Backward Pass (Backpropagation)
After computing the output and loss (difference between predictions and true labels), the network performs the backward pass:
1. **Gradient Computation:** The gradient of the loss with respect to each parameter (weights and biases) is calculated using the chain rule.
2. **Parameter Update:** These gradients are used to update the parameters, typically via stochastic gradient descent.

### Training Process
Training alternates between forward and backward passes:
- **Forward pass:** Computes predictions and loss for the current parameters.
- **Backward pass:** Computes gradients and updates parameters to minimize the loss.
This iterative process continues for multiple epochs, gradually improving the network's performance on the task.

## NeuralNet.py

This file implements a modular neural network framework from scratch using NumPy. It is organized into several classes, each responsible for a different aspect of neural network computation:

### CoreFunctions
- **errorRate**: Computes the error rate between true and predicted labels.
- **addBiasFwd / addBiasBkd**: Adds a bias term to the input (forward) and removes it during backpropagation.
- **linearFwd / linearBkd**: Implements the forward and backward pass for a linear (fully connected) layer.
- **tanhFwd / tanhBkd**: Implements the forward and backward pass for the tanh activation function.
- **softmaxFwd / softmaxBkd**: Implements the forward and backward pass for the softmax activation function.
- **NLLLossFwd / NLLLossBkd**: Computes the negative log-likelihood loss and its gradient.

### ForwardPass
- **SLNN**: Performs the forward pass for a single-layer neural network (input → tanh → output → softmax).
- **DLNN**: Performs the forward pass for a double-layer neural network (input → tanh → tanh → output → softmax).

### BackwardPass
- **SLNN**: Computes gradients for all parameters in a single-layer neural network using backpropagation.
- **DLNN**: Computes gradients for all parameters in a double-layer neural network using backpropagation.

### NeuralNetModel
- **trainSLNN**: Trains a single-layer neural network using stochastic gradient descent, tracking training and validation loss.
- **trainDLNN**: Trains a double-layer neural network similarly.
- **predictSingle / predictDouble**: Makes predictions using the trained single- or double-layer model.

## fashion_classify.ipynb

This Jupyter notebook demonstrates how to use the `NeuralNet.py` library to train and evaluate neural networks on the Fashion MNIST dataset. Key steps include:

- **Data Loading and Preprocessing**: Loads Fashion MNIST data, reshapes and normalizes it, and splits it into training, validation, and test sets.
- **Helper Functions**: Provides visualization and utility functions for displaying images and tables.
- **Training**: Trains both a single-layer and a double-layer neural network using the custom library, and plots loss curves.
- **Prediction and Evaluation**: Uses the trained models to predict labels on the test set and computes accuracy.
- **Visualization**: Visualizes predictions, coloring images green for correct and red for incorrect predictions, and displays predicted labels.

Similarly, the `mnist_classify.ipynb` notebook classifies images from the Mnist dataset.

## Usage
- Run `fashion_classify.ipynb` and `mnist_classify.ipynb` to see the full workflow, from data loading to model evaluation and visualization.
- Modify the notebook to experiment with different architectures or datasets.

---
This project is intended for educational purposes and demonstrates the inner workings of neural networks and backpropagation without relying on high-level deep learning libraries for the core logic.
