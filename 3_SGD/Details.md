# Programming Assignment 3: Stochastic Gradient Descent (SGD) on MNIST

## Overview

This assignment focuses on implementing and analyzing Stochastic Gradient Descent (SGD) for optimizing hinge loss and log loss on a subset of the MNIST dataset. The subset includes images of the digits 0 and 8, classified using a binary label system.

## Guidelines

- **MNIST Data**: If there are issues with loading the dataset via the script, it can be manually downloaded from [this GitHub repository](https://github.com/amplab/datascience-sp14/blob/master/lab7/mldata/mnist-original.mat).
- **Python Version**: Ensure that your code is compatible with Python 3.

## Task 1: SGD for Hinge Loss

### Description
Implement SGD to optimize the hinge loss with L2-regularization. The hinge loss function is defined as:
\( \ell(w, x, y) = C \cdot \max\{0, 1 - y \langle w, x \rangle \} + 0.5 \|w\|^2 \)

### Subtasks
1. **Train the Classifier**:
   - Implement cross-validation on the validation set to find the optimal learning rate \( \eta_0 \) with \( T = 1000 \) iterations and \( C = 1 \).
   - Plot the average accuracy on the validation set as a function of \( \eta_0 \).

2. **Optimize Regularization Parameter \( C \)**:
   - Based on the best \( \eta_0 \), cross-validate to find the best \( C \).
   - Plot the average accuracy on the validation set as a function of \( C \).

3. **Long-Term Training**:
   - Train the classifier with the best \( C \) and \( \eta_0 \) for \( T = 20000 \) iterations.
   - Visualize the final weight vector \( w \) as a 28x28 image.

4. **Test Set Evaluation**:
   - Evaluate the accuracy of the classifier on the test set.

## Task 2: SGD for Log Loss

### Description
Optimize the log loss, which is defined as:
\( \ell_{\text{log}}(w, x, y) = \log(1 + e^{-y w \cdot x}) \)

### Subtasks
1. **Classifier Training**:
   - Find the best \( \eta_0 \) using cross-validation with \( T = 1000 \).
   - Train the classifier for \( T = 20000 \) and visualize the weight vector \( w \).

2. **Weight Norm Analysis**:
   - Track and plot the norm of \( w \) over \( T = 20000 \) iterations to analyze how it changes.

3. **Final Model Evaluation**:
   - Assess the accuracy of the classifier on the test set and explain the observed changes in the weight vector norm.

