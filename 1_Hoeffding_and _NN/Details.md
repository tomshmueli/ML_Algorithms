# Programming Assignment 1

This repository contains solutions for two main tasks: Visualizing the Hoeffding Bound and implementing the Nearest Neighbor (NN) algorithm on the MNIST dataset.

## Task 1: Visualizing the Hoeffding Bound

### Objective
Generate a visualization of the empirical probabilities of deviations from the expected mean for a Bernoulli distribution, and compare it with the Hoeffding Bound.

### Method
1. **Data Generation**: Generate an N × n matrix of samples from Bernoulli(1/2), where N = 200,000 and n = 20.
2. **Empirical Mean Calculation**: For each row, calculate the empirical mean.
3. **Probability Calculation**: For 50 values of ε ∈ [0,1], calculate the empirical probability that the absolute difference between the empirical mean and 1/2 is greater than ε.
4. **Visualization**: Plot the empirical probability as a function of ε and overlay the Hoeffding bound.

## Task 2: Nearest Neighbor on MNIST Dataset

### Objective
Study the performance of the k-Nearest Neighbor algorithm on the MNIST dataset of handwritten digits.

### Dataset
- The MNIST dataset includes 70,000 images (28x28 pixels) of handwritten digits.
- Each image is treated as a 784-dimensional vector.

### Implementation Steps
1. **Data Loading and Preparation**: Load the MNIST dataset and prepare training and testing sets.
2. **k-NN Function**:
   - **Inputs**: Training images, labels, a query image, and a value k.
   - **Output**: Prediction of the query image based on the k nearest neighbors using the Euclidean L2 metric.
3. **Experiments**:
   - Run the k-NN algorithm on the first 1,000 training images and calculate the prediction accuracy using k = 10.
   - Explore the effect of different k values (1 to 100) on prediction accuracy.
   - Investigate how increasing the number of training images affects accuracy.

### Discussion Points
- Compare the results against a completely random predictor.
- Discuss the optimal number of neighbors (k) for the best prediction accuracy.
- Analyze the trade-offs in increasing the training set size.

## Usage

Details on how to run and reproduce the results can be found in the respective script documentation within this repository. Ensure you have scikit-learn version 1.0.2 or above installed to avoid compatibility issues.

## Results

- The resulting plots and accuracy measurements are provided in the `results` folder.
- This includes a comparison of empirical and theoretical probabilities, accuracy as a function of k, and accuracy improvements with increasing training sizes.
