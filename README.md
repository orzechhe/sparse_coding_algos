# Sparse Coding for EECS 559 Project
This is the implementation of sparse coding algorithms from Arora et al. (2015) for the EECS 559 project at the University of Michigan. The project evaluates Algorithms 2 (Neural Update), 3 (Pairwise Initialization), and 5 (Low-Error Variant), comparing them to K-SVD, on synthetic and MNIST datasets to test synthetic and real world data. Metrics include reconstruction error, column-wise error, closeness (Algorithm 3), and runtime, with results visualized through error plots and dictionary patches.
Overview

*The project implements:*

**Algorithm 2:** Neural Update with adaptive learning rate for dictionary learning.

**Algorithm 3:** Pairwise Initialization using random projections.

**Algorithm 5:** Low-Error Variant with momentum and early stopping.

**K-SVD:** Baseline using scikit-learn’s MiniBatchDictionaryLearning.


\
**Results:**

Algorithm 2 achieves high reconstruction errors (0.188 synthetic, 0.110 MNIST).
Algorithm 3’s initialization yields high closeness (9.973 synthetic, 18.347 MNIST), which can limit performance.
K-SVD does best on MNIST (0.052 error).
High column-wise errors (e.g., 1.414, 1.403) indicate dictionary recovery challenges.

\
**Prerequisites:**

*Software:*
Python 3.11
Libraries: numpy, torch, torchvision, scikit-learn, matplotlib


*Hardware:* I tested on a MacBook Air (CPU only). GPU is best for MNIST processing.

*Install:*
pip install numpy torch torchvision scikit-learn matplotlib
\
*Repository Structure*
sparse_coding_eecs559/
├── sparse_coding_with_mnist_metrics.py  # Main script for experiments
└── README.md                           # This file



*Expected Output:*

Console output with metrics:Processing Synthetic Data...
Algorithm 3 - Closeness: 9.9725, Error: 0.5004, Column Error: 1.3205, Runtime: 0.6117s
Algorithm 2 - Error: 0.1878, Column Error: 1.4135, Runtime: 0.6807s
...
Processing MNIST Data...
Algorithm 3 - Closeness: 18.3468, Error: 0.1389, Column Error: 1.3994, Runtime: 31.5921s
K-SVD - Error: 0.0521, Column Error: 0.0000, Runtime: 16.9299s


*Generated plots:*
synthetic_error.png: Reconstruction error vs. iterations (synthetic).
mnist_error.png: Reconstruction error vs. iterations (MNIST).
dict_algo2.png, dict_algo5.png, dict_ksvd.png: Learned dictionary patches.


**Results**

*The script produces:*

Synthetic Data (( n=200, p=50, m=50, k=3 )):
Algorithm 2: Error 0.188 (expected 0.123), Column Error 1.414, Runtime 0.681s.
Algorithm 3: Closeness 9.973, Error 0.500, Column Error 1.321, Runtime 0.612s.
Algorithm 5: Error 0.463 (expected 0.005), Column Error 1.336, Runtime 0.717s.
K-SVD: Error 0.216, Column Error 1.421, Runtime 1.633s.


MNIST Data (( n=784, p=200, m=1000, k=20 )):
Algorithm 2: Error 0.110 (expected 0.160), Column Error 1.403, Runtime 18.943s.
Algorithm 3: Closeness 18.347, Error 0.139, Column Error 1.399, Runtime 31.592s.
Algorithm 5: Error 0.141 (expected 0.0013), Column Error 1.397, Runtime 24.506s.
K-SVD: Error 0.052, Column Error 0.000, Runtime 16.930s.
