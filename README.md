# Understanding Deep Learning — C++ Implementations

C++ implementations and benchmarks of core mathematical concepts from Simon J.D. Prince's *Understanding Deep Learning*, built with zero external dependencies.

## Contents

### 01. 1D Linear Model (`01_linear_model`)
- **Closed-Form Solution (OLS):** Analytical calculation of optimal weight ($w$) and bias ($b$) using variance and covariance.
- **Gradient Descent:** Iterative optimization using squared error loss derivatives.
- **Benchmark:** Execution time comparison using `std::chrono::high_resolution_clock`.

### 02. Shallow Neural Network (`02_shallow_neural_network`)

Using **10 hidden ReLU units** (`31` trainable parameters), the network learns a piecewise-linear approximation of $f(x) = 0.5x^2$ over $x \in [-2, 2]$, reducing Least Squares Error (LSE) loss by **99.96%**.

### Terminal Output ->

```text
NETWORK INFO: [input=1, hidden=10, output=1, parameters=31]

------------------------- BEFORE TRAINING -------------------------

weights_input_hidden:   [ -0.190091 -0.818001 0.846167 -0.70021 -0.595079 -0.269536 0.0117199 -0.526019 0.509674 0.910528 ]
weights_hidden_out:     [ 0.642395 0.544225 0.894213 -0.654247 -0.851901 0.245249 -0.593021 0.961072 0.856564 -0.404536 ]
biases_hidden:          [ 0 0 0 0 0 0 0 0 0 0 ]
bias_out:               [ 0 ]

LSE Loss:       3.90924

-------------------------     TRAINING    -------------------------

Trained with -> learning_rate: 0.005, epoch: 5000

------------------------- AFTER TRAINING  -------------------------

weights_input_hidden:   [ -0.687931 -0.919615 0.974167 -0.598325 -0.458488 -0.314113 -0.0154929 -0.694214 0.704933 0.864869 ]
weights_hidden_out:     [ 1.25095 0.703478 1.05011 -0.544569 -0.762769 0.299829 -0.593618 1.08282 1.31964 -0.287745 ]
biases_hidden:          [ -0.846882 -0.145757 -0.26534 -0.0334033 -0.0473648 -0.0614682 0.0310486 -0.217389 -0.878124 0.0132775 ]
bias_out:               [ 0.0243673 ]

LSE Loss:       0.00158642

-------------------------    TESTING     -------------------------

INPUT   |       DATA    |       GUESS
        |               |
-2      |       2       |       1.98177
-1.5    |       1.125   |       1.14745
-1      |       0.5     |       0.511967
-0.5    |       0.125   |       0.107931
0       |       0       |       0.00211579
0.5     |       0.125   |       0.115138
1       |       0.5     |       0.506795
1.5     |       1.125   |       1.13503
2       |       2       |       1.99182
