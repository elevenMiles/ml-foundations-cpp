# Understanding Deep Learning — C++ Implementations

From-scratch C++20 implementations and benchmarks of core mathematical concepts from Simon J.D. Prince's *Understanding Deep Learning*, built with zero external dependencies.

## Contents

### 01. 1D Linear Model (`01_linear_model`)
- **Closed-Form Solution (OLS):** Analytical calculation of optimal weight ($w$) and bias ($b$) using variance and covariance.
- **Gradient Descent:** Iterative optimization using squared error loss derivatives.
- **Benchmark:** Execution time comparison using `std::chrono::high_resolution_clock`.
