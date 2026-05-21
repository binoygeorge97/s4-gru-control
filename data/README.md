# 📘 data/dataloader.py

## Overview

This module generates synthetic training data for a **microgrid dynamical system** using a physics-based state-space model.

The system follows the discrete-time dynamics:

x_{k+1} = A_d x_k + B_d u_k

where:
- x_k : system state (6-dimensional)
- u_k : control input (3-dimensional)
- A_d, B_d : discrete system matrices

The module produces time-series datasets suitable for training sequence models such as S4.

---

## Key Features

- Physics-driven data generation (no external datasets required)
- Fully vectorized for fast batch generation
- Configurable dataset size, sequence length, and batch size
- APRBS-style input signals for realistic system excitation
- Train/test split with optional shuffling and reproducibility

---

## Main Functions

### get_discrete_matrices

```python
get_discrete_matrices(A_continuous, dt=0.01)
