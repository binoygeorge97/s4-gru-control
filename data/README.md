data/dataloader.py
Overview
This module generates synthetic training data for a microgrid dynamical system using a known physics-based state-space model. It is designed to produce time-series datasets for training sequence models (e.g., S4) to learn system dynamics.
The generated dataset follows the transition:
xk+1=Adxk+Bdukx_{k+1} = A_d x_k + B_d u_kxk+1​=Ad​xk​+Bd​uk​
where:

xkx_kxk​: system state (6-dimensional)
uku_kuk​: control input (3-dimensional)
Ad,BdA_d, B_dAd​,Bd​: discrete-time system matrices


Key Features

✅ Physics-driven data generation (no external datasets required)
✅ Fully vectorized for fast batch generation
✅ Configurable dataset size, sequence length, and batch size
✅ Uses APRBS-style control signals for realistic excitation
✅ Train/test split with optional shuffling and reproducibility


Main Functions

get_discrete_matrices(A_continuous, dt=0.01)
Converts a continuous-time system into discrete-time using the bilinear (Tustin) transform.
Arguments

A_continuous (np.ndarray): Continuous-time system matrix (6×6)
dt (float): Time step

Returns

Ad (np.ndarray): Discrete state matrix (6×6)
Bd (np.ndarray): Discrete input matrix (6×3)


generate_microgrid_data_fast(Ad, Bd, batch_size, length=100, seed=42)
Generates synthetic trajectories of the dynamical system.
Arguments

Ad, Bd: Discrete system matrices
batch_size (int): Number of trajectories
length (int): Sequence length
seed (int): Random seed (for reproducibility)

Returns

inputs (numpy.ndarray): shape (batch, length, 9)

[x_k (6 dims), u_k (3 dims)]


targets (numpy.ndarray): shape (batch, length, 6)

x_{k+1} (next state)




create_microgrid_dataloaders(...)
Creates batched training and testing datasets.
Pythoncreate_microgrid_dataloaders(    Ad,    Bd,    bsz=32,    L=100,    n_train=30000,    n_test=200,    seed=42,    shuffle=True,    verbose=False)Show more lines

Arguments









































ArgumentDescriptionAd, BdDiscrete system matricesbszBatch sizeLSequence lengthn_trainNumber of training samplesn_testNumber of test samplesseedRandom seedshuffleWhether to shuffle training dataverbosePrint dataset generation info

Returns
Pythontrainloader, testloader, d_input, d_outputShow more lines

trainloader: list of (inputs, targets) batches
testloader: same format
d_input = 9 (6 states + 3 controls)
d_output = 6 (next state)


Example Usage
Pythonfrom data.dataloader import get_discrete_matrices, create_microgrid_dataloaders# Define continuous system matrix AA = np.random.randn(6, 6)# Convert to discrete-timeAd, Bd = get_discrete_matrices(A)# Create datasettrainloader, testloader, d_in, d_out = create_microgrid_dataloaders(    Ad,    Bd,    bsz=64,    n_train=5000,    verbose=True)Show more lines

Notes

Inputs are structured as [state, control] at each timestep
Targets are the next state
Data is generated on the fly—no files are stored
Designed for training sequence-to-sequence regression models


Dataset Metadata
Pythond_input = 9d_output = 6dt = 0.01Show more lines

✅ Summary
This module provides a fast, reproducible pipeline for generating physically grounded time-series data for training neural network models to learn and control dynamical systems.
