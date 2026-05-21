import numpy as np
import time

# =========================================================================
# 1. THE PHYSICS ENGINE (Bilinear Discrete Matrices - Paper Dynamics)
# =========================================================================
# CHANGE 1: Require `A_continuous` as an argument and remove the hardcoded matrices
def get_discrete_matrices(A_continuous, dt=0.01):

    # Input matrix (B_i) is the same for all nodes
    B_i = np.array([[0.0], [1.0]])

    # Assemble into global 6x3 B matrix
    B = np.block([
        [B_i, np.zeros((2,1)), np.zeros((2,1))],
        [np.zeros((2,1)), B_i, np.zeros((2,1))],
        [np.zeros((2,1)), np.zeros((2,1)), B_i]
    ])

    # Bilinear Transform (Tustin)
    I = np.eye(6)
    inv_term = np.linalg.inv(I - (dt / 2.0) * A_continuous)
    Ad = inv_term @ (I + (dt / 2.0) * A_continuous)
    Bd = inv_term @ B * dt

    return Ad, Bd

# =========================================================================
# 2. VECTORIZED GENERATOR (Generates all data instantly in NumPy)
# =========================================================================
def fast_vectorized_aprbs(batch_size, length, min_val, max_val, hold_prob, rng=None):
    if rng is None: rng = np.random.RandomState(42)
    random_amps = rng.uniform(min_val, max_val, size=(batch_size, 3, length))

    switches = rng.rand(batch_size, 3, length) > hold_prob
    switches[:, :, 0] = True

    signal = np.zeros((batch_size, 3, length))
    current_amp = random_amps[:, :, 0]

    for k in range(length):
        current_amp = np.where(switches[:, :, k], random_amps[:, :, k], current_amp)
        signal[:, :, k] = current_amp

    return signal

# CHANGE 2: Require `Ad` and `Bd` as arguments instead of generating them inside
def generate_microgrid_data_fast(Ad, Bd, batch_size, length=100, seed=42):
    rng = np.random.RandomState(seed)

    U_signals = fast_vectorized_aprbs(batch_size, length, -1.0, 1.0, hold_prob=0.8, rng=rng)

    # Updated to 9 input channels (6 states, 3 controls)
    batch_inputs = np.zeros((batch_size, length, 9))
    batch_targets = np.zeros((batch_size, length, 6))
    X_current = np.zeros((batch_size, 6))

    for k in range(length):
        U_k = U_signals[:, :, k]

        batch_inputs[:, k, 0:6] = X_current
        batch_inputs[:, k, 6:9] = U_k

        # Vectorized physics update across the entire batch uses the INJECTED matrices
        X_next = X_current.dot(Ad.T) + U_k.dot(Bd.T)
        batch_targets[:, k, :] = X_next
        X_current = X_next

    return batch_inputs, batch_targets

# =========================================================================
# 3. THE CUSTOM "LIST" DATALOADER
# =========================================================================
# CHANGE 3: Require `Ad` and `Bd` to pass them down to the data generator
def create_microgrid_dataloaders(Ad, Bd, bsz=32, L=100):
    n_train = 30000
    n_test = 200
    total_samples = n_train + n_test

    print(f"[*] Generating {total_samples} samples for Microgrid Dataset...")
    start_time = time.time()

    # Pass the injected matrices down to the generator
    all_inputs, all_targets = generate_microgrid_data_fast(Ad, Bd, batch_size=total_samples, length=L)
    print(f"[*] Raw data generated in {time.time() - start_time:.2f} seconds.")

    # 2. Manual Array Slicing for Train/Test Split
    train_in = all_inputs[:n_train]
    train_out = all_targets[:n_train]

    test_in = all_inputs[n_train:]
    test_out = all_targets[n_train:]

    # 3. Manual Chunking Loop for Training Data
    trainloader = []
    num_train_batches = n_train // bsz
    for i in range(num_train_batches):
        batch_x = train_in[i*bsz : (i+1)*bsz]
        batch_y = train_out[i*bsz : (i+1)*bsz]
        trainloader.append((batch_x, batch_y))

    # 4. Manual Chunking Loop for Testing Data
    testloader = []
    num_test_batches = n_test // bsz
    for i in range(num_test_batches):
        batch_x = test_in[i*bsz : (i+1)*bsz]
        batch_y = test_out[i*bsz : (i+1)*bsz]
        testloader.append((batch_x, batch_y))

    print(f"[*] Created Custom Loaders: {len(trainloader)} Train batches, {len(testloader)} Test batches.")

    return trainloader, testloader, 9, 6





Datasets = {
    "microgrid": create_microgrid_dataloaders,
}

DatasetMetadata = {
    "microgrid": {
        "input_labels": [
            "$e_1$ (Energy at Node 1)",
            "$r_1$ (Impedance at Node 1)",
            "$e_2$ (Energy at Node 2)",
            "$r_2$ (Impedance at Node 2)",
            "$e_3$ (Energy at Node 3)",
            "$r_3$ (Impedance at Node 3)",
            "Control $u_1$", "Control $u_2$", "Control $u_3$",
            #"Disturbance $w_1$", "Disturbance $w_2$", "Disturbance $w_3$"
        ],
        "output_labels": [
            "Next $e_1$ (Node 1)",
            "Next $r_1$ (Node 1)",
            "Next $e_2$ (Node 2)",
            "Next $r_2$ (Node 2)",
            "Next $e_3$ (Node 3)",
            "Next $r_3$ (Node 3)"
        ],
        "dt": 0.01
    }
}
