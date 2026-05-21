import torch
import optax
from functools import partial
from tqdm import tqdm
import os
import shutil

from flax import serialization


# 2. UTILITIES
# ==============================================================================

def create_optimizer(model, base_lr, weight_decay, total_steps, lr_layer_multipliers=None):
    """
    Creates an NNX Optimizer.
    """
    if total_steps > 0:
        schedule_fn = lambda lr: optax.cosine_onecycle_schedule(
            peak_value=lr,
            transition_steps=total_steps,
            pct_start=0.1,
        )
    else:
        schedule_fn = lambda lr: optax.constant_schedule(lr)

    tx = optax.adamw(
        learning_rate=schedule_fn(base_lr),
        weight_decay=weight_decay
    )

    # Explicitly specify that we are optimizing nnx.Param variables
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


@nnx.jit
def train_step(model, optimizer, x_batch, y_batch, dropout_keys):
    """
    NNX Train Step for REGRESSION (MSE Loss) using optimizer.update.
    """
    def loss_fn(model):
        predictions, _ = batched_reg_runner(model, x_batch, dropout_keys, True)
        loss = jnp.mean((predictions - y_batch) ** 2)
        return loss, predictions

    # Get loss and gradients
    (loss, preds), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    # Apply updates using the optimizer
    optimizer.update(model, grads)

    return loss

@nnx.jit
def eval_step(model, x_batch, y_batch):
    B = x_batch.shape[0]
    dummy_keys = jax.random.split(jax.random.PRNGKey(0), B)
    predictions, _ = batched_reg_runner(model, x_batch, dummy_keys, False)
    loss = jnp.mean((predictions - y_batch) ** 2)
    return loss

def validate(model, testloader):
    losses = []
    for batch in testloader:
        inputs, targets = batch
        inputs = jnp.array(inputs)
        targets = jnp.array(targets)
        loss = eval_step(model, inputs, targets)
        losses.append(loss)
    return np.mean(losses)

def train_epoch(rng, model, optimizer, trainloader):
    batch_losses = []
    for batch in tqdm(trainloader, desc="Training", disable=True):
        inputs, targets = batch
        inputs = jnp.array(inputs)
        targets = jnp.array(targets)

        rng, drop_rng = jax.random.split(rng)
        batch_keys = jax.random.split(drop_rng, inputs.shape[0])

        loss = train_step(model, optimizer, inputs, targets, batch_keys)
        batch_losses.append(loss)

    return rng, np.mean(batch_losses)




# Pass Ad, Bd, and a UNIQUE save_path into the function!
def safe_train_regression(dataset, layer, seed, model_cfg, train_cfg, Ad, Bd, unique_save_path):
    print(f"[*] Setting Randomness (Seed: {seed})...")
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)
    key, model_rng, train_rng = jax.random.split(key, 3)

    # 1. Use the dynamic Ad and Bd to create the data
    trainloader, testloader, d_input, d_output = create_microgrid_dataloaders(
        Ad=Ad, Bd=Bd, bsz=train_cfg['bsz'], L=model_cfg.get('l_max', 100)
    )

    # 2. Initialize Model (Identical to your original code)
    rngs = nnx.Rngs(params=model_rng, dropout=0)
    stacked_args = model_cfg.copy()
    s4_N = stacked_args.pop('N')
    l_max = stacked_args.pop('l_max')
    if 'embedding' in stacked_args:
        stacked_args.pop('embedding')

    model = StackedModelRegression(
        layer_cls=S4LayerEnsemble,
        layer_args={'N': s4_N, 'l_max': l_max},
        d_input=d_input,
        d_output=d_output,
        decode=False,
        rngs=rngs,
        **stacked_args
    )

    # 3. Initialize Optimizer (Identical to your original code)
    optimizer = create_optimizer(
        model,
        base_lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
        total_steps=len(trainloader) * train_cfg['epochs']
    )

    # 4. Training Loop
    best_loss = 1e9

    for epoch in range(train_cfg['epochs']):
        train_rng, train_loss = train_epoch(train_rng, model, optimizer, trainloader)
        test_loss = validate(model, testloader)

        if test_loss < best_loss:
            best_loss = test_loss

            # Use the UNIQUE save path provided by the Ray worker!
            os.makedirs(os.path.dirname(unique_save_path), exist_ok=True)

            full_config = {
                'dataset': dataset, 'layer': layer,
                'model': model_cfg, 'train': train_cfg
            }
            save_model(model, full_config, unique_save_path)

    return model, best_loss



def save_model(model, config, filename="s4_model.msgpack"):
    # 1. Extract the parameters
    # FIX: Convert the NNX State object to a standard Python dictionary
    model_state = nnx.state(model, nnx.Param).to_pure_dict()

    # 2. Bundle with config
    checkpoint_data = {
        'model_state': model_state,
        'config': config
    }

    # 3. Serialize to bytes
    byte_data = serialization.to_bytes(checkpoint_data)

    # 4. Write to file
    with open(filename, 'wb') as f:
        f.write(byte_data)

    print(f"✅ Model saved to {filename}")



@ray.remote(num_gpus=0.2) # Adjust this based on your VRAM (e.g., 0.2 = 20% of GPU)
def train_single_model(matrix_dict, hp_dict):
    # Prevent JAX VRAM Hoarding and concurrent initialization crashes
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    matrix_id = matrix_dict["matrix_id"]
    A_continuous = matrix_dict["A_continuous"]

    print(f"[*] Worker Started: Training Matrix {matrix_id}...")

    # Static configuration logic
    model_cfg = {
        "d_model": hp_dict["d_model"],
        "n_layers": hp_dict["n_layers"],
        "N": hp_dict["N"],
        "l_max": hp_dict["l_max"],
        "dropout": hp_dict["dropout"],
        "prenorm": hp_dict["prenorm"],
        "embedding": False,
    }

    train_cfg = {
        "epochs": hp_dict["epochs"],
        "bsz": hp_dict["batch_size"],
        "lr": hp_dict["lr"],
        "weight_decay": 0.0,
    }

    unique_save_path = f"checkpoints/sweep/mat{matrix_id}_best_model.msgpack"

    # Execute Pipeline
    Ad, Bd = get_discrete_matrices(A_continuous)

    trained_model, final_mse = safe_train_regression(
        dataset="microgrid",
        layer="s4",
        seed=42,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        Ad=Ad,
        Bd=Bd,
        unique_save_path=unique_save_path
    )

    return {"matrix_id": matrix_id, "mse": final_mse, "path": unique_save_path}

