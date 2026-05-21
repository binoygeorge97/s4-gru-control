
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from .s4 import S4LayerEnsemble

class SequenceBlockNNX(nnx.Module):
    def __init__(self,
                 layer_cls: type[nnx.Module],
                 layer_args: dict,
                 d_model: int,
                 dropout: float,
                 prenorm: bool = True,
                 glu: bool = True,
                 decode: bool = False,
                 *, rngs: nnx.Rngs):

        self.d_model = d_model
        self.prenorm = prenorm
        self.glu = glu
        self.decode = decode
        self.dropout_rate = dropout

        self.seq = layer_cls(
            **layer_args,
            D_MODEL=d_model,
            decode=decode,
            rngs=rngs
        )

        # Mixing Layers
        keys = jax.random.split(rngs.params(), 3)
        self.norm = nnx.LayerNorm(d_model, rngs=nnx.Rngs(params=keys[0]))
        self.out = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(params=keys[1]))
        if self.glu:
            self.out2 = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(params=keys[2]))

        self.drop = nnx.Dropout(dropout, broadcast_dims=[0])

    def __call__(self, x, s4_state, *, rngs: nnx.Rngs = None, training: bool = True):
        skip = x

        if self.prenorm:
            x = self.norm(x)

        # --- ROBUST FIX: Manual JAX Vmap ---
        # 1. Split the S4 layer into Graph (Static) and Params (Data)
        seq_graph, seq_params = nnx.split(self.seq)

        # 2. Define a Pure Function for ONE channel
        def run_one_channel(params_slice, u_slice, state_slice):
            # Reconstruct the layer for this single channel
            single_layer = nnx.merge(seq_graph, params_slice)
            # Run it
            return single_layer(u_slice, state_slice)

        # 3. Use standard JAX vmap
        # seq_params: Axis 0 corresponds to D_MODEL (H)
        # x (Input): Axis 1 corresponds to H -> (L, H)
        # s4_state: Axis 0 corresponds to H -> (H, N)
        x, new_s4_state = jax.vmap(
            run_one_channel,
            in_axes=(0, 1, 0),  # Map over params(0), input(1), state(0)
            out_axes=(1, 0)     # Stack output(1), new_state(0)
        )(seq_params, x, s4_state)

        # -----------------------------------

        x = nnx.gelu(x)

        if training and rngs:
             x = self.drop(x, rngs=rngs)

        if self.glu:
            gate = jax.nn.sigmoid(self.out2(x))
            x = self.out(x) * gate
        else:
            x = self.out(x)

        if training and rngs:
            x = self.drop(x, rngs=rngs)

        x = skip + x

        if not self.prenorm:
            x = self.norm(x)

        return x, new_s4_state



class StackedModelRegression(nnx.Module):
    def __init__(self,
                 layer_cls: type[nnx.Module],
                 layer_args: dict,
                 d_input: int,
                 d_output: int,
                 d_model: int,
                 n_layers: int,
                 prenorm: bool = True,
                 dropout: float = 0.0,
                 decode: bool = False,
                 *, rngs: nnx.Rngs):

        self.d_model = d_model
        self.d_output = d_output
        self.n_layers = n_layers
        self.prenorm = prenorm
        self.decode = decode
        self.dropout = dropout

        keys = jax.random.split(rngs.params(), 3)

        # 1. Linear Encoder (No Embeddings!)
        # Projects 1 feature (sine value) -> d_model (Hidden)
        self.encoder = nnx.Linear(d_input, d_model, rngs=nnx.Rngs(params=keys[0]))

        # 2. Linear Decoder
        # Projects d_model -> 1 output value
        self.decoder = nnx.Linear(d_model, d_output, rngs=nnx.Rngs(params=keys[1]))

        layer_keys = jax.random.split(keys[2], n_layers)
        self.layers = []
        for i in range(n_layers):
            self.layers.append(
                SequenceBlockNNX(
                    layer_cls=layer_cls,
                    layer_args=layer_args,
                    d_model=d_model,
                    dropout=dropout,
                    prenorm=prenorm,
                    decode=decode,
                    glu=True,
                    rngs=nnx.Rngs(params=layer_keys[i])
                )
            )

    def __call__(self, x, states=None, *, rngs: nnx.Rngs = None, training: bool = True):
        # x shape: (B, L, 1) or (L, 1)

        # --- FIX 1: Handle Rank-1 Input ---
        was_1d = False
        if x.ndim == 1:
            x = x[jnp.newaxis, :]
            was_1d = True

        # # Causal Padding for CNN mode
        # if not self.decode:
        #     x = jnp.pad(x[:-1], [(1, 0), (0, 0)])

        # --- NO NORMALIZATION (Input is already standard float) ---

        x = self.encoder(x)
        current_states = states if states is not None else [None] * self.n_layers

        new_states = []
        for layer, state in zip(self.layers, current_states):
            x, new_s = layer(x, state, rngs=rngs, training=training)
            new_states.append(new_s)

        x = self.decoder(x)

        # --- FIX 2: NO SOFTMAX (Regression Output) ---
        output = x

        if was_1d:
            output = output.squeeze(0)

        return output, new_states

    # Add the init_state helper for inference
    def init_state(self, N: int):
        return [jnp.zeros((self.d_model, N), dtype=jnp.complex64) for _ in range(self.n_layers)]




# 1. VMAP RUNNERS (The "BatchStackedModel" replacement)

batched_reg_runner = nnx.vmap(
    lambda m, x, k, is_train: m(x, states=None, rngs=nnx.Rngs(dropout=k), training=is_train),
    in_axes=(nnx.StateAxes({nnx.Param: None}), 0, 0, None),
    out_axes=0
)
