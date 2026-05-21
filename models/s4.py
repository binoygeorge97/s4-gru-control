import jax
from jax.numpy.linalg import inv, matrix_power
from jax.nn.initializers import normal, zeros, ones
import flax.nnx as nnx
import optax

# --- Helper Functions (JAX-compatible) ---

def scan_SSM(Ab, Bb, Cb, u, x0):
    """Run the SSM state-space equation."""
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)



def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """Initializer for the log_step parameter."""
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            jnp.log(dt_max) - jnp.log(dt_min)
        ) + jnp.log(dt_min)
    return init


def causal_convolution(u, K):
    #jax.debug.print("DEBUG: u shape={} | K shape={}", u.shape, K.shape)
    #print("DEBUG: u shape={} | K shape={}", u.shape, K.shape)
    assert K.shape[0] == u.shape[0]
    ud = jnp.fft.rfft(jnp.pad(u, (0, K.shape[0])))
    Kd = jnp.fft.rfft(jnp.pad(K, (0, u.shape[0])))
    out = ud * Kd
    return jnp.fft.irfft(out)[: u.shape[0]]

def hippo_initializer(N):
    Lambda, P, B, _ = make_DPLR_HiPPO(N)
    return init(Lambda.real), init(Lambda.imag), init(P), init(B)


def init(x):
    def _init(key, shape):
        assert shape == x.shape
        return x

    return _init


def make_DPLR_HiPPO(N):
    """Diagonalize NPLR representation"""
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, jnp.newaxis] * P[jnp.newaxis, :]

    # Check skew symmetry
    S_diag = jnp.diagonal(S)
    Lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)
    # assert np.allclose(Lambda_real, S_diag, atol=1e-3)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = jnp.linalg.eigh(S * -1j)

    P = V.conj().T @ P
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V


def make_NPLR_HiPPO(N):
    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = jnp.sqrt(jnp.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    return nhippo, P, B


def make_HiPPO(N):
    P = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = P[:, jnp.newaxis] * P[jnp.newaxis, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    return -A

@jax.jit
def cauchy(v, omega, lambd):
    """Cauchy matrix multiplication: (n), (l), (n) -> (l)"""
    cauchy_dot = lambda _omega: (v / (_omega - lambd)).sum()
    return jax.vmap(cauchy_dot)(omega)


def kernel_DPLR(Lambda, P, Q, B, C, step, L):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = jnp.exp((-2j * jnp.pi) * (jnp.arange(L) / L))

    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = jnp.fft.ifft(atRoots, L).reshape(L)
    return out.real


def discrete_DPLR(Lambda, P, Q, B, C, step, L):
    # Convert parameters to matrices
    B = B[:, jnp.newaxis]
    Ct = C[jnp.newaxis, :]

    N = Lambda.shape[0]
    A = jnp.diag(Lambda) - P[:, jnp.newaxis] @ Q[:, jnp.newaxis].conj().T
    I = jnp.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = jnp.diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().T.reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()



class S4LayerEnsemble(nnx.Module):
    def __init__(self, N: int, l_max: int, D_MODEL: int, decode: bool, *, rngs: nnx.Rngs):
        self.N, self.decode, self.l_max, self.D_MODEL = N, decode, l_max, D_MODEL
        init_A_re, init_A_im, init_P, init_B = hippo_initializer(self.N)
        init_C, init_D, init_log_step = normal(stddev=0.5**0.5), ones, log_step_initializer()
        vmap_in_axes = (0, None)
        vmap_init_A_re = jax.vmap(init_A_re, in_axes=vmap_in_axes)
        vmap_init_A_im = jax.vmap(init_A_im, in_axes=vmap_in_axes)
        vmap_init_P = jax.vmap(init_P, in_axes=vmap_in_axes)
        vmap_init_B = jax.vmap(init_B, in_axes=vmap_in_axes)
        vmap_init_C = jax.vmap(init_C, in_axes=vmap_in_axes)
        vmap_init_D = jax.vmap(init_D, in_axes=vmap_in_axes)
        vmap_init_log_step = jax.vmap(init_log_step, in_axes=vmap_in_axes)
        keys = jax.random.split(rngs.params(), 7)
        lr_meta = {'lr': 0.1}
        self.Lambda_re = nnx.Param(vmap_init_A_re(jax.random.split(keys[0], D_MODEL), (N,)), metadata=lr_meta)
        self.Lambda_im = nnx.Param(vmap_init_A_im(jax.random.split(keys[1], D_MODEL), (N,)), metadata=lr_meta)
        self.P = nnx.Param(vmap_init_P(jax.random.split(keys[2], D_MODEL), (N,)), metadata=lr_meta)
        self.B = nnx.Param(vmap_init_B(jax.random.split(keys[3], D_MODEL), (N,)), metadata=lr_meta)
        self.C_real_imag = nnx.Param(vmap_init_C(jax.random.split(keys[4], D_MODEL), (N, 2)), metadata=lr_meta)
        self.D = nnx.Param(vmap_init_D(jax.random.split(keys[5], D_MODEL), (1,)), metadata=lr_meta)
        self.log_step = nnx.Param(vmap_init_log_step(jax.random.split(keys[6], D_MODEL), (1,)), metadata=lr_meta)

        # --- NO MORE self.x_k_1 ---
        # if self.decode:
        #     self.x_k_1 = nnx.Variable(jnp.zeros((D_MODEL, N,), dtype=jnp.complex64))

    # --- __call__ signature has changed ---
    def __call__(self, u, x_k_1):
        """
        Takes in a single state vector x_k_1 [N,]
        Returns a single output y_s [L,] and new state x_k [N,]
        """
        dt_min, dt_max = 0.001, 1.0
        step = jnp.exp(self.log_step.value)
        step = jnp.clip(step, dt_min, dt_max)

        Lambda = jnp.clip(self.Lambda_re.value, None, -1e-4) + 1j * self.Lambda_im.value
        C_complex = self.C_real_imag.value[..., 0] + 1j * self.C_real_imag.value[..., 1]
        #step = jnp.exp(self.log_step.value)

        if not self.decode:
            # CNN mode is stateless, so we ignore x_k_1 and return it unchanged
            K = kernel_DPLR(Lambda, self.P.value, self.P.value, self.B.value, C_complex, step, self.l_max)
            y_s = causal_convolution(u, K) + self.D.value * u
            return y_s, x_k_1 # Return state unchanged
        else:
            # RNN mode uses and returns state
            Ab, Bb, Cb = discrete_DPLR(Lambda, self.P.value, self.P.value, self.B.value, C_complex, step, self.l_max)
            u_r = u[:, jnp.newaxis]
            x_k, y_s = scan_SSM(Ab, Bb, Cb, u_r, x_k_1) # Use passed-in state

            # --- DO NOT MUTATE SELF ---
            # self.x_k_1.value = x_k

            # --- Return the output and the new state ---
            return y_s.reshape(-1).real + self.D.value * u, x_k
