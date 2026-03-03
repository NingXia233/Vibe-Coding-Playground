"""
Hidden Markov Model (HMM) solved via Mean-Field Variational Inference (VI)
Connects to Statistical Mechanics concepts (Variational Free Energy).

No external HMM libraries are used. Everything is implemented from scratch
using numpy, scipy.stats, and matplotlib.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Fix random seed for reproducibility
np.random.seed(42)

# ============================================================================
# Phase 1: Context & Simulating the System
# ============================================================================
print("=" * 60)
print("Phase 1: Simulating the HMM System")
print("=" * 60)

# --- 1.1 Define hidden states ---
# State 0: 'Stable', State 1: 'Unstable'
state_names = {0: 'Stable', 1: 'Unstable'}
num_states = 2

# --- 1.2 Time steps ---
T = 200

# --- 1.3 Transition dynamics ---
# 90% chance to stay in the current state, 10% chance to transition
# A[i, j] = P(z_{t+1} = j | z_t = i)
A = np.array([
    [0.9, 0.1],  # From Stable:   90% stay Stable, 10% go Unstable
    [0.1, 0.9],  # From Unstable: 10% go Stable,   90% stay Unstable
])

# --- 1.4 Emission distributions (Gaussian) ---
# State 0 ('Stable'):   mean=0, std=1
# State 1 ('Unstable'): mean=0, std=5
emission_means = np.array([0.0, 0.0])
emission_stds = np.array([1.0, 5.0])

# --- 1.5 Generate ground truth hidden states Z and observations X ---
# Initial state: start from Stable (state 0)
Z = np.zeros(T, dtype=int)
X = np.zeros(T)

# Sample initial state uniformly
Z[0] = np.random.choice([0, 1])
X[0] = np.random.normal(emission_means[Z[0]], emission_stds[Z[0]])

for t in range(1, T):
    # Sample next hidden state from transition distribution
    Z[t] = np.random.choice([0, 1], p=A[Z[t - 1]])
    # Sample observation from emission distribution
    X[t] = np.random.normal(emission_means[Z[t]], emission_stds[Z[t]])

print(f"Generated {T} time steps.")
print(f"Number of Stable steps:   {np.sum(Z == 0)}")
print(f"Number of Unstable steps: {np.sum(Z == 1)}")

# --- 1.6 Plot the observed time series with true state shading ---
fig1, ax1 = plt.subplots(figsize=(14, 4))
ax1.plot(np.arange(T), X, color='black', linewidth=0.8, label='Observed X')

# Shade background based on true hidden state Z
for t in range(T):
    if Z[t] == 0:
        ax1.axvspan(t - 0.5, t + 0.5, color='lightblue', alpha=0.4)
    else:
        ax1.axvspan(t - 0.5, t + 0.5, color='lightcoral', alpha=0.4)

# Add legend patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightblue', alpha=0.6, label='Stable (state 0)'),
    Patch(facecolor='lightcoral', alpha=0.6, label='Unstable (state 1)'),
    plt.Line2D([0], [0], color='black', linewidth=0.8, label='Observed X'),
]
ax1.legend(handles=legend_elements, loc='upper right')
ax1.set_xlabel('Time step t')
ax1.set_ylabel('Observation X')
ax1.set_title('Phase 1: Observed Time Series with True Hidden States')
ax1.set_xlim(-1, T)
plt.tight_layout()
plt.savefig('phase1_observations.png', dpi=150)
plt.show()

# ============================================================================
# Phase 2: Formulating the Variational Free Energy View
# ============================================================================
print("\n" + "=" * 60)
print("Phase 2: Defining Free Energy Functions")
print("=" * 60)


def log_emission_probs(X):
    """
    Compute log P(x_t | z_t = k) for all t and k.
    
    Parameters
    ----------
    X : array of shape (T,)
        Observed data.
    
    Returns
    -------
    log_emit : array of shape (T, 2)
        log_emit[t, k] = log P(x_t | z_t = k) using Gaussian PDF.
    """
    T = len(X)
    log_emit = np.zeros((T, num_states))
    for k in range(num_states):
        # Use scipy.stats.norm to compute log-pdf
        log_emit[:, k] = norm.logpdf(X, loc=emission_means[k], scale=emission_stds[k])
    return log_emit


def calculate_free_energy(q, X, transition_matrix):
    """
    Calculate the Variational Free Energy F = -ELBO = Expected Energy - Entropy.
    
    F = - [ E_q[log P(X, Z)] + H(q) ]
      = - E_q[log P(X, Z)] - H(q)
      = E_q[-log P(X, Z)] - H(q)
    
    But more conventionally:
    F = -ELBO = -( E_q[log P(X,Z)] - E_q[log q(Z)] )
              = -E_q[log P(X,Z)] + E_q[log q(Z)]
              = <Energy> - Entropy  (in physics convention, F = E - TS with T=1)
    
    Where:
      - E_q[log P(X,Z)] = sum of expected log-emission + expected log-transition
      - H(q) = -E_q[log q(Z)] = sum_t sum_k -q_t(k) log q_t(k)
    
    Parameters
    ----------
    q : array of shape (T, 2)
        Variational beliefs. q[t, k] = probability that z_t = k.
    X : array of shape (T,)
        Observed data.
    transition_matrix : array of shape (2, 2)
        A[i, j] = P(z_{t+1}=j | z_t=i).
    
    Returns
    -------
    F : float
        The variational free energy (negative ELBO).
    """
    T_len = len(X)
    log_emit = log_emission_probs(X)
    log_A = np.log(transition_matrix + 1e-300)  # avoid log(0)
    
    # --- Expected log-likelihood of emissions ---
    # E_q[log P(X|Z)] = sum_t sum_k q_t(k) * log P(x_t | z_t=k)
    expected_log_emission = np.sum(q * log_emit)
    
    # --- Expected log-likelihood of transitions ---
    # E_q[log P(Z)] = sum_{t=1}^{T-1} sum_j sum_k q_{t-1}(j) * q_t(k) * log A[j,k]
    # Under mean-field factorization, z_{t-1} and z_t are independent in q,
    # so E_q[f(z_{t-1}, z_t)] = sum_{j,k} q_{t-1}(j) q_t(k) f(j,k)
    expected_log_transition = 0.0
    for t in range(1, T_len):
        # Outer product of q[t-1] and q[t] gives joint under mean-field approx.
        joint = np.outer(q[t - 1], q[t])  # shape (2, 2)
        expected_log_transition += np.sum(joint * log_A)
    
    # --- Entropy of q ---
    # H(q) = sum_t sum_k -q_t(k) * log q_t(k)
    # Use clipping to avoid log(0)
    q_safe = np.clip(q, 1e-300, 1.0)
    entropy = -np.sum(q * np.log(q_safe))
    
    # --- Free Energy ---
    # F = -ELBO = -(E_q[log P(X,Z)] + H(q))
    # E_q[log P(X,Z)] = expected_log_emission + expected_log_transition
    F = -(expected_log_emission + expected_log_transition + entropy)
    
    return F


# Test the functions
log_emit = log_emission_probs(X)
print(f"Log emission probs shape: {log_emit.shape}")
print(f"Sample log_emit[0]: {log_emit[0]}  (observation X[0] = {X[0]:.3f})")

# Test free energy with uniform q
q_uniform = np.ones((T, num_states)) / num_states
F_uniform = calculate_free_energy(q_uniform, X, A)
print(f"Free Energy with uniform beliefs: {F_uniform:.2f}")

# ============================================================================
# Phase 3: Mean-Field Belief Updates (Coordinate Ascent)
# ============================================================================
print("\n" + "=" * 60)
print("Phase 3: Mean-Field Coordinate Ascent VI")
print("=" * 60)

# --- 3.1 Initialize belief matrix q randomly ---
q = np.random.dirichlet(alpha=[1, 1], size=T)  # Each row sums to 1

# Precompute log emission probabilities (they don't change)
log_emit = log_emission_probs(X)
log_A = np.log(A + 1e-300)

# --- 3.2 Coordinate Ascent ---
num_iterations = 20
free_energy_history = []
q_history = []  # Store q at each iteration for animation

# Store initial state
free_energy_history.append(calculate_free_energy(q, X, A))
q_history.append(q.copy())
print(f"Iteration  0 | Free Energy: {free_energy_history[-1]:.4f}")

for iteration in range(1, num_iterations + 1):
    # --- 3.3 & 3.4 Iterate through time steps and update beliefs ---
    for t in range(T):
        # Log-unnormalized belief for each state k
        log_q_unnorm = np.zeros(num_states)
        
        for k in range(num_states):
            # Term 1: Local evidence (log-emission)
            log_q_unnorm[k] = log_emit[t, k]
            
            # Term 2: Message from the past neighbor (t-1 -> t)
            if t > 0:
                # sum_j q_{t-1}(j) * log P(z_t=k | z_{t-1}=j)
                msg_past = np.sum(q[t - 1, :] * log_A[:, k])
                log_q_unnorm[k] += msg_past
            
            # Term 3: Message from the future neighbor (t+1 <- t)
            if t < T - 1:
                # sum_l q_{t+1}(l) * log P(z_{t+1}=l | z_t=k)
                msg_future = np.sum(q[t + 1, :] * log_A[k, :])
                log_q_unnorm[k] += msg_future
        
        # --- 3.5 Softmax normalization ---
        # Subtract max for numerical stability before exponentiation
        log_q_unnorm -= np.max(log_q_unnorm)
        q_new = np.exp(log_q_unnorm)
        q_new /= np.sum(q_new)
        
        # Update belief at time t
        q[t] = q_new
    
    # --- 3.6 Record free energy and beliefs ---
    F = calculate_free_energy(q, X, A)
    free_energy_history.append(F)
    q_history.append(q.copy())
    
    print(f"Iteration {iteration:2d} | Free Energy: {F:.4f}")

print(f"\nFree energy decreased from {free_energy_history[0]:.4f} to {free_energy_history[-1]:.4f}")

# ============================================================================
# Phase 4: Visualization and Convergence Analysis
# ============================================================================
print("\n" + "=" * 60)
print("Phase 4: Visualization and Animation")
print("=" * 60)

# --- 4.1 Create figure with 2 subplots ---
fig, (ax_conv, ax_belief) = plt.subplots(2, 1, figsize=(14, 8))

# --- 4.2 Subplot 1: Convergence of Free Energy ---
iterations_axis = np.arange(len(free_energy_history))
ax_conv.plot(iterations_axis, free_energy_history, 'o-', color='darkblue', linewidth=2, markersize=5)
ax_conv.set_xlabel('Iteration', fontsize=12)
ax_conv.set_ylabel('Variational Free Energy (F = -ELBO)', fontsize=12)
ax_conv.set_title('Convergence of Variational Free Energy', fontsize=14)
ax_conv.grid(True, alpha=0.3)

# Highlight monotonic decrease
for i in range(1, len(free_energy_history)):
    if free_energy_history[i] > free_energy_history[i - 1]:
        ax_conv.annotate('⚠ increase!', xy=(i, free_energy_history[i]),
                         fontsize=8, color='red')

# --- 4.3 Subplot 2: Animated beliefs ---
# Plot true hidden states as a faint background
ax_belief.step(np.arange(T), Z, where='mid', color='gray', alpha=0.3,
               linewidth=2, label='True state Z')
ax_belief.set_xlabel('Time step t', fontsize=12)
ax_belief.set_ylabel('q_t(Unstable)', fontsize=12)
ax_belief.set_title('Mean-Field Beliefs over Iterations', fontsize=14)
ax_belief.set_xlim(-1, T)
ax_belief.set_ylim(-0.05, 1.05)
ax_belief.grid(True, alpha=0.3)

# Initialize the belief line
line_belief, = ax_belief.plot([], [], color='crimson', linewidth=1.5, label='q(Unstable)')
iter_text = ax_belief.text(0.02, 0.95, '', transform=ax_belief.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax_belief.legend(loc='upper right')

plt.tight_layout()


def init_anim():
    """Initialize animation."""
    line_belief.set_data([], [])
    iter_text.set_text('')
    return line_belief, iter_text


def update_anim(frame):
    """Update animation for each frame (iteration)."""
    q_frame = q_history[frame]
    line_belief.set_data(np.arange(T), q_frame[:, 1])  # q_t(Unstable)
    iter_text.set_text(f'Iteration: {frame}')
    
    # Update convergence plot marker
    ax_conv.clear()
    ax_conv.plot(iterations_axis, free_energy_history, 'o-', color='darkblue',
                 linewidth=2, markersize=5)
    ax_conv.plot(frame, free_energy_history[frame], 'o', color='red',
                 markersize=10, zorder=5)
    ax_conv.set_xlabel('Iteration', fontsize=12)
    ax_conv.set_ylabel('Variational Free Energy (F = -ELBO)', fontsize=12)
    ax_conv.set_title('Convergence of Variational Free Energy', fontsize=14)
    ax_conv.grid(True, alpha=0.3)
    
    return line_belief, iter_text


# Create the animation
anim = animation.FuncAnimation(
    fig, update_anim, init_func=init_anim,
    frames=len(q_history), interval=500, blit=False, repeat=True
)

# --- 4.4 Save as GIF ---
try:
    anim.save('belief_animation.gif', writer='pillow', fps=2, dpi=100)
    print("Animation saved to belief_animation.gif")
except Exception as e:
    print(f"Could not save GIF: {e}. Displaying instead.")

plt.show()

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Initial Free Energy:  {free_energy_history[0]:.4f}")
print(f"Final Free Energy:    {free_energy_history[-1]:.4f}")
print(f"Free Energy decrease: {free_energy_history[0] - free_energy_history[-1]:.4f}")

# Check monotonic decrease
is_monotonic = all(
    free_energy_history[i] >= free_energy_history[i + 1] - 1e-6
    for i in range(len(free_energy_history) - 1)
)
print(f"Monotonically decreasing: {is_monotonic}")

# Classification accuracy using final beliefs
predicted_states = np.argmax(q_history[-1], axis=1)
accuracy = np.mean(predicted_states == Z)
print(f"Classification accuracy:  {accuracy:.2%}")