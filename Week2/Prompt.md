# Role: Expert in Probabilistic Graphical Models and Variational Inference
You are a senior AI researcher assisting a student in implementing a Hidden Markov Model (HMM) solved via Mean-Field Variational Inference (VI), connecting to Statistical Mechanics concepts (Variational Free Energy).

# Core Constraint
**DO NOT use any existing HMM libraries (like hmmlearn).** You must implement the data generation, Variational Free Energy calculation, and Mean-Field coordinate ascent updates from scratch using `numpy`, `scipy.stats`, and `matplotlib`.

# Task Breakdown
Please write a single, cohesive Python script that sequentially accomplishes the following 4 phases. Add clear comments separating each phase.

## Phase 1: Context & Simulating the System
1. Define 2 hidden states: `0` ('Stable') and `1` ('Unstable').
2. Time steps: `T = 200`.
3. Transition dynamics: 90% chance to stay in the current state, 10% chance to transition. (Define a 2x2 transition probability matrix).
4. Emission distributions (Normal distribution):
   - 'Stable' (state 0): mean = 0, std = 1.
   - 'Unstable' (state 1): mean = 0, std = 5.
5. Generate the ground truth hidden states sequence `Z` and the observed noisy sensor data `X`.
6. **Plotting**: Plot the observed time series `X`. Use `axvspan` to shade the background color based on the true hidden state `Z` (e.g., light blue for Stable, light red for Unstable).

## Phase 2: Formulating the Variational Free Energy View
1. Define a function `log_emission_probs(X)` that returns a `T x 2` matrix containing the log-probability of each observation under both states using Gaussian PDFs.
2. Define a function `calculate_free_energy(q, X, transition_matrix)` where `q` is the `T x 2` array of beliefs (probabilities summing to 1 for each time step).
   - *Concept*: Variational Free Energy (F) = Expected Energy - Entropy = -ELBO.
   - Expected Energy = Expected log-likelihood of emissions + Expected log-likelihood of transitions.
   - Entropy = $\sum_{t} \sum_{k} -q_t(k) \log q_t(k)$.
   - The function should return a scalar value representing the total Free Energy.

## Phase 3: Mean-Field Belief Updates (Message Passing)
1. Initialize the belief matrix `q` (`T x 2`) randomly. Ensure each row sums to 1.
2. Implement Coordinate Ascent for Mean-Field VI. Run for `num_iterations = 20`.
3. Inside the loop, iterate through time steps `t` from 1 to T-2 (handle boundaries appropriately or loop 0 to T-1 with edge case handling).
4. Update rule for $q_t(k)$ (log-space before softmax):
   $\log q_t(k) \propto \log P(x_t | z_t=k) + \sum_{j} q_{t-1}(j) \log P(z_t=k | z_{t-1}=j) + \sum_{l} q_{t+1}(l) \log P(z_{t+1}=l | z_t=k)$
   *Hint: The physical analogy is a 1D chain where neighbors $t-1$ and $t+1$ exert a "pull" based on the transition matrix, and the local observation acts as a local magnetic field.*
5. Apply softmax to normalize $q_t$ to probabilities.
6. Store the calculated Free Energy at the end of each iteration in a list. Also, store a copy of the belief matrix `q` at each iteration for later animation.

## Phase 4: Visualization and Convergence Analysis
1. Create a figure with 2 subplots.
2. **Subplot 1 (Convergence)**: Plot the Variational Free Energy over the 20 iterations. It MUST show a monotonically decreasing trend.
3. **Subplot 2 (Belief Animation)**: Use `matplotlib.animation.FuncAnimation` to animate the beliefs over the 20 iterations.
   - The x-axis is time `t` (0 to 200).
   - The y-axis is the probability `q_t(state=Unstable)`.
   - Plot the true hidden states as a static faint background line (0 for Stable, 1 for Unstable) for comparison.
   - Show how the beliefs `q` "snap" into confidently predicting the regimes as iterations progress.
4. Save the animation as a `.gif` or display it using `plt.show()`.

Ensure the code is robust, well-commented, and mathematically aligns with the Variational Inference / Evidence Lower Bound (ELBO) principles described in the prompt.