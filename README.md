# Reinforcement Learning

All my programs related to reinforcement learning are uploaded into this repository.

## Folder Structure

- `CapAM-MRTA-main` - Contains the CapAM-MRTA project files.
- `Dynamic Programming` - Files related to dynamic programming techniques in reinforcement learning.
- `Lunar Lander.py` - A Python script implementing reinforcement learning for the Lunar Lander environment.
- `Temporal Difference` - Files related to temporal difference methods in reinforcement learning.
- `Discretization` - Files and scripts for state space discretization techniques, which involve dividing the continuous state space into discrete buckets to make learning manageable for certain algorithms.
- `Frozen_Lake.ipynb` - A Jupyter Notebook implementing reinforcement learning for the Frozen Lake environment.
- `Monte_Carlo` - Files related to Monte Carlo methods in reinforcement learning.
- `Reference` - Reference materials and documentation.
- `Lunar Lander` - Files related to Deep Q-Network (DQN) implementations.
  - **Deep Q-Network (DQN):** DQN extends Q-learning by using deep neural networks to approximate the action-value function. This approach enables solving complex environments with high-dimensional state spaces, such as the Lunar Lander environment. [Result Video Link](https://drive.google.com/file/d/1J5dgN0O1mKSbpeAJtTyug-yA3CbFKC_6/view?usp=sharing)
- `REINFORCE` - Implements the REINFORCE algorithm, a policy gradient method that optimizes the policy directly by maximizing the expected reward. This is particularly suited for environments with high variance in rewards.

## Algorithm Details

### Deep Q-Network (DQN)
DQN uses a deep neural network to approximate the Q-function, which maps state-action pairs to their expected rewards. Key features include:
- **Experience Replay:** Stores past experiences and samples them randomly to break correlation during training.
- **Target Network:** Maintains a separate network to stabilize updates by providing consistent target Q-values.
- **Applications:** Works well in discrete action spaces, such as Atari games and the Lunar Lander environment.

### REINFORCE
REINFORCE is a Monte Carlo-based policy gradient algorithm that directly optimizes the policy by sampling complete episodes and updating the policy weights using the gradient of the expected return:
- **Advantages:** Works well in continuous action spaces and directly optimizes stochastic policies.
- **Applications:** Suitable for environments with complex policies like robotic control.

### Discretization
Discretization involves converting continuous state spaces into discrete bins, enabling algorithms like Q-Learning or SARSA to operate effectively in environments where states are otherwise challenging to represent:
- **Techniques:** Uniform binning, adaptive binning, and clustering-based methods.
- **Applications:** Useful for problems like mountain car and cart-pole, where state variables are continuous.

## Dependencies

Ensure you have the following dependencies installed:
- Python 3.x
- Jupyter Notebook
- torch
- gym
- numpy
- matplotlib

```python
pip install torch gym numpy matplotlib
