# Exp-5 Implementation of On Policy Monte Carlo Control using Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description







## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm

1. Initialize the policy π and action-value function Q(s,a) arbitrarily.
2. Generate an episode by following the current policy until the terminal state.
3. Calculate the return G for each state-action pair using the obtained rewards.
4. Update Q(s,a) by averaging the observed returns.
5. Evaluate the current policy using the updated Q values.
6. Improve the policy by selecting the action with the highest Q(s,a) value.
7. Repeat Steps 2–6 until the policy becomes stable or converges.

## Python Program

-------------------------------------------------
#### Monte Carlo Control


```py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
```

```py
# -------------------------------------------------
# Create Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)

n_states = env.observation_space.n
n_actions = env.action_space.n

print("Number of states:", n_states)
print("Number of actions:", n_actions)
```

```py

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 20000
gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100

```

```py
# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))
episode_rewards = []


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

# Write your code here

def epsilon_greedy_action(state, epsilon):
    """
    Selects an action using epsilon-greedy policy.
    """
    
    # Exploration
    if np.random.random() < epsilon:
        return env.action_space.sample()
    
    # Exploitation
    return np.argmax(Q[state])


```

```py
# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------

def generate_episode(epsilon):
    """
    Generates one episode using the current epsilon-greedy policy.
    Returns a list of (state, action, reward).
    """

    episode = []

    state, info = env.reset()

    for _ in range(max_steps_per_episode):
        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated or truncated:
            break

    return episode


```

```py
# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

epsilon = epsilon_start

for episode in range(num_episodes):

    # Generate a complete episode
    episode_data = generate_episode(epsilon)

    # Store total reward
    total_reward = sum(
        reward for state, action, reward in episode_data
    )

    episode_rewards.append(total_reward)

    # Calculate returns backwards
    G = 0.0

    for state, action, reward in reversed(episode_data):

        G = gamma * G + reward

        # Incremental Monte Carlo update
        Q[state, action] = Q[state, action] + \
            alpha * (G - Q[state, action])

    # Decay epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    # Display progress
    if (episode + 1) % 5000 == 0:
        avg_reward = np.mean(episode_rewards[-1000:])
        print(
            f"Episode {episode + 1}/{num_episodes}, "
            f"Epsilon: {epsilon:.4f}, "
            f"Average Reward: {avg_reward:.3f}"
        )
```

```py
# -------------------------------------------------
# Extract Greedy Policy
# -------------------------------------------------

optimal_policy = np.argmax(Q, axis=1)
state_values = np.max(Q, axis=1)

```

```py
# -------------------------------------------------
# Display Results
# -------------------------------------------------

def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)
    print("Name: Loknaath P")
    print("Register Number: 212223240080")
    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(optimal_policy)

success_rate = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", success_rate)


```

```py
# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500
moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Monte Carlo Control Learning Curve")
plt.grid(True)
plt.show()

env.close()
```

---

## Output
### Episodes=20000

#### Final Q-table:
<img width="376" height="357" alt="image" src="https://github.com/user-attachments/assets/d8304a86-3456-4bfd-9218-ebde1212689e" />



#### Estimated State-Value Function:
<img width="423" height="156" alt="image" src="https://github.com/user-attachments/assets/078dc8d4-5e44-4b29-bc3f-ab5bf7c7c1be" />


#### Learned Policy:
<img width="316" height="116" alt="image" src="https://github.com/user-attachments/assets/0a23c373-e722-4630-b374-a4d284525320" />


#### Average reward over last 1000 episodes: 
<img width="480" height="51" alt="image" src="https://github.com/user-attachments/assets/684249f0-4a9b-4fc1-9795-33b5d6373e95" />

#### Plot Learning Curve:
<img width="817" height="552" alt="image" src="https://github.com/user-attachments/assets/086fc822-7046-413e-ad70-96862fcc96d5" />



### Episodes=5000

#### Final Q-table:
<img width="335" height="352" alt="image" src="https://github.com/user-attachments/assets/23eb9a08-31e6-4724-8863-f5d05870afeb" />


#### Estimated State-Value Function:
<img width="358" height="155" alt="image" src="https://github.com/user-attachments/assets/9b8f03ee-69ee-4daf-b6ff-8f409c7875f8" />


#### Learned Policy:
<img width="348" height="112" alt="image" src="https://github.com/user-attachments/assets/8ff2b9ed-ae83-4b17-9d09-e96606ecd7df" />


#### Average reward over last 1000 episodes: 
<img width="515" height="48" alt="image" src="https://github.com/user-attachments/assets/f0f08144-49c8-44c7-8e02-be314a5d1cfe" />


#### Plot Learning Curve:
<img width="831" height="561" alt="image" src="https://github.com/user-attachments/assets/c6799244-a29d-4ace-add6-8539292cb2d1" />


---

## Result
Monte Carlo methods successfully estimated the action values Q(s,a) and improved the policy through repeated episode sampling.


---

## Inference
```text
1. The Monte Carlo method successfully improves the policy over 5,000 training episodes.
2. The average reward increases steadily from around 0.05 to approximately 0.90.
3. The increasing reward indicates that the agent gradually learns better actions through repeated episodes.
4. The learning curve becomes more stable toward the later episodes, showing improved policy performance.
5. Small fluctuations in the curve indicate continued exploration and variation in episode outcomes.
6. The results demonstrate that more training episodes help Monte Carlo control achieve better action-value estimates.
7. Overall, 5,000 episodes provide sufficient training to show clear learning and convergence toward a high-performing policy.

```





---

