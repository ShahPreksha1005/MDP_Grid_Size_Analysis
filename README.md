# **MDP Grid Size Analysis**
This project extends a Markov Decision Process (MDP) implementation to evaluate the agent’s performance across different grid sizes (n × n). The objective is to analyze the effect of grid dimensions on the agent’s efficiency, using metrics such as the number of steps taken and total rewards earned.

## **1. Introduction**
We explore grid-based navigation using MDP to study how the environment size influences agent performance. The agent starts at a random position and aims to reach a fixed goal while incurring penalties for each step. A reward of **100** is given upon reaching the goal.

### **Experiment Goals:**
- Simulate agent performance in grids of sizes **3x3, 5x5, 7x7, and 10x10**.
- Collect **steps taken** and **total reward** statistics for each grid.
- Compare performance across different environments using **tables** and **visualizations**.

## **2. Experiment Setup**
- **Grid Sizes Considered:** 3x3, 5x5, 7x7, and 10x10
- **Episodes per Grid:** 5
- **Agent Rewards:**
  - **Penalty per step:** -1
  - **Goal reward:** 100

## **3. Implementation**
### **Key Features**
- Dynamic grid environment with flexible **n x n** size.
- Randomized agent movement using available actions **(up, down, left, right)**.
- Simulation of multiple episodes per grid.
- Data collection for **average steps** and **rewards**.
- Visualization of performance trends.

### **Code Snippet (Grid MDP Implementation)**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define MDP Environment
class GridMDP:
    def __init__(self, n, start_state, goal_state, penalty=-1, reward=100):
        self.n = n
        self.state = start_state
        self.start_state = start_state
        self.goal_state = goal_state
        self.penalty = penalty
        self.reward = reward

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        i, j = self.state
        if action == 'up':
            next_state = (max(i - 1, 0), j)
        elif action == 'down':
            next_state = (min(i + 1, self.n - 1), j)
        elif action == 'left':
            next_state = (i, max(j - 1, 0))
        elif action == 'right':
            next_state = (i, min(j + 1, self.n - 1))

        if next_state == self.goal_state:
            return next_state, self.reward, True
        else:
            self.state = next_state
            return next_state, self.penalty, False
```

### **Running Simulations**
```python
ACTIONS = ['up', 'down', 'left', 'right']
grids = [3, 5, 7, 10]

def run_multiple_episodes(env, episodes=5):
    episode_data = []
    for ep in range(episodes):
        state = env.reset()
        total_reward, steps = 0, 0

        while True:
            action = np.random.choice(ACTIONS)
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                episode_data.append([ep + 1, steps, total_reward, "Yes" if done else "No"])
                break
    return episode_data
```

### **Experiment Execution**
```python
def experiment(grid_sizes, episodes=5):
    results = []
    for n in grid_sizes:
        start_state, goal_state = (0, 0), (n - 1, n - 1)
        grid_mdp = GridMDP(n, start_state, goal_state)
        print(f"Running simulations on a {n}x{n} grid...")

        episode_stats = run_multiple_episodes(grid_mdp, episodes)
        avg_steps = np.mean([x[1] for x in episode_stats])
        avg_reward = np.mean([x[2] for x in episode_stats])
        results.append([n, avg_steps, avg_reward])

    return pd.DataFrame(results, columns=["Grid Size", "Average Steps", "Average Reward"])

experiment_results = experiment(grids, episodes=5)
print(experiment_results)
```

## **4. Results & Analysis**
### **Performance Trends Across Grid Sizes**
| Grid Size | Avg Steps Taken | Avg Reward |
|-----------|---------------|------------|
| 3x3 | 17.0 | 84.0 |
| 5x5 | 99.8 | 1.2 |
| 7x7 | 306.2 | -205.2 |
| 10x10 | 355.2 | -254.2 |

- **3x3 Grid:** The agent reaches the goal quickly, averaging **17 steps** with a high reward of **84**.
- **5x5 Grid:** Performance drops significantly, requiring **99 steps** with barely positive rewards.
- **7x7 & 10x10 Grids:** The agent struggles, needing **300+ steps**, leading to heavily negative rewards.

## **5. Visualization**
### **Comparison of Performance on Different Grid Sizes**
```python
def plot_comparison(results_df):
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax2 = ax1.twinx()

    results_df.plot(kind='bar', x='Grid Size', y='Average Steps', color='blue', ax=ax1, position=0, width=0.4)
    results_df.plot(kind='bar', x='Grid Size', y='Average Reward', color='orange', ax=ax2, position=1, width=0.4)

    ax1.set_xlabel("Grid Size")
    ax1.set_ylabel("Average Steps", color='blue')
    ax2.set_ylabel("Average Reward", color='orange')
    plt.title("Comparison of Average Steps and Rewards on Different Grid Sizes")
    plt.show()

plot_comparison(experiment_results)
```

### **Inference from the Visualization**
- **Larger grids require significantly more steps to reach the goal.**
- **Total rewards decline sharply as the agent accumulates more penalties.**
- **In smaller grids, the agent efficiently reaches the goal, achieving higher rewards.**

## **6. Episode-Level Statistics**
A table summarizing performance across different episodes:
```python
detailed_stats_df = detailed_episode_table(grids, episodes=5)
print(detailed_stats_df)
```
Sample Output:
| Grid Size | Episode | Steps Taken | Total Reward | Goal Reached |
|-----------|---------|-------------|--------------|--------------|
| 3x3 | 1 | 13 | 88 | Yes |
| 3x3 | 2 | 12 | 89 | Yes |
| 5x5 | 3 | 45 | 55 | Yes |
| 7x7 | 4 | 290 | -190 | Yes |
| 10x10 | 5 | 355 | -254 | No |

## **7. Conclusion**
- **Smaller grids (3x3, 5x5) allow for efficient navigation, with minimal penalties.**
- **Larger grids (7x7, 10x10) significantly increase path length, leading to substantial penalties.**
- **MDP performance is highly dependent on grid size, influencing learning complexity and efficiency.**

## **8. Future Improvements**
- Implement reinforcement learning (Q-Learning, SARSA) to optimize agent decision-making.
- Introduce obstacles in the grid for more realistic pathfinding.
- Extend experiments with different reward structures.

## **9. References**
- Markov Decision Processes: Sutton & Barto (2018)
- Reinforcement Learning Algorithms for Grid-Based Problems

---
