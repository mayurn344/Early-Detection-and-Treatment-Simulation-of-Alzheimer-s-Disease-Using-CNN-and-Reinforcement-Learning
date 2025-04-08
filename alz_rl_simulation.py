from alz_rl_env import AlzheimerEnv
import numpy as np
import matplotlib.pyplot as plt

# Q-learning parameters
episodes = 500
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0  # for exploration
epsilon_decay = 0.995
min_epsilon = 0.01

env = AlzheimerEnv()

# Q-table [states x actions]
q_table = np.zeros((4, 3))  # 4 stages, 3 actions

reward_per_episode = []

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1, 2])  # explore
        else:
            action = np.argmax(q_table[state])    # exploit best

        next_state, reward, done = env.step(action)

        # Q-value update
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
        q_table[state, action] = new_value

        state = next_state
        total_reward += reward

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    reward_per_episode.append(total_reward)

    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1}: Total Reward = {total_reward}, Epsilon = {epsilon:.3f}")

print("\n🏁 Training finished!")
print("🧠 Learned Q-table:")
print(q_table)
class_names = ['CN', 'EMCI', 'LMCI', 'AD']
print("\n🧠 Learned Q-table (Rows = States, Cols = Actions [No, Mild, Strong Treatment]):")
for i, row in enumerate(q_table):
    print(f"{class_names[i]}: {row}")

# Plot rewards
plt.plot(reward_per_episode)
plt.title("Episode Reward over Time")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plt.show()
