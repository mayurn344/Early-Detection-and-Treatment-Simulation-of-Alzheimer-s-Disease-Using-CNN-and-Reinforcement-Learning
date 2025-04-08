import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
from alz_rl_env import AlzheimerEnv

# Optional: import your trained model if you have one
# from model import predict_class

# Set up Streamlit page
st.set_page_config(page_title="Alzheimer's Progression Q-Learning", layout="centered")

st.title("🧠 Alzheimer's Progression Simulation using Q-Learning")
st.write("This simulation models disease progression from CN → EMCI → LMCI → AD with different treatments using Q-learning.")

# 🖼️ Image Upload Section
st.subheader("📷 Upload Brain Image for Classification (Optional)")
uploaded_image = st.file_uploader("Upload a brain scan image (e.g., MRI)", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Brain Image", use_column_width=True)

    # Dummy prediction function (replace with real model)
    def predict_class(img):
        # Example: returns one of the 4 states randomly
        return np.random.choice(["CN", "EMCI", "LMCI", "AD"])

    prediction = predict_class(image)
    st.success(f"🧠 Predicted Disease Stage: **{prediction}**")
else:
    st.info("Upload a brain scan image to detect disease stage (optional).")

# Q-learning parameters
episodes = st.slider("Select number of episodes:", 100, 1000, 500, step=100)
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

# Q-Learning Simulation
if st.button("Run Q-Learning Simulation"):
    env = AlzheimerEnv()
    q_table = np.zeros((4, 3))  # States x Actions
    reward_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice([0, 1, 2])  # Explore
            else:
                action = np.argmax(q_table[state])    # Exploit

            next_state, reward, done = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
            q_table[state, action] = new_value

            state = next_state
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        reward_per_episode.append(total_reward)

    st.success("✅ Training Completed!")

    st.subheader("📈 Rewards per Episode")
    fig, ax = plt.subplots()
    ax.plot(reward_per_episode)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Over Time")
    st.pyplot(fig)

    st.subheader("🧠 Learned Q-Table")
    actions = ["No Treatment", "Mild", "Strong"]
    states = ["CN", "EMCI", "LMCI", "AD"]

    q_table_rounded = np.round(q_table, 2)
    st.table(
        pd.DataFrame(q_table_rounded, columns=actions, index=states)
    )
