import numpy as np
import random

class AlzheimerEnv:
    def __init__(self):
        # 0: CN, 1: EMCI, 2: LMCI, 3: AD
        self.states = [0, 1, 2, 3]
        self.current_state = random.choice(self.states[:-1])  # Don't start from AD
        self.done = False

    def reset(self):
        self.current_state = random.choice(self.states[:-1])
        self.done = False
        return self.current_state

    def step(self, action):
        """
        Action:
        0 - No Treatment
        1 - Mild Treatment
        2 - Strong Treatment
        """
        # Transition probabilities
        prob = {
            0: [0.2, 0.5, 0.2, 0.1],  # No treatment
            1: [0.4, 0.4, 0.15, 0.05],  # Mild
            2: [0.6, 0.3, 0.08, 0.02]   # Strong
        }

        current = self.current_state
        if current == 3:
            self.done = True
            return current, -10, self.done

        transition = prob[action]
        next_state = np.random.choice([current, min(current+1, 3)], p=[transition[current], 1-transition[current]])

        self.current_state = next_state

        # Rewards
        if next_state == 0:
            reward = 5
        elif next_state == 1:
            reward = 3
        elif next_state == 2:
            reward = 1
        else:
            reward = -10  # AD

        self.done = next_state == 3
        return next_state, reward, self.done
