# RL_Agent_0110.py
import numpy as np
import torch

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.9, min_epsilon=0.1, epsilon_decay=0.99, device='cuda'):
        """
        Initialize the Q-Learning Agent.

        :param n_states: Number of possible states.
        :param n_actions: Number of possible actions.
        :param alpha: Learning rate.
        :param gamma: Discount factor.
        :param epsilon: Exploration rate.
        :param min_epsilon: Minimum exploration rate.
        :param epsilon_decay: Decay rate for epsilon.
        :param device: Device to use for computations ('cpu' or 'cuda').
        """
        self.q_table = {}  # Initialize Q-table as a dictionary
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions
        self.device = device

    def select_action(self, state, candidate_stops, n_vehicles):
        """
        Select n_vehicles actions based on the current state and Q-table.
        """
        state_key = str(state)
        # print('555555 select action')
        
        # Initialize Q values for a new state
        if state_key not in self.q_table:
            self.q_table[state_key] = torch.zeros(len(candidate_stops), device=self.device)
            # self.q_table[state_key] = torch.rand(len(candidate_stops), device=self.device)

        if torch.rand(1).item() < self.epsilon:
            # Randomly select n_vehicles actions
            # actions = np.random.choice(candidate_stops, size=n_vehicles, replace=True).tolist()
            action = np.random.choice(candidate_stops).item()
        else:
            # Select top n_vehicles actions based on Q values
            action_values = {stop: self.q_table[state_key][i] for i, stop in enumerate(candidate_stops)}
            # action = sorted(action_values, key=action_values.get, reverse=True)[:n_vehicles]
            action = max(action_values, key=action_values.get)

        action_index = candidate_stops.index(action)
        predicted_reward = self.q_table[state_key][action_index].item()
        print(f"Selected actions: {action}, Candidate stops: {candidate_stops}, Predicted reward: {predicted_reward}")
        print(self.q_table[state_key])
        return action

    def learn(self, state, actions, rewards, next_state, candidate_stops, n_vehicles):
        """
        Update the Q-table using the Q-learning formula for multiple actions.
        """
        state_key = str(state)
        next_state_key = str(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = torch.zeros(len(candidate_stops), device=self.device)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = torch.zeros(len(candidate_stops), device=self.device)

        for action, reward in zip(actions, rewards):
            # Update Q value for each action
            action_index = candidate_stops.index(action)
            predict_value = self.q_table[state_key][action_index]
            target = reward + self.gamma * torch.max(self.q_table[next_state_key])
            target_value = target[action_index]
            # print(f"Before Update: {self.q_table[state_key][action_index]}")
            # print(action_index)
            # print(state_key)
            # print(f"Target: {target_value}, \nPredict: {predict_value}")
            self.q_table[state_key][action_index] += self.alpha * (target_value - predict_value)
            # print(f"Updated Q(s={state}, a={action}): {self.q_table[state_key][action_index]}")
            # print(f"After Update: {self.q_table[state_key][action_index]}")

    def decay_epsilon(self):
        """
        Decay the exploration rate.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        print(f"Updated epsilon: {self.epsilon}")