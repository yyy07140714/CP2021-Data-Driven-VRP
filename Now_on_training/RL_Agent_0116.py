# RL_Agent_0116.py
import numpy as np
import torch
import random

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.9, min_epsilon=0.1, epsilon_decay=0.995, device='cuda'):
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

    def select_action(self, state, candidate_stops, n_vehicles, current_position, distance_mat):
        """
        選擇動作，結合 Q 值和距離優化。
        
        :param state: 當前狀態。
        :param candidate_stops: 可選停靠點。
        :param n_vehicles: 當前車輛數量。
        :param current_position: 車輛當前位置。
        :param distance_mat: 距離矩陣。
        :return: 選擇的動作。
        """
        state_key = str(state)
        all_stops = list(range(74))  # 確保與 Q 表一致

        # 初始化 Q 表
        if state_key not in self.q_table:
            self.q_table[state_key] = torch.zeros(len(all_stops), device=self.device)

        # 提取 Q 值
        action_values = {stop: self.q_table[state_key][stop] for stop in candidate_stops}

        # 若無可選動作，返回 None
        if not candidate_stops:
            print(f"No candidate stops available for state {state_key}.")
            return None

        # 加入距離權重的調整
        def compute_action_score(action):
            # 獲取 Q 值和距離得分
            q_value = action_values.get(action, 0).item()
            distance_score = -distance_mat[current_position, action]  # 距離越小越好（負值）
            # 綜合得分：Q 值和距離得分加權
            return q_value + 0.1 * distance_score

        # 動作選擇策略
        if random.random() < self.epsilon:  # 探索
            action = random.choice(candidate_stops)
        else:  # 利用，選擇綜合得分最高的動作
            action = max(candidate_stops, key=compute_action_score)

        return action

    def learn(self, state, actions, rewards, next_state, candidate_stops, n_vehicles):
        """
        Update the Q-table using the Q-learning formula for multiple actions.
        """
        state_key = str(state)
        next_state_key = str(next_state)
        all_stops = list(range(74))

        if state_key not in self.q_table:
            self.q_table[state_key] = torch.zeros(len(all_stops), device=self.device)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = torch.zeros(len(all_stops), device=self.device)
        
        
        for action, reward in zip(actions, rewards):
            # Update Q value for each action
            action_index = all_stops.index(action)
            predict_value = self.q_table[state_key][action_index]
            target = reward + self.gamma * torch.max(self.q_table[next_state_key])
            # if target.numel() > 1:  # 如果 target 是向量
            target_value = target[action_index]
            # else:
                # target_value = target.item()

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