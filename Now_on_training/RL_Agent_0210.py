# RL_Agent_0210.py
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

    def select_action(self, state, candidate_stops, n_vehicles, current_position, distance_mat, capacities, vehicle_idx, vehicle_load, demands, day):
        state_key = str(state)
        all_stops = list(range(74))  # 確保與 Q 表一致

        # 初始化 Q 表
        if state_key not in self.q_table:
            self.q_table[state_key] = torch.zeros(len(all_stops), device=self.device)

        action_values = {stop: self.q_table[state_key][stop] for stop in candidate_stops}

        if not candidate_stops:
            print(f"No candidate stops available for state {state_key}.")
            return None

        def compute_action_score(action):
            q_value = action_values.get(action, 0).item()

            # 確認capacities格式
            if isinstance(capacities, (int, float, np.int64, np.float64)):  
                car_capacity = capacities  # 直接使用
            elif isinstance(capacities, torch.Tensor):
                car_capacity = capacities.item()  # 轉換成標量
            elif isinstance(capacities, (list, np.ndarray)):
                car_capacity = capacities[day]  # 正常索引
            else:
                raise ValueError(f"Unexpected capacities type: {type(capacities)} on day {day}")

            # 確認demands格式
            if isinstance(demands[day], torch.Tensor):
                action_demand_value = demands[day][action].item()
                max_demand_value = torch.max(demands[day]).item()
            else:
                action_demand_value = demands[day][action]
                max_demand_value = max(demands[day])

            demand_value = action_demand_value / max_demand_value if max_demand_value > 0 else 0

            # print(f"Day {day}, , Vehicle {vehicle_idx}, Capacity: {car_capacity}\nAction {action}, Demand: {action_demand_value}, Max Demand: {max_demand_value}, Normalized: {demand_value}")
            
            # 確認distance格式
            local_max_distance = max(distance_mat[current_position, candidate_stops])  # 只考慮當前可選的動作
            normalized_distance = distance_mat[current_position, action] / local_max_distance if local_max_distance > 0 else 1
            action_values_cpu = [v.cpu().item() if isinstance(v, torch.Tensor) else v for v in action_values.values()]
            distance_weight = max(0.1, np.mean(action_values_cpu) * 0.005)  # 降低影響


            # 計算剩餘可用載重
            remaining_capacity = car_capacity - vehicle_load  
            # print('remaining_capacity', remaining_capacity)

            if action_demand_value > remaining_capacity:
                overload_penalty = -10  # 強烈懲罰，避免選擇超載動作
            else:
                overload_penalty = 0  # 容量足夠，不影響分數

            # **綜合 Q 值、距離懲罰、需求、容量約束**
            return q_value + distance_weight * (-normalized_distance)
            # return q_value + distance_weight * (-normalized_distance) + 0.05 * demand_value + overload_penalty



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
