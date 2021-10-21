# RL_opt_0110.py
import numpy as np
from Util import VRPGurobi, VRPsolutiontoList, eval_ad, eval_sd
import torch
from tqdm import tqdm
import os
import csv
import matplotlib.pyplot as plt


class StateEncoder:
    def __init__(self, device='cuda'):
        """
        Initialize the StateEncoder.

        :param device: The device to be used for tensor computations, e.g., 'cpu' or 'cuda'.
        """
        self.device = device

    def get_state(self, current_position, current_load, day, stops_list, demands):
        """
        Generate the current state of the environment for a specific vehicle and day.

        :param current_position: Current position of the vehicle (index of the stop).
        :param current_load: Current load of the vehicle.
        :param day: Current day index.
        :param stops_list: List of stops for all days.
        :param demands: Demand data for all stops.
        :return: Encoded state vector.
        """
        # Get demands for the day
        day_demands = torch.as_tensor(demands[day], device=self.device)

        # Calculate demand-related features
        remaining_demand_sum = torch.sum(day_demands).item()
        stops_left = len(stops_list) - 1  # Exclude the initial position (0)

        # Calculate average demand and demand variance
        avg_demand = torch.mean(day_demands).item()
        demand_variance = torch.var(day_demands).item()

        # Encode state
        state = self.encode_state(
            current_position, remaining_demand_sum, current_load, stops_left, avg_demand, demand_variance
        )

        # Debug output for state information
        # print(f"Vehicle Position: {current_position}, Current Load: {current_load}, Remaining Demand Sum={remaining_demand_sum}, "
            # f"Stops Left={stops_left}, Average Demand={avg_demand}, Demand Variance={demand_variance}")
        return state

    def encode_state(self, position, remaining_demand_sum, vehicle_load, stops_left, avg_demand, demand_variance):
        """
        Encode the current state as a vector.

        :param position: Current position of the vehicle (index of the stop).
        :param remaining_demand_sum: Total remaining demand to be fulfilled.
        :param vehicle_load: Current load of the vehicle.
        :param stops_left: Number of stops left to visit.
        :param avg_demand: Average demand of the stops.
        :param demand_variance: Variance of the demand.
        :return: Encoded state as a tuple.
        """
        # print('vehicle_load:', vehicle_load)
        # print('type:', type(vehicle_load))

        if isinstance(vehicle_load, torch.Tensor):
            # 如果是 PyTorch 張量，轉為列表後處理
            vehicle_load = vehicle_load.tolist()
            # print("Converted tensor to list:", vehicle_load)
            if isinstance(vehicle_load, list) and len(vehicle_load) > 0:
                vehicle_load_ = vehicle_load[0]  # 假設只需要第一個元素
            else:
                vehicle_load_ = vehicle_load  # 預設值
        elif isinstance(vehicle_load, list):
            # 如果是 Python list，檢查是否有元素
            vehicle_load_ = vehicle_load[0] if len(vehicle_load) > 0 else 0
        else:
            # 如果是單一數值，直接使用
            vehicle_load_ = vehicle_load

        # print("Processed vehicle_load_:", vehicle_load_)

        return (
            int(position),
            int(remaining_demand_sum),
            int(vehicle_load_),
            int(stops_left),
            float(avg_demand),
            float(demand_variance)
        )


class PredictOptimizeVRP:
    def __init__(self, rl_agent, state_encoder, assignment_handler, reward_calculator, epochs=3, min_epsilon_threshold=0.05, device='cuda', 
    cache_dir='baseline_cache', csv_routes='baseline_routes.csv', csv_summary='baseline_summary.csv', csv_training='training.csv', **kwargs):
        """
        Initialize the PredictOptimizeVRP class.

        :param rl_agent: Reinforcement Learning agent.
        :param state_encoder: State encoder for encoding the environment state.
        :param assignment_handler: Assignment handler for vehicle stop assignments.
        :param reward_calculator: Reward calculator.
        :param epochs: Number of training epochs.
        :param min_epsilon_threshold: Minimum epsilon threshold for RL agent.
        :param device: Device to use for computation ('cuda' or 'cpu').
        :param cache_dir: Directory for storing cache files.
        :param csv_routes: Name of the CSV file for storing routes and distances.
        :param csv_summary: Name of the CSV file for storing summary data.
        """
        self.rl_agent = rl_agent
        self.state_encoder = state_encoder
        self.assignment_handler = assignment_handler
        self.reward_calculator = reward_calculator
        self.action_executor = ActionExecutor(
            self.assignment_handler,
            self.reward_calculator,
            self.state_encoder,  # 傳入 state_encoder
            self.rl_agent  # 傳入 rl_agent
        )
        self.epochs = epochs
        self.device = device
        self.cache_dir = cache_dir
        self.min_epsilon_threshold = min_epsilon_threshold
        self.csv_routes = csv_routes
        self.csv_summary = csv_summary
        self.csv_training = csv_training

    def initialize_baseline(self, trgt, weekday, stops_list, n_vehicleslist, distance_mat, demands, capacities, test_days, epoch):
        """
        Initialize baseline solutions for the test days and save results to the specified CSV files.

        :param trgt: Target solution.
        :param weekday: Day of the week information.
        :param stops_list: List of stops.
        :param n_vehicleslist: Number of vehicles per day.
        :param distance_mat: Distance matrix.
        :param demands: Demands for each stop.
        :param capacities: Vehicle capacities.
        :param test_days: List of test days.
        :param epoch: Current epoch.
        """
        # Check if CSV files exist
        if os.path.exists(self.csv_routes) and os.path.exists(self.csv_summary):
            print(f"CSV files {self.csv_routes} and {self.csv_summary} already exist. Skipping baseline initialization.")
            return

        # Open the CSV files for appending results
        with open(self.csv_routes, mode='a', newline='') as routes_file, open(self.csv_summary, mode='a', newline='') as summary_file:
            routes_writer = csv.writer(routes_file)
            summary_writer = csv.writer(summary_file)

            # Write headers if the files do not exist
            if not os.path.exists(self.csv_routes):
                routes_header = ["Epoch", "Day", "AD", "SD"]
                max_vehicles = max(n_vehicleslist)
                for i in range(max_vehicles):
                    routes_header.extend([f"Vehicle_{i + 1}_Route", f"Vehicle_{i + 1}_Distance"])
                routes_header.append("Total_Distance")  # Add Total_Distance as the last column
                routes_writer.writerow(routes_header)

            if not os.path.exists(self.csv_summary):
                summary_writer.writerow(["Epoch", "Day", "AD", "SD", "Fulfilled_Demand", "Total_Distance", "Reward"])

            from Util import VRPGurobi, VRPsolutiontoList, eval_ad, eval_sd

            # Iterate over the test days
            for d in test_days:
                # Run the baseline algorithm
                solved, comment, sol, u = VRPGurobi(distance_mat, demands[d], capacities[d], n_vehicleslist[d], stops_list[d])

                if solved:
                    actual_solution = VRPsolutiontoList(trgt[d])
                    vrp_solution = VRPsolutiontoList(sol)

                    fulfilled_demand = sum(torch.sum(torch.tensor([demands[d][stop] for stop in route if stop != 0], device=self.device)) for route in vrp_solution)
                    total_distance = sum(sum(distance_mat[route[i], route[i + 1]] for i in range(len(route) - 1)) for route in vrp_solution)

                    ad_diffcount, _ = eval_ad(vrp_solution, actual_solution)
                    sd_diffcount, _ = eval_sd(vrp_solution, actual_solution)

                    # Calculate reward
                    reward = fulfilled_demand.item() - total_distance - ad_diffcount - sd_diffcount

                    # Write to the routes CSV
                    routes_row = [epoch, d, ad_diffcount, sd_diffcount]
                    for route in vrp_solution:
                        route_distance = sum(distance_mat[route[i], route[i + 1]] for i in range(len(route) - 1))
                        routes_row.extend([" -> ".join(map(str, route)), route_distance])
                    routes_row.append(total_distance)  # Add Total_Distance as the last column
                    routes_writer.writerow(routes_row)

                    # Write to the summary CSV
                    summary_writer.writerow([
                        epoch, d, ad_diffcount, sd_diffcount, 
                        fulfilled_demand.item(), total_distance, reward
                    ])

        print(f"Baseline initialization completed. Results saved to {self.csv_routes} and {self.csv_summary}.")


    def fit(self, trgt, weekday, stops_list, n_vehicleslist, distance_mat, active_days, demands, capacities, training_days, log=None):
        print('Starting training...')
        print(f"Training on {len(demands)} days of demands.")
        baseline_results = self.initialize_baseline(trgt, weekday, stops_list, n_vehicleslist, distance_mat, demands, capacities, training_days, log)
        distance_mat = torch.tensor(distance_mat, device=self.device)
        demands = [torch.tensor(d, device=self.device) for d in demands]
        capacities = [torch.tensor(c, device=self.device) for c in capacities]

        # Initialize CSV file
        with open(self.csv_training, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Epoch', 'Day', 'AD', 'SD']
            max_vehicles = max(n_vehicleslist)
            for i in range(max_vehicles):
                header.append(f'Vehicle_{i + 1}_Route')
                header.append(f'Vehicle_{i + 1}_Distance')
            writer.writerow(header)

            epoch_total_distances = []  # 記錄每個 epoch 的總距離
            epoch_total_rewards = [] 

            for ep in tqdm(range(self.epochs), desc="Epochs"):
                print(f"Epoch {ep + 1}/{self.epochs} starting...")
                epoch_rewards = []
                total_distance_this_epoch = 0  # 初始化當前 epoch 的總距離
                total_reward_this_epoch = 0

                for day_idx, d in enumerate(tqdm(training_days, desc=f"Epoch {ep + 1} Days (Training)", leave=False)):
                    print(f"Epoch {ep + 1}, Day {day_idx + 1}/{len(training_days)}: Training on day {d}")
                    candidate_stops = stops_list[d]
                    n_vehicles = n_vehicleslist[d]

                    vehicles = self.assignment_handler.assign_stops_to_vehicles(
                        n_vehicles, candidate_stops, demands, capacities, trgt, d,
                        self.reward_calculator, self.state_encoder, self.rl_agent
                    )

                    total_reward = 0
                    for vehicle_idx, vehicle in enumerate(vehicles):
                        for stop_idx in range(len(vehicle) - 1):
                            current_position = vehicle[stop_idx]
                            next_position = vehicle[stop_idx + 1]

                            state = self.state_encoder.get_state(
                                current_position=current_position,
                                current_load=0, 
                                day=d,
                                stops_list=candidate_stops,
                                demands=demands
                            )

                            next_state = self.state_encoder.get_state(
                                current_position=next_position,
                                current_load=0, 
                                day=d,
                                stops_list=candidate_stops,
                                demands=demands
                            )

                            fulfilled_demand, distance_penalty, reward = self.reward_calculator.compute_reward(
                                vehicles, demands, distance_mat, trgt, d, capacities
                            )

                            self.rl_agent.learn(
                                state, [next_position], [reward], next_state, candidate_stops, n_vehicles
                            )

                            total_reward_this_epoch += reward

                    ad_diffcount, _ = eval_ad(vehicles, VRPsolutiontoList(trgt[d]))
                    sd_diffcount, _ = eval_sd(vehicles, VRPsolutiontoList(trgt[d]))

                    route_distances = [
                        sum(distance_mat[route[i], route[i + 1]].item() for i in range(len(route) - 1)) if len(route) > 1 else 0
                        for route in vehicles
                    ]
                    total_distance_this_day = sum(route_distances)
                    total_distance_this_epoch += total_distance_this_day

                    row = [ep + 1, d, ad_diffcount, sd_diffcount]
                    for route, distance in zip(vehicles, route_distances):
                        row.append(' -> '.join(map(str, route)))
                        row.append(distance)
                    remaining_columns = (max_vehicles - len(vehicles)) * 2
                    row.extend([''] * remaining_columns)
                    writer.writerow(row)

                    if log:
                        log.info(f"Day {d} - AD: {ad_diffcount}, SD: {sd_diffcount}, Reward: {total_reward}")
                        log.info(f"Vehicle Routes: {vehicles}, Distances: {route_distances}")

                    epoch_rewards.append(total_reward)

                epoch_total_distances.append(total_distance_this_epoch)  # 記錄當前 epoch 的總距離
                epoch_total_rewards.append(total_reward_this_epoch)
                self.rl_agent.decay_epsilon()
                if self.rl_agent.epsilon <= self.min_epsilon_threshold:
                    print(f"Training stopped due to epsilon threshold. Final Epsilon: {self.rl_agent.epsilon}")
                    break

                print('epoch rewards', epoch_rewards)

                # 繪製當前 epoch 的總距離圖
                epoch_total_distances = [
                    d.item() if isinstance(d, torch.Tensor) and d.numel() == 1  # 單元素張量
                    else d.sum().item() if isinstance(d, torch.Tensor)  # 多元素張量，計算總和
                    else float(d) if isinstance(d, (int, float))  # 如果已是數值，直接使用
                    else 0  # 預設值，避免錯誤
                    for d in epoch_total_distances]
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(epoch_total_distances) + 1), epoch_total_distances, marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('Total Distance')
                plt.title('Total Distance per Epoch')
                plt.grid(True)
                plt.savefig(f'./Output/epoch_distance.png')
                plt.close()

                epoch_total_rewards = [
                    r.item() if isinstance(r, torch.Tensor) and r.numel() == 1  # 單元素張量
                    else r.sum().item() if isinstance(r, torch.Tensor)  # 多元素張量，計算總和
                    else float(r) if isinstance(r, (int, float))  # 如果已是數值，直接使用
                    else 0  # 預設值，避免錯誤
                    for r in epoch_total_rewards]

                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(epoch_total_rewards) + 1), epoch_total_rewards, marker='o', color='green')
                plt.xlabel('Epoch')
                plt.ylabel('Total Reward')
                plt.title('Total Reward per Epoch')
                plt.grid(True)
                plt.savefig(f'./Output/epoch_reward.png')
                plt.close()

                print(f"Epoch {ep + 1} completed. Total Distance: {total_distance_this_epoch}")

            print(f"Training results have been saved to {self.csv_training}")

    def test(self, trgt, weekday, stops_list, n_vehicleslist, distance_mat, active_days, demands, capacities, test_days, log=None):
        """
        Test the trained RL agent on the test days.
        """
        print('Testing model...')
        test_results = []
        print(f"Testing on {len(demands)} days of demands.")  # 應為 201
        
        for d in test_days:
            print(f"Testing on Day {d}...")
            candidate_stops = stops_list[d]
            n_vehicles = n_vehicleslist[d]
            vehicles = [[0] for _ in range(n_vehicles)]
            vehicle_loads = [0] * n_vehicles
            episode_total_reward = 0
            
            # 提取當天的需求，避免多次索引
            day_demands = demands[d]
            
            # 初始化狀態
            state = self.state_encoder.get_state(
                current_position=0, 
                current_load=0, 
                day=d, 
                stops_list=candidate_stops, 
                demands=demands  # 傳入當天需求
            )
            done = False

            while not done:
                all_actions = []
                for vehicle_idx, vehicle in enumerate(vehicles):
                    position = vehicle[-1]
                    current_load = vehicle_loads[vehicle_idx]
                    # 動態更新狀態
                    state = self.state_encoder.get_state(
                        current_position=position,
                        current_load=current_load,
                        day=d,
                        stops_list=candidate_stops,
                        demands=demands  # 傳入當天需求
                    )

                    # 選擇動作
                    action = self.rl_agent.select_action(state, candidate_stops, n_vehicles)
                    all_actions.append((vehicle_idx, action, state))

                for vehicle_idx, action, state in all_actions:
                    if action in candidate_stops:
                        candidate_stops.remove(action)
                        vehicles[vehicle_idx].append(action)
                        vehicle_loads[vehicle_idx] += day_demands[action]  # 使用當天的需求

                    # 獲取獎勵並更新狀態
                    initial_reward, final_reward, fulfilled_demand, distance_penalty, next_state, done = self.action_executor.take_action(action, state, d, stops_list, demands, capacities, distance_mat, n_vehicleslist, weekday, active_days, trgt)
                    fulfilled_demand = fulfilled_demand.sum().item() if isinstance(fulfilled_demand, torch.Tensor) else sum(fulfilled_demand)
                    if isinstance(final_reward, np.ndarray):
                        final_reward = final_reward.item()
                    episode_total_reward += final_reward
                    state = next_state
                print('candidate stops',candidate_stops)
                # 如果候選站點都分配完畢，結束當天
                if not candidate_stops:
                    done = True

            print(f"Day {d} completed. Total reward: {episode_total_reward}")
            test_results.append({
                'day': d,
                'reward': episode_total_reward,
                'vehicles': vehicles
            })

            if log:
                log.info(f"Test Day {d} - Total Reward: {episode_total_reward}")

        return test_results

class AssignmentHandler:
    def __init__(self, distance_mat):
        self.distance_mat = distance_mat

    def assign_stops_to_vehicles(self, n_vehicles, candidate_stops, demands, capacities, trgt, day, reward_calculator, state_encoder, rl_agent):
        """
        Assign stops to vehicles using global reward maximization, with dynamic state updates for each vehicle.
        """
        vehicles = [[0] for _ in range(n_vehicles)]  # 初始化每輛車的路線
        vehicle_loads = [0] * n_vehicles  # 初始化每輛車的負載

        # 確保候選站點不包含起始點
        candidate_stops = [stop for stop in candidate_stops if stop != 0]

        # 當候選站點尚未分配完畢
        while candidate_stops:
            # print('caaandidate: ', candidate_stops)
            all_actions = []
            for vehicle_idx, vehicle in enumerate(vehicles):
                # 獲取當前車輛狀態
                current_position = vehicle[-1]
                current_load = vehicle_loads[vehicle_idx]

                # 獲取車輛的當前狀態向量
                state = state_encoder.get_state(
                    current_position=current_position,
                    current_load=current_load,
                    day=day,
                    stops_list=candidate_stops,
                    demands=demands
                )

                # 為當前車輛選擇動作
                # print('car id',vehicle_idx,'day', day)
                action = rl_agent.select_action(state, candidate_stops, n_vehicles)
                all_actions.append((vehicle_idx, action, state))

            # 執行分配和更新
            for vehicle_idx, action, state in all_actions:
                # 如果 action 是列表，则逐个移除
                if isinstance(action, list):
                    for single_action in action:
                        if single_action in candidate_stops:
                            print(f"Assigning stop {single_action} to Vehicle {vehicle_idx}.")
                            candidate_stops.remove(single_action)  # 从候选站点中移除
                            vehicles[vehicle_idx].append(single_action)  # 分配给车辆
                            vehicle_loads[vehicle_idx] += demands[day][single_action]
                            _, _, reward = reward_calculator.compute_reward(vehicles, demands, self.distance_mat, trgt, day, capacities)
                            print(f"Vehicle_list {vehicle_idx} chose action {single_action}, reward: {reward}")
                            print('Left candidate', candidate_stops)
                elif action in candidate_stops:
                    print(f"Assigning stop {action} to Vehicle {vehicle_idx}.")
                    candidate_stops.remove(action)
                    vehicles[vehicle_idx].append(action)
                    vehicle_loads[vehicle_idx] += demands[day][action]
                    _, _, reward = reward_calculator.compute_reward(vehicles, demands, self.distance_mat, trgt, day, capacities)
                    print(f"Vehicle_one {vehicle_idx} chose action {action}, reward: {reward}")
                    print('Left candidate', candidate_stops)

        # 確保每輛車返回起始點
        for vehicle in vehicles:
            vehicle.append(0)

        return vehicles

    def distance(self, vehicle, stop):
        return self.distance_mat[vehicle, stop]

class RewardCalculator:
    def __init__(self, eval_ad, eval_sd, distance_mat):
        self.eval_ad = eval_ad
        self.eval_sd = eval_sd
        self.distance_mat = torch.tensor(distance_mat, dtype=torch.float64)
        self.max_distance = self.distance_mat.max().item()
        self.normalized_distance_mat = self.distance_mat / self.max_distance


    def compute_reward(self, formatted_solution, demands, distance_mat, trgt, d, capacities):
        """
        Compute the reward for a given VRP solution.

        :param formatted_solution: The proposed VRP solution.
        :param demands: Demands for each stop.
        :param distance_mat: Distance matrix between stops.
        :param trgt: Ground truth solution.
        :param d: Current day index.
        :param capacities: Vehicle capacity limits.
        :return: Fulfilled demand, distance penalty, and computed reward.
        """
        actual_solution = VRPsolutiontoList(trgt[d])
        vrp_solution = formatted_solution
        fulfilled_demand = sum(sum(demands[stop] for stop in route if stop != 0) for route in vrp_solution)
        print('fulfilled',fulfilled_demand)
        if isinstance(fulfilled_demand, int):
            fulfilled_demand = torch.tensor([fulfilled_demand], device='cuda:0')
        max_demand = fulfilled_demand.max().item()
        normalized_fulfill = fulfilled_demand / max_demand

        distance_penalty = sum(sum(self.normalized_distance_mat[route[i], route[i + 1]] for i in range(len(route) - 1)) for route in vrp_solution)
        total_distance = sum(sum(self.distance_mat[route[i], route[i + 1]].item() for i in range(len(route) - 1))for route in vrp_solution)
        if isinstance(distance_penalty, torch.Tensor):
            distance_penalty = distance_penalty.item()
        # print(distance_penalty)
        # max_distance = distance_penalty.max().item()
        # normalized_distance = distance_penalty / max_distance
        # print('nor dis', normalized_distance)

        # Add capacity violation penalty
        # capacity_violation_penalty = 0
        # for route, capacity in zip(vrp_solution, capacities):
        #     route_load = sum(demands[stop] for stop in route if stop != 0)
        #     if route_load > capacity:
        #         capacity_violation_penalty += (route_load - capacity) * 10  # Penalty factor for exceeding capacity

        # Evaluate difference with actual solution
        ad_diffcount, ad_diffrelative = self.eval_ad(vrp_solution, actual_solution)
        sd_diffcount, sd_diffrelative = self.eval_sd(vrp_solution, actual_solution)
        # Compute final reward
        # reward = normalized_fulfill-distance_penalty
        reward = 100+ normalized_fulfill*100 - distance_penalty - ad_diffcount*0.01 - sd_diffcount*0.01
        #- capacity_violation_penalty

        return fulfilled_demand, total_distance, reward


class ActionExecutor:
    def __init__(self, assignment_handler, reward_calculator, state_encoder, rl_agent):
        self.assignment_handler = assignment_handler
        self.reward_calculator = reward_calculator
        self.state_encoder = state_encoder  # 添加 state_encoder
        self.rl_agent = rl_agent  # 添加 rl_agent

    def take_action(self, action, state, d, stops_list, demands, capacities, distance_mat, n_vehicleslist, weekday, active_days, trgt):
        # print(f"Action taken for vehicle routing decision: {action}")

        # 更新分配邏輯
        vehicle_assignments = self.assignment_handler.assign_stops_to_vehicles(
            n_vehicleslist[d],
            stops_list[d],
            demands,
            capacities[d],
            trgt,
            d,
            self.reward_calculator,
            self.state_encoder,  # 傳遞 state_encoder
            self.rl_agent  # 傳遞 rl_agent
        )
        # print('Actual solution:',VRPsolutiontoList(trgt[d]))
        # print("Final vehicle assignments:", vehicle_assignments)

        # 獎勵計算
        fulfilled_demand, distance_penalty, reward = self.reward_calculator.compute_reward(
            vehicle_assignments, demands, distance_mat, trgt, d, capacities
        )
        # print(f"Reward: {reward}, Fulfilled Demand: {fulfilled_demand}, Distance Penalty: {distance_penalty}")
        if d + 1 >= len(demands):
            next_state = None
            done = True
        else:
            next_state = self.state_encoder.get_state(
                current_position=0,  # 假設所有車輛都重置到起點
                current_load=0,  # 假設所有車輛的載重清空
                day=d + 1,
                stops_list=stops_list,
                demands=demands
            )
            done = d == len(trgt) - 1

        return reward, reward, fulfilled_demand, distance_penalty, next_state, done