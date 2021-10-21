#RLExp_0210.py
from RL_opt_0210 import StateEncoder, PredictOptimizeVRP, AssignmentHandler, RewardCalculator, ActionExecutor
from RL_Agent_0210 import QLearningAgent
from Util import VRPGurobi, VRPsolutiontoList, eval_ad, eval_sd
import numpy as np
import csv
import logging
from tqdm import tqdm
import torch

device = 'cuda'
print(f"Using device: {device}")
# 初始化 PredictOptimizeVRP
epochs = 120 # 設置訓練的 epoch 數量
date = 210
csv_training_path = f'./Output/training/{epochs}epochs_nor_0{date}.csv'

# 加載數據
npzfile_stops = np.load("./Data/daily_stops.npz", allow_pickle=True)
stops_list = npzfile_stops['stops_list']  
nr_vehicles = npzfile_stops['nr_vehicles']  
weekday = npzfile_stops['weekday']  
capacities_list = npzfile_stops['capacities_list']  
demands_list = npzfile_stops['demands_list']  
npzfile_route = np.load("./Data/daily_routematrix.npz", allow_pickle=True)
opmat = npzfile_route['incidence_matrices'] 
stop_wise_days = npzfile_route['stop_wise_active'] 
distance_matrix = np.load("./Data/Distancematrix.npy") 
edge_mat = np.load("./Data/edge_category.npy") 
all_days = set(range(1, 201))
test_days = {
         7,   14,  18,  25,  26,  32,  33,  
         39,  40,  46,  53,  56,  63,  70, 
         77,  82,  86,  88,  93,  100, 107,
         117, 123, 126,
         154, 160, 166, 173, 180, 187, 194,
         155, 161, 167, 174, 181, 188, 195,
         149, 156, 168, 175, 182, 189, 196,
         150, 162, 169, 176, 183, 190, 197,
         157, 163, 170, 177, 184, 191, 198, 
         158, 164, 171, 178, 185, 192, 199,
         159, 165, 172, 179, 186, 193, 200}
training_days = sorted(all_days - test_days)

# 設置日誌
log1 = logging.getLogger('log1')
fileHandler = logging.FileHandler('Rlexp.log', mode='w')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fileHandler.setFormatter(formatter)
log1.addHandler(fileHandler)
log1.setLevel(logging.INFO)

# 初始化 CSV 文件
csv_filename = 'RLexp_results.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 寫入 CSV 文件的表頭
    writer.writerow(['Epoch', 'Day', 'Action', 'Reward', 'Total Reward'])

# 超參數
rl_agent = QLearningAgent(
    n_states=500,
    n_actions=len(stops_list[0]),
    alpha=0.5,
    gamma=0.9,
    epsilon=0.5,
    min_epsilon = 0.01,
    epsilon_decay=0.999, 
    device=device
)

# 初始化模組
state_encoder = StateEncoder(device=device)
reward_calculator = RewardCalculator(eval_ad=eval_ad, eval_sd=eval_sd, distance_mat=distance_matrix)
assignment_handler = AssignmentHandler(distance_matrix, rl_agent)
action_executor = ActionExecutor(assignment_handler, reward_calculator, state_encoder, rl_agent)


vrp_model = PredictOptimizeVRP(
    rl_agent=rl_agent,
    state_encoder=state_encoder,
    assignment_handler=assignment_handler,
    reward_calculator=reward_calculator,
    distance_mat=distance_matrix, 
    epochs=epochs,  # 使用動態設定的 epochs
    min_epsilon_threshold=0.01,
    device=device,
    csv_routes='./Output/baseline/baseline_routes_1.csv',
    csv_summary='./Output/baseline/baseline_summary_1.csv',
    csv_training=csv_training_path  # 動態生成的檔案名稱
)

# 訓練模型
vrp_model.fit(
    trgt=opmat,
    weekday=weekday,
    stops_list=stops_list,
    n_vehicleslist=nr_vehicles,
    distance_mat=distance_matrix,
    active_days=stop_wise_days,
    demands=demands_list,
    capacities=capacities_list,
    training_days=training_days,
    log=log1
)

# 測試模型
test_results = vrp_model.test(
    trgt=opmat,
    weekday=weekday,
    stops_list=stops_list,
    n_vehicleslist=nr_vehicles,
    distance_mat=distance_matrix,
    active_days=stop_wise_days,
    demands=demands_list,
    capacities=capacities_list,
    test_days=test_days,
    log=log1
)

# 保存測試結果到 CSV
test_csv_filename = f'RLexp_test_results_0{date}.csv'
with open(test_csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Day', 'Reward', 'Fulfilled Demand', 'Distance Penalty'])
    for result in test_results:
        writer.writerow([
            result['day'],
            result['reward'],
            result['fulfilled_demand'],
            result['distance_penalty']
        ])

print("測試結果已保存到", test_csv_filename)
