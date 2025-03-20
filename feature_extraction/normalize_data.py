import numpy as np
import os
import pickle

def read_28_features_data(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def filter_valid_data(data):
    return [np.array(item) for item in data if isinstance(item, np.ndarray) and item.shape[1] == 28]

def split_into_30_frames(data, split_num=30):
    processed = []
    for arr in data:
        N, C = arr.shape
        if N >= split_num:
            indices = np.linspace(0, N - 1, split_num, dtype=int)
            processed.append(arr[indices])
    return np.array(processed) if processed else None

def min_max_normalize(data, X_min, X_max):
    return [(X - X_min) / (X_max - X_min + 1e-8) for X in data]

# 設定資料夾路徑
folder_path = r"C:\data\processed\yoga_pose_features"
save_folder = r"C:\data\yoga_features_norm"
os.makedirs(save_folder, exist_ok=True)

# 定義資料集的檔案名稱（更簡潔的格式）
poses = ["downdog", "boat", "plank", "triangle", "warrior_two", "warrior_three"]
train_files = {pose: f"{pose}_all_frame_features.pkl" for pose in poses}
test_files = {pose: f"{pose}_all_frame_features_2.pkl" for pose in poses}

def process_data(yoga_pose_files, data_type):
    print(f"Processing {data_type} data...")
    
    # 讀取數據
    yoga_pose_data = {pose: read_28_features_data(os.path.join(folder_path, file)) for pose, file in yoga_pose_files.items()}
    
    # 過濾無效數據
    filtered_data = {pose: filter_valid_data(data) for pose, data in yoga_pose_data.items()}
    
    # 切割成 30 幀
    split_data = {pose: split_into_30_frames(data) for pose, data in filtered_data.items()}
    
    return split_data

# 處理訓練與測試資料
train_split_data = process_data(train_files, "train")
test_split_data = process_data(test_files, "test")

# 計算訓練資料的最大最小值
all_pose_features = np.concatenate([np.concatenate(train_split_data[pose], axis=0) for pose in train_split_data if train_split_data[pose] is not None])
features_max = np.max(all_pose_features, axis=0)
features_min = np.min(all_pose_features, axis=0)

# 儲存最大最小值
np.save(os.path.join(save_folder, "max_value.npy"), features_max)
np.save(os.path.join(save_folder, "min_value.npy"), features_min)

# 正規化資料
train_normalized_data = {pose: min_max_normalize(train_split_data[pose], features_min, features_max) for pose in train_split_data if train_split_data[pose] is not None}
test_normalized_data = {pose: min_max_normalize(test_split_data[pose], features_min, features_max) for pose in test_split_data if test_split_data[pose] is not None}

# 轉換為數據集格式
X_train = np.concatenate(list(train_normalized_data.values()), axis=0)
y_train = np.concatenate([np.full(len(d), i) for i, d in enumerate(train_normalized_data.values())])
X_test = np.concatenate(list(test_normalized_data.values()), axis=0)
y_test = np.concatenate([np.full(len(d), i) for i, d in enumerate(test_normalized_data.values())])

# 分離特徵（長度與角度）
length_features = X_train[:, :, :22]
angle_features = X_train[:, :, 22:]
length_features_drop_zero = np.delete(length_features, 12, axis=2)

# 儲存處理後的資料
np.save(os.path.join(save_folder, "X_train.npy"), X_train)
np.save(os.path.join(save_folder, "y_train.npy"), y_train)
np.save(os.path.join(save_folder, "X_test.npy"), X_test)
np.save(os.path.join(save_folder, "y_test.npy"), y_test)
np.save(os.path.join(save_folder, "length_features.npy"), length_features_drop_zero)
np.save(os.path.join(save_folder, "angle_features.npy"), angle_features)

print("Data processing complete!")