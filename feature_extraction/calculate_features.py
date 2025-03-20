import os
import numpy as np
import pickle
import zipfile

# 解壓縮資料集
def unzip_data(zip_path, extract_to):
    """解壓縮資料集"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"資料已解壓至：{extract_to}")

# 用戶選擇資料路徑
def get_data_root_choice():
    """讓使用者選擇是否使用已解壓縮的資料或是進行解壓縮"""
    choice = input("是否使用已解壓縮的資料？(y/n): ").strip().lower()
    if choice == 'y':
        # 使用已解壓縮資料
        return r"C:\data\landmarks_results"
    else:
        # 解壓縮資料並使用
        zip_file_path = input("請提供包含 landmarks_results.zip 的路徑：").strip()
        extract_folder = r"C:\data\extracted_landmarks"
        unzip_data(zip_file_path, extract_folder)
        return extract_folder

# 目標資料夾路徑
DATA_ROOT = get_data_root_choice()  # 用戶選擇的資料夾或解壓縮後的資料夾
SAVE_ROOT = r"C:\data\yoga_pose_features"  # 儲存特徵資料的資料夾，名稱已簡化

# 定義各個瑜伽動作的檔案名稱
yoga_poses = {
    "plank": {"filename": "plank_all_frame_features.pkl"},
    "boat": {"filename": "boat_all_frame_features.pkl"},
    "downdog": {"filename": "downward_dog_all_frame_features.pkl"},
    "triangle": {"filename": "triangle_all_frame_features.pkl"},
    "warrior_two": {"filename": "warrior_two_all_frame_features.pkl"},
    "warrior_three": {"filename": "warrior_three_all_frame_features.pkl"},
}

# --- 函式定義 ---
def get_all_frame_landmarks(data):
    """讀取 .npz 檔案，回傳有效幀索引和 landmarks 座標的字典"""
    all_frame_landmarks = {}
    all_frame_indices = data['frame_indices'][1:]  # 從第二幀開始
    for frame_idx in all_frame_indices:
        landmarks = data[str(frame_idx)]
        all_frame_landmarks[frame_idx] = landmarks
    return all_frame_landmarks

def calculate_angles(landmarks, file_name, frame_idx):
    """計算單幀的六個關節角度"""
    def get_angle(p1, p2, p3):
        """計算三個點形成的夾角 (p2 為頂點)"""
        vec1 = np.array(p1) - np.array(p2)
        vec2 = np.array(p3) - np.array(p2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if np.isnan(norm1) or np.isnan(norm2) or norm1 == 0 or norm2 == 0:
            print(f"檔案: {file_name}, 幀: {frame_idx}, 異常點: {p1, p2, p3} 含 NaN 或長度為 0")
            return 0

        cos_theta = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        return angle

    frame_angles = np.array([
        get_angle(landmarks[13], landmarks[11], landmarks[23]),  # 左上半身
        get_angle(landmarks[14], landmarks[12], landmarks[24]),  # 右上半身
        get_angle(landmarks[25], landmarks[24], landmarks[23]),  # 左膝上
        get_angle(landmarks[26], landmarks[24], landmarks[23]),  # 右膝上
        get_angle(landmarks[28], landmarks[26], landmarks[24]),  # 右膝
        get_angle(landmarks[27], landmarks[25], landmarks[23])   # 左膝
    ])
    return frame_angles

def calculate_normalized_distances(landmarks, file_name=None, frame_idx=None):
    """計算正規化後的距離"""
    if landmarks.shape != (33, 2):
        raise ValueError("輸入的 landmarks 必須是 (33,2) 的 numpy 陣列")

    avg_point = np.mean(landmarks[:11], axis=0)
    point_11 = landmarks[11]
    point_23 = landmarks[23]

    norm_dist = np.linalg.norm(point_11 - point_23)
    if norm_dist == 0:
        norm_dist = 1
        if file_name and frame_idx is not None:
            print(f"[影片: {file_name}, 幀: {frame_idx}] 第 11 號點到 23 號點的距離為 0!!")

    distances = [np.linalg.norm(point_11 - avg_point) / norm_dist]
    for i in range(12, 33):
        dist = np.linalg.norm(point_11 - landmarks[i]) / norm_dist
        distances.append(dist)

    return np.array(distances)

# --- 主要程式邏輯 ---
for pose_name, pose_data in yoga_poses.items():
    filename = pose_data["filename"]
    folder_path = os.path.join(DATA_ROOT, pose_name + "_landmarks")
    npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

    All_video_frame_features = []

    for item in npz_files:
        file_path = os.path.join(folder_path, item)
        try:
            data = np.load(file_path, allow_pickle=True)
        except Exception as e:
            print(f"載入 {file_path} 發生錯誤: {e}")
            continue

        all_frame_landmarks = get_all_frame_landmarks(data)
        frame_features = []

        for frame_idx, landmarks in all_frame_landmarks.items():
            theta = calculate_angles(landmarks, item, frame_idx)
            dist = calculate_normalized_distances(landmarks, file_name=item, frame_idx=frame_idx)
            feature_vector = np.concatenate([dist, theta])
            frame_features.append(feature_vector)

        if frame_features:
            frame_features = np.array(frame_features)
            print(f"影片: {item}, shape: {frame_features.shape}")
            All_video_frame_features.append(frame_features)

    # 確保儲存資料夾存在
    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)

    save_path = os.path.join(SAVE_ROOT, filename)

    with open(save_path, "wb") as f:
        pickle.dump(All_video_frame_features, f)

    print(f"{pose_name} 特徵已儲存至: {save_path}")

    # --- 讀取驗證 ---
    with open(save_path, "rb") as f:
        loaded_data = pickle.load(f)

    print(f"載入的 {pose_name} 資料數量 (影片數): {len(loaded_data)}")
    print(f"每個 {pose_name} 影片的有效幀數和特徵數量:", [arr.shape for arr in loaded_data])