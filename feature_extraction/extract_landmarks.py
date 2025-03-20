import zipfile
import os
import numpy as np
import cv2
import mediapipe as mp

# 解壓縮資料集
def unzip_data(zip_path, extract_to):
    """解壓縮資料集"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"資料已解壓至：{extract_to}")

# 設定資料夾路徑
zip_file_path = r"C:\data\dataset.zip"  # 資料集的zip檔案
extract_folder = r"C:\data\extracted_dataset"  # 解壓後的資料夾
output_base_folder = r"C:\data\landmarks_results"  # 解析結果的儲存基底資料夾

# 解壓縮資料
unzip_data(zip_file_path, extract_folder)

# ---------------------------------------------------
# **函數：擷取影片中有效的幀數與關鍵點**
# ---------------------------------------------------
def extract_valid_frame_idx(video_path):
    """
    解析影片，並擷取包含有效人體關鍵點的幀數與像素座標。

    參數：
        video_path (str) : 影片的完整路徑

    返回：
        valid_frames (list) : 包含有效幀索引的列表
        landmark_data (dict) : 每個有效幀的關鍵點座標 {frame_idx: [(pixel_x, pixel_y), ...]}
    """
    # 初始化 MediaPipe Pose 模組
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # 讀取影片
    cap = cv2.VideoCapture(video_path)
    
    valid_frames = []  # 儲存有效幀的索引
    landmark_data = {}  # 儲存對應幀數的 landmarks (像素座標)
    frame_count = 0  # 幀索引計數

    # 讀取影片寬度與高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 遍歷影片的所有幀
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 影片結束，退出迴圈

        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 轉換 BGR 至 RGB (MediaPipe 格式)
        results = pose.process(rgb_frame)  # 偵測人體姿勢

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_landmarks = []

            # 解析 landmarks 並轉換為像素座標
            for landmark in landmarks:
                pixel_x = int(landmark.x * width)
                pixel_y = int(landmark.y * height)

                # 檢查是否為有效座標
                if np.isnan(pixel_x) or np.isnan(pixel_y) or not (0 <= pixel_x < width and 0 <= pixel_y < height):
                    break  # 若任何座標無效，則丟棄此幀
                frame_landmarks.append((pixel_x, pixel_y))

            # 若當前幀的所有座標都有效，則加入有效幀列表
            if len(frame_landmarks) == len(landmarks):
                valid_frames.append(frame_count)
                landmark_data[frame_count] = frame_landmarks

    cap.release()
    pose.close()
    
    return valid_frames, landmark_data


# ---------------------------------------------------
# **處理每個瑜伽姿勢資料夾**
# ---------------------------------------------------
pose_folders = ["Boat", "Downward dog", "Plank", "Triangle", "Warrior three", "Warrior two"]

# 根據每個瑜伽姿勢資料夾，處理影片
for pose in pose_folders:
    # 影片來源資料夾
    folder_path = os.path.join(extract_folder, pose)
    
    # 解析結果儲存資料夾（為每個姿勢創建資料夾）
    output_folder = os.path.join(output_base_folder, pose + "_landmarks")
    os.makedirs(output_folder, exist_ok=True)  # 確保資料夾存在

    # 獲取所有 .mp4 影片
    mp4_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]

    # 處理每個影片並儲存結果
    for video_file in mp4_files:
        input_path = os.path.join(folder_path, video_file)
        
        # 擷取影片的有效幀索引與關鍵點資料
        valid_frames, landmark_data = extract_valid_frame_idx(input_path)

        # 轉換 landmark_data 為 NumPy 陣列
        landmark_array = {str(frame_idx): np.array(landmarks) for frame_idx, landmarks in landmark_data.items()}

        # 定義輸出檔案名稱
        output_file = os.path.join(output_folder, video_file.replace(".mp4", ".npz"))

        # 儲存為 `.npz` 檔案（包含幀索引與對應的關鍵點資料）
        np.savez(output_file, frame_indices=np.array(valid_frames), **landmark_array)

        print(f"已處理並儲存：{output_file}")
