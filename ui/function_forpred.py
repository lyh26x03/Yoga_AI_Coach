# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:15:44 2025

@author: Eva Cai
"""

import numpy as np

class AngleFormodel:
    @staticmethod
    def get_angle(a, b):
        """ 計算兩個向量的夾角（度數） """
        angle_radians = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angle_radians = np.clip(angle_radians, -1.0, 1.0)  # 避免浮點誤差
        return np.degrees(np.arccos(angle_radians))
    
    @staticmethod
    def get_vector(landmarks, p1, p2, i):
        """ 計算向量 (p2 - p1) """
        return landmarks[i, p2] - landmarks[i, p1]
    
    @staticmethod
    def calculate_pose_angles(landmarks):
        """ 計算 6 個身體角度 """
        angles = np.zeros((landmarks.shape[0], 6))  
        
        for i in range(landmarks.shape[0]):
            angles[i, 0] = AngleFormodel.get_angle(
                AngleFormodel.get_vector(landmarks, 11, 13, i),
                AngleFormodel.get_vector(landmarks, 11, 23, i)
            )  # 左上半身 (13,11,23)

            angles[i, 1] = AngleFormodel.get_angle(
                AngleFormodel.get_vector(landmarks, 12, 14, i),
                AngleFormodel.get_vector(landmarks, 12, 24, i)
            )  # 右上半身 (14,12,24)

            angles[i, 2] = AngleFormodel.get_angle(
                AngleFormodel.get_vector(landmarks, 24, 25, i),
                AngleFormodel.get_vector(landmarks, 24, 23, i)
            )  # 左膝上 (25,24,23)

            angles[i, 3] = AngleFormodel.get_angle(
                AngleFormodel.get_vector(landmarks, 24, 26, i),
                AngleFormodel.get_vector(landmarks, 24, 23, i)
            )  # 右膝上 (26,24,23)

            angles[i, 4] = AngleFormodel.get_angle(
                AngleFormodel.get_vector(landmarks, 26, 28, i),
                AngleFormodel.get_vector(landmarks, 26, 24, i)
            )  # 右膝 (28,26,24)

            angles[i, 5] = AngleFormodel.get_angle(
                AngleFormodel.get_vector(landmarks, 25, 27, i),
                AngleFormodel.get_vector(landmarks, 25, 23, i)
            )  # 左膝 (27,25,23)

        return angles

def landmarks_distances(landmarks):
    """
    計算各個關節與 11 號點的距離，並進行標準化。
    """
    N = landmarks.shape[0]

    # 計算 0-10 號點的平均座標 (形狀: (N, 2))
    face_point = np.mean(landmarks[:, :11, :], axis=1)

    # 提取 11 號點、23 號點和其餘點
    point_11 = landmarks[:, 11, :]
    point_23 = landmarks[:, 23, :]
    other_points = landmarks[:, 12:, :]

    distances = np.linalg.norm(other_points - point_11[:, np.newaxis, :], axis=2)
    face_distance = np.linalg.norm(face_point - point_11, axis=1, keepdims=True)
    all_distances = np.concatenate([face_distance, distances], axis=1)
    L_side_width = np.linalg.norm(point_11 - point_23, axis=1, keepdims=True)

    L_side_width = np.where(L_side_width == 0, 1e-6, L_side_width)
    normalized_distances = all_distances / L_side_width

    return normalized_distances

def min_max_normalize(X,max_Path,min_Path):
    """
    對 X (影片數, 幀數, 特徵數) 進行 Min-Max 標準化，並記錄 min/max 值。

    :param X: np.ndarray, shape (num_videos, num_frames, num_features)
    :return: X_normalized (相同 shape), X_max (1, 1, num_features), X_min (1, 1, num_features)
    """
    
#   combine_min_max = np.load(Path)
#   X_min = combine_min_max[0:1, :]  # 取出第一列，保持 shape=(1,28)
#   X_max = combine_min_max[1:2, :]  # 取出第二列，保持 shape=(1,28)
    
    X_max = np.load(max_Path)     
    X_min = np.load(min_Path)
    

    X_normalized = np.where(X < X_min, 0, (X - X_min) / (X_max - X_min + 1e-8))

    return X_normalized, X_max, X_min
