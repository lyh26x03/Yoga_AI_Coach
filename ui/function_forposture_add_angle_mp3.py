# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 22:51:49 2025

@author: Eva Cai
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import pygame 
import time

'''
name_forangle,                                                #角度需求_名稱
list_of_points_forbodyangle_l, list_of_points_forbodyangle_r, #角度需求_標示在畫面左右上側
diff_angle_forcolorline,                                      #超出範圍紅色/中間值白色/中兩側黃色
list_of_points_fordashline                                    #start/length/end point
angle_fordashline
'''

def get_name(name):
    pose_dict = {
        "Boat":        (["Knee","Torso_thigh"],[[23, 25, 27], [11, 23, 25]], [[24, 26, 28], [12, 24, 26]], [[115,118,122,125],[40,60,70,75]], [[25, 27, 23], [23, 11, 25]],  [-120,-65]),
        "DownwardDog": (["Torso_thigh",""], [[11, 23, 25],[]], [[12, 24, 26],[]], [[75,78,82,90],[]], [[23, 25, 11],[]], [80,0]),
        "Plank":       (["Back_thigh","Shoulder"], [[11,23,27],[13, 11, 23]], [[12, 24, 28],[14, 12, 24]], [[140, 145, 155, 160],[75, 80, 90, 95]], [[23, 27, 11],[11, 23, 13]], [150, 85]),
        "Triangle":    (["Torso inclination", " Shoulder "], [[25, 23, 11],[15, 11, 12]], [[26, 24, 12],[16, 12, 11]], [[45, 50, 60, 65],[170, 175, 180, 185]], [[23,11, 25],[12, 16, 11]],  [55, 180]),
        "Warriorthree":(["Back_thigh"," Abdomen_Thigh"], [[25, 23, 11], [25, 23, 11]],[[26, 24, 12],[26, 24, 12]], [[176, 178, 180, 181],[80, 85, 100, 105]], [[23, 25, 11],[24, 12, 26]], [180,-90]), 
        "Warriortwo":  (["Shoulder" , "Knee"], [[15, 11, 12], [23, 25, 27]], [[16, 12, 11], [24, 26, 28]], [[170, 175, 180, 181], [90, 95, 105, 110]], [[12, 16, 11],[25, 23, 27]], [-180, -100])
    }

    # 如果 `name` 不存在於 `pose_dict`，回傳對應格式的空值
    return pose_dict.get(name, ([], [], [], [], [], []))


def to_pixel(image,index, landmarks):
    """ 將 Mediapipe 的關鍵點轉換為像素座標 """
    return (int(landmarks.landmark[index].x * image.shape[1]),
            int(landmarks.landmark[index].y * image.shape[0]))

def adjust_lists(list_points, angles):
    list_points = [[element + 1 for element in sublist] for sublist in list_points]
    angles = [-element for element in angles]
    return list_points, angles


class DashedLineDrawer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        pygame.mixer.init()
        self.success_sound = pygame.mixer.Sound(r"C:\Users\User\Desktop\UI_recognition\Image\Correct.wav") 
        self.green_start_time = None  # 記錄綠色開始的時間
    
    def play_sound(self):
        """ 播放音效 """
        self.success_sound.play()
        
    def draw_dashed_line(self, image, p1, p2, color, thickness=2, gap=10):
        """繪製虛線"""
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        points = np.linspace(p1, p2, int(dist // gap))
        for i in range(0, len(points) - 1, 2):
            cv2.line(image, tuple(points[i].astype(int)), tuple(points[i + 1].astype(int)), color, thickness)

    def get_angle_line_length(self,image, landmarks,angle_offset,list_of_points_fordashline):
        """從虛線出發,angle_offset方向"""
        # 取得影像大小
        frame_height, frame_width = image.shape[:2]

        # 取得人體關鍵點像素座標
        start_point = to_pixel(image,list_of_points_fordashline[0], landmarks)
        length_point =  to_pixel(image,list_of_points_fordashline[1], landmarks)
        length = math.sqrt((length_point[0] - start_point[0]) ** 2 +
                             (length_point[1] - start_point[1]) ** 2)

        end_point = to_pixel(image,list_of_points_fordashline[2], landmarks)

        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]

        # 計算原始向量角度（以水平向右為基準）
        base_angle = math.atan2(dy, dx)  # OpenCV y 軸向下，所以 dy 取負號
        target_angle = base_angle - math.radians(angle_offset)  # 向上旋轉
        # 計算旋轉後終點座標
        end_x = int(start_point[0] + length * math.cos(target_angle))
        end_y = int(start_point[1] + length * math.sin(target_angle))

        self.draw_dashed_line(image, start_point, (end_x, end_y),color = (0, 255, 0))
        
    def draw_landmarks_and_angles(self, image, name, landmarks, angle_calculator):
        """繪製姿勢資訊""" 

        # 取得姿勢參數
        (name_forangle, list_of_points_forbodyangle_l, list_of_points_forbodyangle_r,
         diff_angle_forcolorline,
         list_of_points_fordashline,
         angle_fordashline) = get_name(name)
        
        
        color_g = (0, 255, 0)
        color_w = (255, 255, 255)
        color_red = (0, 0, 255)
        color_y = (0, 255, 255)

        
        # 個別單獨判斷
        diff_x = landmarks.landmark[11].x - landmarks.landmark[27].x
        pose_idx_map = {"Boat": range(2),"DownwardDog": range(1), "Plank": range(2),"Triangle": range(2),"Warriorthree": range(2),"Warriortwo": range(2)}
        idx = pose_idx_map[name]
        
        if name == "Boat" and diff_x < 0:
            list_of_points_fordashline, angle_fordashline = adjust_lists(list_of_points_fordashline, angle_fordashline)
        elif name in ["Plank", "DownwardDog"] and diff_x > 0:
            list_of_points_fordashline, angle_fordashline = adjust_lists(list_of_points_fordashline, angle_fordashline)
            
            
        
        diff_y = landmarks.landmark[15].y - landmarks.landmark[16].y
        if name == "Triangle" and  diff_y > 0:
            list_of_points_fordashline = [[24,12, 26],[11, 15, 12]]
            angle_fordashline = [-element for element in angle_fordashline]
        
        Left = angle_calculator.angle_body(image,landmarks,[11,23,25])
        Right = angle_calculator.angle_body(image,landmarks,[12,24,26])
        diff_angle_1 = Left-Right
        if name == "Warriorthree":
            if diff_x > 0:  # 臉面向右
                angle_fordashline = [-element for element in angle_fordashline] #抬左腳
                if diff_angle_1 < 0: #抬右腳
                    list_of_points_fordashline = [[24, 26, 23],[23,11,25]]
                    angle_fordashline = [-element for element in angle_fordashline]
    
            elif diff_x < 0 and diff_angle_1 < 0:  # 臉面向左且抬右腳
                list_of_points_fordashline = [[24,26,23],[23,11,25]]
                angle_fordashline = [-element for element in angle_fordashline]
             
        
        Left = angle_calculator.angle_body(image,landmarks,[23,25,27])
        Right = angle_calculator.angle_body(image,landmarks,[24,26,28])
        diff_angle_2 = Left-Right
        if name == "Warriortwo" and diff_angle_2 >0:
            list_of_points_fordashline = [[11,15,12],[26,24,28]]
            angle_fordashline = [-element for element in angle_fordashline]
    
        
        for number in idx:
            # 繪製固定度數的虛線直線
            self.get_angle_line_length(image,landmarks,angle_fordashline[number],list_of_points_fordashline[number])
            
            left_angle = angle_calculator.angle_body(image,landmarks,list_of_points_forbodyangle_l[number])
            right_angle = angle_calculator.angle_body(image,landmarks,list_of_points_forbodyangle_r[number])

        
            # 繪製不同顏色的線段
            list_marker_point_line = [list_of_points_fordashline[number][1],list_of_points_fordashline[number][0],list_of_points_fordashline[number][2]]
            Angle_forcompare = angle_calculator.angle_body(image,landmarks,list_marker_point_line)
            if Angle_forcompare < diff_angle_forcolorline[number][0]   or Angle_forcompare > diff_angle_forcolorline[number][-1]:
                colors = color_red
            elif diff_angle_forcolorline[number][0] <= Angle_forcompare < diff_angle_forcolorline[number][1] or diff_angle_forcolorline[number][2] < Angle_forcompare <= diff_angle_forcolorline[number][-1]:
                colors = color_y
            else:
                colors = color_g

            # 設定基礎顏色
            if name in ["Plank", "DownwardDog", "Boat"]:
                color_l = colors
                color_r = colors
            elif name == "Triangle":
                if number == 1:
                    color_l = colors
                    color_r = colors
                elif diff_angle_1 > 0:
                    color_r, color_l = colors, color_w
                elif diff_angle_1 < 0:
                    color_l, color_r = colors, color_w
            elif name == "Warriorthree":
                if diff_angle_1 > 0 and number == 0:
                    color_l, color_r = colors, color_w
                elif diff_angle_1 > 0 and number == 1:
                    color_r, color_l = colors, color_w
                elif diff_angle_1 < 0 and number == 0:
                    color_r, color_l = colors, color_w
                elif diff_angle_1 < 0 and number == 1:
                    color_l, color_r = colors, color_w
                else:
                    color_l, color_r = color_w, color_w  
                    
            elif name == "Warriortwo":
                if number == 0:
                    color_l = colors
                    color_r = colors
                elif diff_angle_2 > 0:
                    color_r, color_l = colors, color_w
                elif diff_angle_2 < 0:
                    color_l, color_r = colors, color_w

            if color_l == color_g or color_r == color_g:  # 如果左右其中一條線是綠色
                if self.green_start_time is None:  # 第一次變成綠色
                    self.green_start_time = time.time()
                elif time.time() - self.green_start_time >= 0.5:  # 持續 1 秒
                    self.play_sound()  # 播放音效
                    self.green_start_time = None  # 防止持續播放，重置計時器
                else:
                    self.green_start_time = None  # 只要有任何變化，計時器重置
        
        
            #xy軸 文字位置
            text_x = image.shape[1] - 280 if number == 1 else 20
            text_y1 = 30  # 第一行文字的 y 座標
            text_y2 = 60  # 第二行文字的 y 座標
                
            cv2.putText(image, f"Right {name_forangle[number]}: {right_angle:.2f}deg", (text_x, text_y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_r, 2)
            cv2.putText(image, f"Left {name_forangle[number]}: {left_angle:.2f}deg", (text_x, text_y2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_l, 2)
        
            points = []
            list_marker_point_line = [list_of_points_fordashline[number][1],list_of_points_fordashline[number][0],list_of_points_fordashline[number][2]]
            for i in  list_marker_point_line:
                x = int(landmarks.landmark[i].x * image.shape[1])
                y = int(landmarks.landmark[i].y * image.shape[0])
                points.append((x, y))
                cv2.circle(image, (x, y), radius=6, color=colors, thickness=-1)


            for j in range(len(points) - 1):
                cv2.line(image, points[j], points[j + 1], color=colors, thickness=3)
        
        
class Angle:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    @staticmethod
    def get_angle(a, b):
        """ 計算兩個向量的夾角（度數） """
        angle_radians = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angle_radians = np.clip(angle_radians, -1.0, 1.0)  # 避免浮點誤差
        return np.degrees(np.arccos(angle_radians))

    def angle_body(self,image, landmarks, list_of_points):
        """ 計算某夾角 """
        mid = to_pixel(image,list_of_points[1], landmarks)
        left = to_pixel(image,list_of_points[0], landmarks)
        right = to_pixel(image,list_of_points[2], landmarks)

        vector1 = np.array([left[0] - mid[0], left[1] - mid[1]])
        vector2 = np.array([right[0] - mid[0], right[1] - mid[1]])
        return self.get_angle(vector1, vector2)




