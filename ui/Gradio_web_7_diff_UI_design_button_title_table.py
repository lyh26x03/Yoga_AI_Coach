import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
import tensorflow as tf
import time  
from datetime import datetime

from collections import deque
from function_forpred import AngleFormodel, landmarks_distances, min_max_normalize  # 匯入自訂函數
from function_forposture_add_angle_mp3 import get_name,DashedLineDrawer,Angle
from function_forGUIdesign import RoundedButton,Background
from function_forMySQL import YogaApp


#讀取檔案
model = tf.keras.models.load_model(r'C:\Users\User\Desktop\UI_recognition\LSTM+attention_All_2_lin_30_frame_split_second.h5')
# 瑜珈動作類別名稱
POSE_CLASSES = ["Boat", "DownwardDog", "Plank", "Triangle", "Warriorthree", "Warriortwo"]

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# 讀取攝影機畫面
camera_link = 0  # 如果是網路攝影機請改成 camera_link = f"http://{ip_address}:{port}/video"
class Recognition(tk.Frame):
    def __init__(self, root, title, **kwargs):
        super().__init__(root, **kwargs)
        self.video_device = cv2.VideoCapture(camera_link)
        self.root = root
        self.title = title
        self.running = False
        self.frames_buffer = deque(maxlen=30)  # 儲存最近 30 幀數據
        
        # **初始化繪製器與角度計算器**
        self.dashed_line_drawer = DashedLineDrawer()
        self.angle_calculator = Angle()
        
        self.current_pose = None
        self.is_recording = False
        self.start_time = None
        
        self.duration = np.array([])
        self.name_forrecord = np.array([])
        self.time_forrecord = np.array([])
        self.today_forrecord = np.array([])
        
        self.hide_skeleton = False

        self.init_ui()
        self.display_loop()
        
    def __del__(self):
        try:
            self.video_device.release()
        except Exception:
            pass
        
    def save_pose(self,name,clock,elapsed_time):
        today = datetime.today().strftime("%Y-%m-%d")
        self.duration = np.append(self.duration, elapsed_time)
        self.name_forrecord = np.append(self.name_forrecord, self.current_pose)
        self.time_forrecord = np.append(self.time_forrecord, datetime.now().strftime("%H:%M:%S"))
        self.today_forrecord = np.append(self.today_forrecord, datetime.today().strftime("%Y-%m-%d"))

    def exit_and_show_records(self):
        """按下結束按鈕後，關閉主視窗並開啟瑜珈紀錄表"""
        print("正在關閉應用程式...")  # 確保函數有被執行
        self.running = False
        
        if self.start_time is not None:  
            elapsed_time = round(time.time() - self.start_time, 2)
            if elapsed_time > 2:
                self.save_pose(self.current_pose,self.start_time,elapsed_time)
        
        print("準備儲存的紀錄：")
        print("日期：", self.today_forrecord)
        print("時間：", self.time_forrecord)
        print("姿勢名稱：", self.name_forrecord)
        print("持續時間：", self.duration)

        self.root.destroy()
        self.MySQL_record = YogaApp(self.today_forrecord, self.time_forrecord, self.name_forrecord, self.duration)
        
    def init_ui(self):
        # 創建畫布
        # fill="both", expand=True 讓其撐滿視窗
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # 使用 Background 類別來管理漸層背景
        self.bg = Background(self.canvas)
        self.bg.update_canvas_size()
        
        #當視窗大小變更時，更新背景
        self.bind("<Configure>", self.bg.update_canvas_size)
        
        # **主標題（背景透明）**
        self.ui_title = tk.Label(
                            self, 
                            text=self.title, 
                            font=("Microsoft JhengHei", 28, "bold"),  
                            fg="#4A2C2A",   # 深棕色
                            bg="#F5E6CC",  # 透明背景
                            highlightthickness=0, 
                            borderwidth=0
                            )

        # **副標題（背景透明）**
        self.ui_confidence = tk.Label(
                            self, 
                            text="等待開始辨識...", 
                            font=("Arial", 16),  
                            fg="#7A5C58",  # 柔和棕色
                            bg="#F5E6CC"  # 透明背景
                            )
        self.ui_confidence.place(relx=0.4, rely=1, anchor="center")  
        self.ui_display = tk.Label(self)
        
        # 初始化按鈕
        self.ui_run_button = RoundedButton(self, text="▶按下以開始辨識", command=self.toggle_run, bg="#8B4513")
        
        self.ui_exit_button = RoundedButton(self, text="結束並查看紀錄", command=self.exit_and_show_records, bg="#B22222")
        self.ui_exit_button.place(relx=0.85, rely=0.9, anchor="center")
        
        # 排版
        self.ui_title.pack(pady=10)
        self.ui_confidence.pack(pady=5)
        
        self.ui_run_button.pack(side='bottom', pady=10)
        self.ui_display.pack(side='bottom', fill='both')
        
        # 版面配置（稍微往左移動）
        self.ui_title.place(relx=0.4, rely=0.05, anchor='center')       
        self.ui_confidence.place(relx=0.4, rely=0.1, anchor='center')    
        self.ui_run_button.place(relx=0.4, rely=0.9, anchor='center')    
        self.ui_display.place(relx=0.4, rely=0.5, anchor='center')   
        
        # 加載圖片
        image_path = r"C:\Users\User\Desktop\UI_recognition\Image\Yoga.jpg"  # 替換成你的圖片路徑
        image = Image.open(image_path)
        image = image.resize((370, 370))  # 調整圖片大小
        photo = ImageTk.PhotoImage(image)

        # 建立標籤來顯示圖片
        self.ui_image_label = tk.Label(self.root, image=photo)
        self.ui_image_label.image = photo  # 防止圖片被垃圾回收
        self.ui_image_label.place(relx=0.83, rely=0.5, anchor='center')  # 讓圖片放在右側
                
    def update_pose(self, name):
        # 如果偵測到的新姿勢與當前姿勢不同
        if name != self.current_pose:
            if self.current_pose is not None and self.is_recording:
                # 記錄舊的姿勢時間
                elapsed_time = round(time.time() - self.start_time, 2)
                if  elapsed_time >= 2:
                    print("準備儲存的紀錄")
                    self.save_pose(self.current_pose,self.start_time,elapsed_time)

            #更新當前姿勢資訊
            print("更新當前姿勢資訊")
            self.current_pose = name
            self.start_time = time.time()
            self.is_recording = True  # 開始記錄新姿勢
        
        else:
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 2:
                print(f"姿勢 {name} 已持續 {int(elapsed_time+2)} 秒")
        
    
    def update_pose_image(self, pose_name):
        new_image_path = fr"C:\Users\User\Desktop\UI_recognition\Image\{pose_name}.jpg"
        image = Image.open(new_image_path)
        image = image.resize((370, 370))  # 調整大小
        photo = ImageTk.PhotoImage(image)

        self.ui_image_label.config(image=photo)
        self.ui_image_label.image = photo  # 防止圖片被垃圾回收
        
    def toggle_run(self):
        """切換辨識狀態"""
        self.running = not self.running
        if self.running:
            self.ui_run_button.config(text='⏹ 按下以停止辨識')
        else:
            self.ui_run_button.config(text='▶ 按下以開始辨識')
            self.frames_buffer.clear()  # 停止時清除已收集的幀數據

    def extract_pose_landmarks(self, image, hide_skeletons = False):
        """
        使用 MediaPipe Pose 擷取 33 個關節點座標
        並在影像上繪製骨架點與連接線
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        keypoints = np.zeros((33, 2), dtype=np.float32)
        if results.pose_landmarks:
            # 過濾掉臉部的關節連線
            if not self.hide_skeleton:  
                filtered_connections = [conn for conn in mp_pose.POSE_CONNECTIONS if conn[0] > 10 and conn[1] > 10]
                face_landmark_indices = set(range(0, 11))
                face_spec = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=0, circle_radius=0)
                circle_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
                line_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    filtered_connections,
                    landmark_drawing_spec={i: face_spec if i in face_landmark_indices else circle_spec for i in range(33)},
                    connection_drawing_spec=line_spec
                    )
            
            keypoints = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark], dtype=np.float32)
            
        return image, keypoints,results.pose_landmarks

    def predict_pose(self):
        """
        使用模型預測瑜珈姿勢，回傳類別名稱與信心指數
        """
        if len(self.frames_buffer) < 30:
            return "收集中...", 0  # 等待30幀
        
        keypoints_sequence = np.array(self.frames_buffer)  # 形狀 (30, 33, 2)
        distances = landmarks_distances(keypoints_sequence)  # 計算標準化距離 (30, 22)
        angles = AngleFormodel.calculate_pose_angles(keypoints_sequence)  # 計算角度 (30, 6)
        
        features = np.concatenate([distances, angles], axis=1)  # 合併距離與角度 (30, 28)
        max_Path = r'C:\Users\User\Desktop\UI_recognition\max_value_first_30_split_second_normalize.npy'
        min_Path = r'C:\Users\User\Desktop\UI_recognition\min_value_first_30_split_second_normalize.npy'
        features_nm, _, _ = min_max_normalize(features,max_Path,min_Path)  # Min-Max 標準化
        features_nm = np.delete(features_nm,12, axis = 1)
        prediction = model.predict(features_nm[np.newaxis, :, :])  # 需要額外 batch 維度 (1, 30, 28)
        class_id = np.argmax(prediction)  # 找到最大機率的類別
        class_name = POSE_CLASSES[class_id]
        confidence = prediction[0][class_id] * 100  # 轉換成百分比
        
        return class_name, confidence
    
    def display_loop(self):
        success, capture = self.video_device.read()
        if success:
            capture, keypoints, landmarks_all = self.extract_pose_landmarks(capture, self.hide_skeleton)
            filtered_keypoints = keypoints[(keypoints[:, 0] > 1) | (keypoints[:, 1] > 1)]
            if self.running and filtered_keypoints.size == 0:
                self.frames_buffer.append(keypoints)  # 儲存骨架數據
                pose_name, pose_conf = self.predict_pose()
            
                if pose_conf > 75:
                    self.ui_confidence.config(text=f"辨識結果: {pose_name} (信心指數: {pose_conf:.2f}%)")
                    self.update_pose(pose_name)
                    self.update_pose_image(pose_name)
                    self.hide_skeleton = True  # 這裡設定為不顯示骨架
                    self.dashed_line_drawer.draw_landmarks_and_angles(capture, pose_name, landmarks_all, self.angle_calculator)
                    
                else:
                    self.ui_confidence.config(text=f"結果辨識中....")
                    self.hide_skeleton = False  # 繼續顯示骨架
                    
                    if self.start_time is not None:
                        elapsed_time = round(time.time() - self.start_time, 2)
        
                        if  elapsed_time < 1.5:
                            print("更新當前姿勢資訊")
                            self.current_pose = None
                            self.is_recording = False
                            self.start_time = None
                            
                        elif elapsed_time > 2:  
                            print("準備儲存的紀錄")
                            self.save_pose(self.current_pose,self.start_time,elapsed_time)
        
                            print("更新當前姿勢資訊")
                            self.current_pose = None
                            self.is_recording = False
                            self.start_time = None
                
                    
            # 顯示畫面
            capture = cv2.cvtColor(capture, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(capture)
            self.capture_tk = ImageTk.PhotoImage(image=pil_img)
            self.ui_display.config(image=self.capture_tk)
        
        # 使主視窗等候 (毫秒數) 後重複執行此函數
        main_window.after(10, self.display_loop)

        
# 初始化 Tkinter GUI
main_window = tk.Tk()
#main_window.attributes('-fullscreen', True)
main_window.geometry('800x800')
r1 = Recognition(main_window, '瑜珈姿勢辨識系統')
r1.pack(expand=True, fill='both')

main_window.mainloop()

r1.video_device.release()
