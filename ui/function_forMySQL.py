'''
CREATE TABLE yoga_poses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    pose_name VARCHAR(255) NOT NULL,
    duration INT NOT NULL
);
'''

import tkinter as tk
from tkinter import ttk
import mysql.connector
from datetime import datetime

class YogaApp(tk.Tk):
    def __init__(self,today, clocks, name, duration):
        super().__init__()
        self.title("瑜珈紀錄表")
        self.geometry("900x600")
        self.configure(bg="#EAF4F4")  
        
        # 儲存資料
        self.today_forrecord = today
        self.time_forrecord = clocks
        self.name_forrecord = name
        self.duration = duration

        self.style = ttk.Style()
        self.style.configure("Custom.Treeview", 
                             background="white", 
                             foreground="black",
                             fieldbackground="white",
                             borderwidth=1,
                             rowheight=30)
        self.style.configure("Custom.Treeview.Heading", font=("Arial", 14, "bold"))
        self.style.map("Custom.Treeview", background=[("selected", "#ADD8E6")])  

        self.sql_init_ui()
        self.connect_db()
        
                
    def sql_init_ui(self):
        self.main_frame = tk.Frame(self, bg="#EAF4F4")
        self.main_frame.pack(fill="both", expand=True)

        columns = ("日期", "時間", "姿勢", "維持秒數")
        self.tree = ttk.Treeview(self.main_frame, columns=columns, show="headings", style="Custom.Treeview")

        self.tree.heading("日期", text="日期", anchor="center")
        self.tree.heading("時間", text="時間", anchor="center")
        self.tree.heading("姿勢", text="姿勢", anchor="center")
        self.tree.heading("維持秒數", text="維持秒數", anchor="center")

        self.tree.column("日期", anchor="center", width=200)
        self.tree.column("時間", anchor="center", width=120)
        self.tree.column("姿勢", anchor="center", width=200)
        self.tree.column("維持秒數", anchor="center", width=120)

        self.tree.pack(side="top", fill="both", expand=True)

        #逐行插入資料
        for i in range(len(self.today_forrecord)):
            self.tree.insert("", "end", values=(self.today_forrecord[i], self.time_forrecord[i], self.name_forrecord[i], self.duration[i]))

        self.button_frame = tk.Frame(self, bg="#EAF4F4")
        self.button_frame.pack(side="bottom", fill="x", pady=10)

        self.upload_button = tk.Button(self.button_frame, 
                                       text="上傳選取", 
                                       command=self.upload_selected, 
                                       bg="#6EC6CA", fg="white", 
                                       font=("Arial", 12, "bold"), 
                                       activebackground="#336699",  
                                       activeforeground="black",
                                       relief="flat", padx=10, pady=5)
        self.upload_button.pack(side="left", expand=True, padx=10)

        self.upload_all_button = tk.Button(self.button_frame, 
                                           text="全部上傳✈", 
                                           command=self.upload_all, 
                                           bg="#57A773", fg="white", 
                                           font=("Arial", 12, "bold"), 
                                           activebackground="#A3D9A5",  
                                           activeforeground="black",
                                           relief="flat", padx=10, pady=5)
        self.upload_all_button.pack(side="left", expand=True, padx=10)

        self.delete_button = tk.Button(self.button_frame, 
                                       text="刪除選取", 
                                       command=self.delete_selected, 
                                       bg="#FFA7A7", fg="white", 
                                       font=("Arial", 12, "bold"), 
                                       activebackground="#E85A4F",  
                                       activeforeground="black",
                                       relief="flat", padx=10, pady=5)
        self.delete_button.pack(side="left", expand=True, padx=10)

    def connect_db(self):
        """ 連接 MySQL 資料庫 """
        self.db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="SQLSQL1234",
            database="yoga_db"
        )
        self.cursor = self.db.cursor()

    def upload_selected(self):
        """ 只上傳選取的資料到 MySQL """
        selected_items = self.tree.selection()
        if not selected_items:
            print("請選取要上傳的資料")
            return

        for item in selected_items:
            date_str, time_str, pose_name, duration = self.tree.item(item)["values"]
            full_datetime = f"{date_str} {time_str}"
            time_obj = datetime.strptime(full_datetime, "%Y-%m-%d %H:%M:%S")

            query = "INSERT INTO yoga_poses (timestamp, pose_name, duration) VALUES (%s, %s, %s)"
            self.cursor.execute(query, (time_obj, pose_name, duration))
            self.tree.delete(item) 

        self.db.commit()
        print("✅選取資料已上傳到 MySQL")

    def upload_all(self):
        """ 上傳所有資料到 MySQL """
        all_items = self.tree.get_children()
        if not all_items:
            print("沒有資料可上傳")
            return

        for item in all_items:
            date_str, time_str, pose_name, duration = self.tree.item(item)["values"]
            full_datetime = f"{date_str} {time_str}"
            time_obj = datetime.strptime(full_datetime, "%Y-%m-%d %H:%M:%S")

            query = "INSERT INTO yoga_poses (timestamp, pose_name, duration) VALUES (%s, %s, %s)"
            self.cursor.execute(query, (time_obj, pose_name, duration))
            self.tree.delete(item)

        self.db.commit()
        print("✅所有資料已上傳到 MySQL")
        self.destroy() 

    def delete_selected(self):
        """ 刪除選取的資料 """
        selected_items = self.tree.selection()
        if not selected_items:
            print("⚠️ 請選取要刪除的資料")
            return

        for item in selected_items:
            self.tree.delete(item)

        print("❌ 選取的資料已刪除")

'''
if __name__ == "__main__":
    app = YogaApp()
    app.mainloop()
'''
