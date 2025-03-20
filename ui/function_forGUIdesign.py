# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:28:47 2025

@author: Eva Cai
"""
import tkinter as tk

class Background:
    def __init__(self, parent):
        """ 初始化背景並創建 Canvas """
        self.parent = parent
        self.canvas = tk.Canvas(parent, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)  # 讓 Canvas 填滿 Frame
        
        self.create_gradient(self.parent.winfo_width(), self.parent.winfo_height())

        # 綁定視窗大小變化事件
        self.parent.bind("<Configure>", self.update_canvas_size)

    def create_gradient(self, width, height):
        """ 創建淡米色漸層背景 """
        self.canvas.delete("gradient")  # 清除舊的背景

        start_color = "#F5E6CC"  # 淡米色
        end_color = "#FFFFFF"  # 白色

        for i in range(height):
            ratio = i / height  # 計算顏色比例
            r = int(int(start_color[1:3], 16) * (1 - ratio) + int(end_color[1:3], 16) * ratio)
            g = int(int(start_color[3:5], 16) * (1 - ratio) + int(end_color[3:5], 16) * ratio)
            b = int(int(start_color[5:7], 16) * (1 - ratio) + int(end_color[5:7], 16) * ratio)
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.canvas.create_line(0, i, width, i, fill=color, tags="gradient")

    def update_canvas_size(self, event=None):
        """ 當視窗大小改變時，重新繪製漸層背景 """
        width = self.parent.winfo_width()
        height = self.parent.winfo_height()
        if width > 1 and height > 1:  # 確保視窗大小有效
            self.create_gradient(width, height)

class RoundedButton(tk.Canvas):
    def __init__(self, parent, text="▶ 按下以開始辨識", command=None, radius=25,
                 bg="#8B4513", fg="white", border_color="#5C3A1E", border_width=4, **kwargs):
        """ 初始化圓角按鈕 """
        width = radius * 8 + border_width * 2
        height = radius * 2 + border_width * 2

        super().__init__(parent, width=width, height=height, bg="white", highlightthickness=0, **kwargs)
        
        self.command = command
        self.radius = radius
        self.bg_color = bg
        self.fg_color = fg
        self.border_color = border_color
        self.border_width = border_width

        # 先畫邊框，再畫按鈕
        self.create_rounded_rect(0, 0, width, height, radius + border_width, fill=border_color, outline="")
        self.create_rounded_rect(border_width, border_width, width - border_width, height - border_width, 
                                 radius, fill=bg, outline="")

        # 加入文字
        self.text_id = self.create_text(width // 2, height // 2, text=text, fill=fg, font=("微軟正黑體", 14, "bold"))

        # 綁定點擊事件
        self.bind("<Button-1>", self.on_click)

    def create_rounded_rect(self, x1, y1, x2, y2, radius, fill, outline):
        """ 畫圓角矩形，避免中間多餘的線條 """
        self.create_arc(x1, y1, x1 + 2 * radius, y1 + 2 * radius, start=90, extent=90, fill=fill, outline=outline)
        self.create_arc(x2 - 2 * radius, y1, x2, y1 + 2 * radius, start=0, extent=90, fill=fill, outline=outline)
        self.create_arc(x1, y2 - 2 * radius, x1 + 2 * radius, y2, start=180, extent=90, fill=fill, outline=outline)
        self.create_arc(x2 - 2 * radius, y2 - 2 * radius, x2, y2, start=270, extent=90, fill=fill, outline=outline)

        self.create_rectangle(x1 + radius, y1, x2 - radius, y2, fill=fill, outline="")
        self.create_rectangle(x1, y1 + radius, x2, y2 - radius, fill=fill, outline="")

    def on_click(self, event):
        """ 按鈕點擊事件 """
        if self.command:
            # 點擊時變色，模擬按下效果
            self.itemconfig(self.text_id, fill="lightgray")
            self.after(100, lambda: self.itemconfig(self.text_id, fill=self.fg_color))

            # 切換文字（開始 / 停止）
            current_text = self.itemcget(self.text_id, "text")
            new_text = "⏹ 按下以停止辨識" if "▶" in current_text else "▶ 按下以開始辨識"
            self.itemconfig(self.text_id, text=new_text)

            # 執行綁定的動作
            self.command()
            

