# Yoga AI Coach

Yoga AI Coach 是一個基於深度學習的瑜伽姿勢識別系統，使用 LSTM 和 Transformer 來分析影片中的瑜伽動作，並提供即時的 AI 指導。

## 目錄結構
```
Yoga_AI_Coach/
├── feature_extraction/    # 特徵提取模組
│   ├── extract_landmarks.py         # 從影片中提取人體關鍵點（MediaPipe）
│   ├── calculate_features.py        # 計算長度和角度特徵
│   └── normalize_data.py            # 標準化特徵數據
├── training/    # 訓練與評估模型
│   ├── preprocess.py                 # 訓練前的數據預處理
│   ├── lstm+attention_model.py       # LSTM + Attention 模型結構
│   ├── train_lstm+attention.py       # 訓練 LSTM + Attention 模型
│   ├── evaluate_model.py             # 評估模型表現
│   ├── transformer_model.py          # Transformer 模型結構
│   └── train_transformer.py          # 訓練 Transformer 模型
├── ui/    # UI 介面相關（目前未使用）
│   ├── function_forGUIdesign.py
│   ├── function_forMySQL.py
│   ├── function_forposture_add_angle_mp3.py
│   ├── function_forpred.py
│   ├── Gradio_web_7_diff_UI_design_button_title_ta...
│   └── Image/
├── data/    # 資料存放
│   ├── dataset.zip    # 壓縮後的數據集
│   ├── processed/    # 預處理後的數據
│   │   ├── landmarks_results.zip    # extract_landmarks.py 產出 (關鍵點數據)
│   │   ├── yoga_pose_features/      # calculate_features.py 產出 (姿勢特徵數據)
│   │   ├── yoga_features_norm/      # normalize_data.py 產出 (標準化特徵數據，供模型訓練)
│   │   └── README.md  # 說明處理後的資料
├── README.md    # 本說明文件
└── requirements.txt    # 依賴環境（尚未整理）
```

## 功能概覽
- **特徵提取 (feature_extraction)**
  - 使用 MediaPipe 提取人體關鍵點。
  - 計算骨骼長度和角度作為特徵。
  - 進行特徵標準化以供模型訓練。

- **模型訓練 (training)**
  - 支援 LSTM + Attention 和 Transformer 模型。
  - 透過預處理步驟確保輸入格式一致。
  - 提供訓練與評估功能。

- **資料處理 (data)**
  - 包含壓縮後的數據集 `dataset.zip` 和處理後的特徵數據。
  - `yoga_features_norm/` 是最終供模型訓練的標準化數據。

## 安裝與使用

### **1. 安裝必要套件**
因 `requirements.txt` 未整理，請手動安裝主要依賴：
```bash
pip install numpy pandas opencv-python mediapipe torch torchvision torchaudio
```

### **2. 下載數據**
解壓縮 `dataset.zip`，並確保 `data/processed/` 內包含所需數據。

### **3. 運行特徵提取**
```bash
python feature_extraction/extract_landmarks.py
python feature_extraction/calculate_features.py
python feature_extraction/normalize_data.py
```

### **4. 訓練模型**
LSTM + Attention：
```bash
python training/train_lstm+attention.py
```
Transformer：
```bash
python training/train_transformer.py
```

### **5. 模型評估**
```bash
python training/evaluate_model.py
```

## UI 部分
目前 `ui/` 內的程式未整理，若不需要可刪除 `run_ui.py`。

## 注意事項
- `data/processed/` 內的數據總大小約 72.5MB，可直接上傳 GitHub。
- `requirements.txt` 尚未整理，如有需要可使用 `pip freeze > requirements.txt` 生成。

---

這樣的 README 結構清晰，包含 **專案架構、功能、安裝步驟、運行方式**，讓使用者能快速上手。

# Yoga_AI_Coach
