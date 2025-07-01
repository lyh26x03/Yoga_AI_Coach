# Yoga AI Coach

Yoga AI Coach 是一個基於深度學習的瑜伽姿勢識別系統，分別使用 LSTM 和 Transformer 來分析影片中的瑜伽動作，並提供即時的 AI 矯正指導！

> *「技術不只是解法，也是一種凝視世界的方式。在此專案，透過深度學習去觀察『動作』——觀察人類如何在時序中表達身體、在空間裡書寫能量。」*

> **打造一個能即時指導互動、精確辨識分類並回饋使用者的瑜珈動作辨識系統，而且有效提升運動效果。**


## 📂 目錄結構

```javascript
Yoga_AI_Coach/
├── feature_extraction/         # 特徵提取模組
│   ├── extract_landmarks.py         # 從影片中提取人體關鍵點（MediaPipe）
│   ├── calculate_features.py        # 計算長度和角度特徵
│   └── normalize_data.py            # 標準化特徵數據
├── training/                  # 訓練與評估模型
│   ├── data_loader.py                 # 訓練前的數據預處理
│   ├── lstm+attention_model.py       # LSTM + Attention 模型結構
│   ├── train_lstm+attention.py       # 訓練 LSTM + Attention 模型
│   ├── evaluate_model.py             # 評估模型表現
│   ├── transformer_model.py          # Transformer 模型結構
│   └── train_transformer.py          # 訓練 Transformer 模型
├── ui/                      # UI 介面
│   ├── function_forGUIdesign.py
│   ├── function_forMySQL.py
│   ├── function_forposture_add_angle_mp3.py
│   ├── function_forpred.py
│   ├── Gradio_web_7_diff_UI_design_button_title_table.py
│   └── Image/
├── data/                    # 資料存放
│   ├── dataset.zip                 # 壓縮後的數據集
│   ├── processed/                 # 預處理後的數據
│   │   ├── landmarks_results.zip         # extract_landmarks.py 產出 (關鍵點數據)
│   │   ├── yoga_pose_features/           # calculate_features.py 產出 (姿勢特徵數據)
│   │   └── yoga_features_norm/           # normalize_data.py 產出 (標準化特徵數據，供模型訓練)
└── README.md                 # 本說明文件
```

---

## 🧩 功能與架構

整體架構圍繞「即時動作辨識」這一核心目標展開，透過三大技術模組與一個互動介面，實現從資料輸入、特徵萃取、模型預測，到使用者回饋的完整流程。

### 1. **特徵提取模組 `feature_extraction/`**

> *把動作轉譯成數字語言。*

原始影片中的人體動作，會經由 MediaPipe 偵測關鍵點（landmarks），接著轉換為具備幾何意義的特徵：

- 骨架長度（如肩膀到手腕）
- 動作角度（如大腿與小腿的夾角）

這些特徵經標準化後，作為深度模型的輸入。此處處理過程已模組化，方便日後擴充更多動作類型或輸入格式。

### 2. **模型訓練與評估模組 `training/`**

> 針對不同特徵組合、模型結構與訓練策略進行多輪實驗。

使用兩種架構進行比較：

- LSTM + Attention：適合序列較短、流暢性高的動作判斷
- Transformer Encoder：擅長捕捉全局關係與時空依賴特徵

此模組包含完整的訓練、測試與評估流程，並透過不同超參數調整與正則化策略，找出效能最佳的架構組合。

---

### 🧠 模型選擇與設計思考－－如何讓模型「看懂動作」？

讓模型學會：**在時間中觀察動作，在空間中辨認姿態。**

我們選擇實作與比較兩種架構：

- **LSTM + Attention**：模仿人的記憶力，**保留動作序列的時間脈絡**，並透過注意力聚焦關鍵姿勢轉折。
- **Transformer Encoder**：從全局視角出發，**同時處理整段序列的互動資訊**，建構更長距離的姿態理解。

| 架構                  | 適用場景        | 核心優勢                   |
| ------------------- | ----------- | ---------------------- |
| LSTM + Attention    | 序列較短、動作流暢   | 強調時間依賴與上下文連貫性；可聚焦關鍵幀位置 |
| Transformer Encoder | 複雜動作、姿勢分化明顯 | 全序列並行建模、捕捉全身協調與長距離依賴關係 |

> 依據不同特徵組合（骨架長度、動作角度、混合特徵）設計多組實驗，並透過各種策略優化模型泛化能力。

此外，考量實作效率與任務目標為「分類」而非「生成」，Transformer 架構僅保留 **Encoder**，省略 Decoder，避免不必要的參數冗餘與過擬合風險。

---

### 📊 實驗最佳結果摘要

| 模型                  | 特徵組合    | 測試準確率     | 參數設計與說明                           |
| ------------------- | ------- | --------- | --------------------------------- |
| LSTM + Attention    | 長度+角度混合 | 95.7%     | 3層 Dense, Dropout 0.5, 800 epochs |
| Transformer Encoder | 長度+角度混合 | 94.3%     | Encoder 3層、Head=6、CosineDecay     |

### 3. **使用者介面模組 `ui/`**

Gradio 建立的使用者介面。

- 攝影機即時偵測。
- 顯示姿勢分類與信心指數。
- 儲存與查詢過往訓練紀錄（MySQL）。

### 4. **資料模組 `data/`**

存放所有資料來源與轉換結果。

- `dataset.zip` 為預處理後 `.npy` 格式。
- 中繼資料（landmarks、features、normalized）皆可獨立使用。

---

### 📁 資料流程簡述

```other
dataset.zip             # 原始特徵資料（.npy 格式）
    ↓ extract_landmarks.py
landmarks_results.zip   # MediaPipe 擷取關鍵點資料
    ↓ calculate_features.py
yoga_pose_features/     # 長度與角度特徵
    ↓ normalize_data.py
yoga_features_norm/     # 正規化後的模型輸入資料
```

每一層處理結果皆存於對應資料夾，並可獨立執行與重現。若面試官僅欲檢閱代碼，可直接搭配已生成好的 `.zip` 檔案進行分析與模型操作，無需重新處理資料。

---

### 🔍 原始資料說明

`dataset.zip` 為經前期處理後的 `.npy` 格式，來源為人工蒐集之 `.mp4` 瑜珈影片。每部影片擷取 30 幀畫面後，透過 MediaPipe Pose 偵測 21 個關鍵點，轉為骨架特徵向量。

處理後的關鍵點、角度與長度資訊已標準化為 `.npy` 格式，使用者可直接進行後續模型訓練。

---

## 🔧 安裝與使用

### 🔢 **特徵提取模組 `feature_extraction/`**

```other
pip install numpy pandas opencv-python mediapipe

# 執行順序如下：
python feature_extraction/extract_landmarks.py
python feature_extraction/calculate_features.py
python feature_extraction/normalize_data.py
```

輸出對應：

- `landmarks_results.zip`
- `yoga_pose_features/`
- `yoga_features_norm/`

若您僅需模型實驗，可直接使用現成輸出資料。

---

### 🧬 **模型訓練與評估模組 `training/`**

#### 📦 安裝套件

```other
pip install tensorflow scikit-learn matplotlib
```

#### 🚀 執行完整流程

```other
# 1. 載入資料與特徵組合選擇
python training/data_loader.py

# 2. 建立模型結構 (LSTM or Transformer)
training/lstm+attention_model.py       # LSTM + Attention 模型定義
training/transformer_model.py          # Transformer 模型定義

# 3. 執行訓練 (包含參數設定、callback 保存)
python training/train_lstm+attention.py
python training/train_transformer.py

# 4. 模型效能評估
python training/evaluate_model.py
```

#### 📁 執行結果與儲存說明

- 輸入特徵組合：
   - `單組` ：骨架長度 **or** 關節角度
   - `複合` ：骨架長度 + 角度
   - 使用者可使用 `data_loader.py` 自行切換
- 模型儲存路徑：
   - `C:/Yoga_AI_Coach/Model_results/`
- 效能評估結果：
   - 混淆矩陣 + Normalized Confusion Matrix
   - F1-Score / Precision / Recall / Macro F1 / Kappa
   - ROC Curve / PR Curve

---

### 🖥️ **使用者介面模組 `ui/`**

Yoga AI Coach 提供一個簡潔友善的使用者介面，支援即時姿勢偵測、動作角度判斷與語音提示，並可記錄每次練習的結果至 MySQL 資料庫，方便使用者追蹤訓練進度。

#### 🔧 安裝套件（UI 模組依賴）

```javascript
pip install pygame opencv-python mediapipe mysql-connector-python
```

#### 🧪 操作流程（模組功能與執行方式）

本模組由數個子功能檔案構成，彼此配合執行以下任務：

| 檔案名稱                                              | 功能描述                                            |
| ------------------------------------------------- | ----------------------------------------------- |
| Gradio_web_7_diff_UI_design_button_title_table.py | ✅ 主程式，啟動 Gradio 介面與攝影機                          |
| function_forposture_add_angle_mp3.py              | ✅ 根據模型預測姿勢判定角度標準，提供色彩標示與音效回饋（正確時播放 Correct.wav） |
| function_forpred.py                               | ✅ 輔助姿勢角度計算、特徵提取與正規化處理                           |
| function_forMySQL.py                              | ✅ 記錄練習姿勢與持續時間，透過 GUI 將結果上傳至 MySQL 資料庫          |
| function_forGUIdesign.py                          | ✅ 自訂 tkinter 背景樣式與圓角按鈕設計                        |

#### ▶ 啟動介面操作（主程式啟動）

```javascript
python ui/Gradio_web_7_diff_UI_design_button_title_table.py
```

✅ 功能總覽：

- 攝影機啟動後可即時偵測姿勢分類（6 種）
- 顯示角度資訊與標準範圍（良好 / 警告 / 錯誤）
- 完成動作時播放音效（`Correct.wav`）
- 每次練習紀錄可選擇上傳至 MySQL（透過 tkinter GUI 操作）

#### 📁 儲存與資料記錄說明

| 項目          | 位置／格式                                               | 說明                       |
| ----------- | --------------------------------------------------- | ------------------------ |
| 音效檔         | ui/Image/Correct.wav                                | ✅ 正確動作時播放提示音（可自定義）       |
| 練習紀錄        | function_forMySQL.py → 資料庫 yoga_db 中 yoga_poses 資料表 | ✅ 儲存「姿勢類別」、「維持秒數」、「時間戳記」 |
| 畫面元件與 UI 風格 | function_forGUIdesign.py                            | 自定義漸層背景與圓角啟動按鈕           |

💡 若無啟動 MySQL 資料庫，仍可使用姿勢辨識與音效提示功能，僅無法上傳與管理紀錄資料。

#### 📝 注意事項

- 若無法播放音效，請確認 `pygame` 初始化是否成功，以及 `Correct.wav` 音檔路徑正確。
- MySQL 預設帳號密碼請依實際環境調整（`localhost` / `root` / `SQLSQL1234`）。
- 請確認已啟動攝影機裝置，否則會無法即時偵測姿勢。
