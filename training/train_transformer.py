import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from transformer_model import create_transformer_model
from data_loader import load_data

# = = = 設定超參數 = = = #
# 根據資料集選擇  # 30 幀，每幀 28 / 22 / 6 個特徵
input_shape = (30, 28)  # All
#input_shape = (30, 22)  # length
#input_shape = (30, 6)  # Angle

num_classes = 6  # 類別數量
num_layers = 3  # Transformer 層數
num_heads = 6  # Self-attention 的頭數
key_dim = 128  # 每個頭的維度
ff_dim = 64  # Feed Forward 層的維度
dropout_rate = 0.8  # Dropout 比例
learning_rate = 0.00002  # 初始學習率
batch_size = 48  # 批次大小
epochs = 2400  # 訓練輪數
patience_es = 100
# = = = = = = = = = = = #

# 學習率衰減機制
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=patience_lr, min_lr=1e-7)

# 早停
early_stopping = EarlyStopping(monitor='val_loss', patience=patience_es, restore_best_weights=True)

# 創建模型
model = create_transformer_modelel(input_shape, num_classes, num_layers, num_heads, key_dim, ff_dim, dropout_rate)

# 編譯模型
model.compile(optimizer=Adam(learning_rate=cosine_lr), 
              loss='sparse_categorical_crossentropy', 
              metrics=['sparse_categorical_accuracy'])

# 訓練模型
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=[reduce_lr, early_stopping]
)

# 訓練完後，儲存整個模型
model_name = 'Transformer_All_1_Eva_30_frame_split_first'
save_model_filepath = f'C:/Users/USER/Desktop/Model_results/{model_name}.h5'
model.save(save_model_filepath)  # 儲存整個模型
print(f"模型已儲存至：{save_model_filepath}")