import tensorflow as tf
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from lstm+attention_model import create_model
from data_loader import load_data

# = = = 設定超參數 = = = #
# 根據資料集選擇  # 30 幀，每幀 28 / 22 / 6 個特徵
input_shape = (30, 28)  # All
#input_shape = (30, 22)  # length
#input_shape = (30, 6)  # Angle

num_classes = 6
dropout_rate = 0.5
learning_rate = 0.00005
batch_size = 8
epochs = 800
patience_lr = 15
patience_es = 105
# = = = = = = = = = = = #

# 設定資料
X_train, X_val, y_train, y_val = load_data()

# 優化器
optimizer = Nadam(learning_rate=learning_rate, beta_1=0.8, beta_2=0.999, clipnorm=1.0)

# 學習率衰減機制
reduce_lr = ReduceLROnPlateau(
    monitor=
    'val_accuracy',
    # 'val_loss',
    factor=0.5,  # new_learning_rate = old_learning_rate * factor   # 原始 factor=0.2
    patience=patience_lr,  # 等待幾個epoch後才調整學習率
    min_lr=1e-6,  # 最小學習率
    verbose=1
)

# 早停
early_stopping = EarlyStopping(
    monitor=
    'val_accuracy',  # 監控驗證集準確度
    # 'val_loss',
    # monitor='loss'
    # monitor='accuracy'
    patience=patience_es,  # 設定早停耐心
    restore_best_weights=True,  # 恢復最佳權重
    mode=
    'max'  # 精度越高越好
    # 'min'  # 損失函數越小越好
)

# 模型創建與編譯
model = create_model(input_shape)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                    callbacks=[reduce_lr, early_stopping, model_checkpoint])

# 儲存整個模型
model_name = 'LSTM_All_model_test2_7_lin_30_frame_split_second'
save_model_filepath = f'C:/Users/USER/Desktop/Model_results/{model_name}.h5'
model.save(save_model_filepath)  # 儲存整個模型
print(f"模型已儲存至：{save_model_filepath}")