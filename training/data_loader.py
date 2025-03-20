import numpy as np
from sklearn.model_selection import train_test_split

# 資料讀取
def load_data():
    # 讀取資料，選擇測試不同的資料集：All/Length/Angle
    # = = = All = = = #
    X = np.load('C:\data\processed\yoga_features_norm\X_train.npy')
    y = np.load('C:\data\processed\yoga_features_norm'\y_train.npy')

    # = = = length = = = #
    # X = np.load('C:\data\processed\yoga_features_norm\length_features.npy')
    # y = np.load('C:\data\processed\yoga_features_norm\y_train.npy')

    # = = = Angle = = = #
    # X = np.load('C:\data\processed\yoga_features_norm\angle_features.npy')
    # y = np.load('C:\data\processed\yoga_features_norm\y_train.npy')
    
    # 切分訓練集與測試集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 顯示資料形狀和類別分佈
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Train Set: {np.bincount(y_train)}")
    print(f"Validation Set: {np.bincount(y_val)}")
    print("Classes: ", np.bincount(y))  # 顯示整體的類別分佈

    return X_train, X_val, y_train, y_val

# 調用此函數讀取資料
X_train, X_val, y_train, y_val = load_data()