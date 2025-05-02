import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import itertools
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, cohen_kappa_score

# = = = 加載最佳模型 = = = #
model = load_model('C:/Yoga_AI_Coach/Model_results/best_model.h5')

# = = = 讀取測試資料 = = = #
X_test = np.load('C:/Yoga_AI_Coach/data/processed/yoga_features_norm/X_test.npy')
y_test = np.load('C:/Yoga_AI_Coach/data/processed/yoga_features_norm/y_test.npy')

# 分開長度與角度特徵
X_test_length = X_test[:, :, :21]  # 長度
X_test_angles = X_test[:, :, 21:]  # 角度

# 選擇模型
Model_name = "LSTM_length_9_lin"  # 或 "LSTM_Angle_9_lin"
checkpoint_filepath = f'C:/Yoga_AI_Coach/Model_results/{Model_name}.h5'
Figure_filepath = "C:/Yoga_AI_Coach/Model_results/Figures/"

# 依據模型類型載入對應的 X_test
if "Angle" in Model_name:
    X_test = X_test_angles
elif "length" in Model_name:
    X_test = X_test_length

# 確保 shape 正確
print("Final X_test shape:", X_test.shape)

# 進行模型測試，獲得測試集損失和準確率
test_loss, test_acc, test_sparse_categorical_acc = model.evaluate(X_test, y_test)  # 使用測試集評估
print(f"Loss: {test_loss}")
print(f"測試集準確率: {test_acc:.2%}")
print(f"測試集分類準確率: {test_sparse_categorical_acc:.2%}")

# 預測結果
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

POSE_CLASSES = ["Boat", "Down_dog", "Plank", "Triangle", "Warrior_3", "Warrior_2"]
num_classes = 6

# 設定儲存圖片的路徑
if not os.path.exists(Figure_filepath):
    os.makedirs(Figure_filepath)

# = = = 混淆矩陣 = = = #
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# 顯示混淆矩陣
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plot_confusion_matrix(cm, POSE_CLASSES, normalize=False, title="Confusion Matrix")
plt.subplot(1, 2, 2)
plot_confusion_matrix(cm, POSE_CLASSES, normalize=True, title="Normalized Confusion Matrix")
plt.savefig(f'{Figure_filepath}confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 顯示分類報告
print(classification_report(y_test, y_pred, target_names=POSE_CLASSES))

# = = = 訓練與驗證損失和準確度曲線 = = = #
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig(f'{Figure_filepath}{Model_name}_Loss_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

# = = = 計算 Macro 和 Weighted F1 分數 = = = #
macro_f1 = f1_score(y_test, y_pred, average='macro')  # Macro F1
weighted_f1 = f1_score(y_test, y_pred, average='weighted')  # Weighted F1
print(f"Macro-averaged F1-score: {macro_f1:.4f}")
print(f"Weighted-averaged F1-score: {weighted_f1:.4f}")

# Cohen's Kappa
kappa = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa: {kappa:.4f}")

# = = = 計算 ROC 曲線和 AUC = = = #
fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, 1])  # 假設二分類情況，改為多類別需要修改
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(f'{Figure_filepath}roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# = = = PR 曲線和 AUCPR = = = #
y_test_onehot = label_binarize(y_test, classes=np.arange(num_classes))
precision = dict()
recall = dict()
aucpr = dict()
for i in range(num_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_onehot[:, i], y_pred_prob[:, i])
    aucpr[i] = auc(recall[i], precision[i])

# 繪製 PR 曲線
plt.figure()
for i in range(num_classes):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {POSE_CLASSES[i]} (AUCPR = {aucpr[i]:.2f})')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.grid(True)
plt.savefig(f'{Figure_filepath}pr_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 計算 Macro-averaged 和 Weighted-averaged AUCPR
macro_aucpr = np.mean([aucpr[i] for i in range(num_classes)])
weights = np.bincount(y_test) / len(y_test)
weighted_aucpr = np.average([aucpr[i] for i in range(num_classes)], weights=weights)

print(f"Macro-averaged AUCPR: {macro_aucpr:.4f}")
print(f"Weighted-averaged AUCPR: {weighted_aucpr:.4f}")
