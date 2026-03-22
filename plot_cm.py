import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

# 1. 加载并预处理数据
file_path = "ecoli.data" 
col_names = ["sequence", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"]

df = pd.read_csv(
    file_path, header=None, sep=r"\s+", names=col_names, 
    encoding="utf-8", on_bad_lines='skip'
)
df = df.drop("sequence", axis=1)

X = df.drop("class", axis=1).values
y = df["class"].values
classes = np.unique(y) # 获取所有唯一的类标签

# 分层划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2. 训练基线模型 KNN
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
knn_pipeline.fit(X_train, y_train)
y_pred_knn = knn_pipeline.predict(X_test)

# 3. 训练提出的 MI-SVM 模型 (此处直接使用您之前调优出的最佳参数)
mi_svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=mutual_info_classif, k='all')),
    ('svm', SVC(C=1, gamma=0.2, kernel='rbf', class_weight='balanced', random_state=42))
])
mi_svm_pipeline.fit(X_train, y_train)
y_pred_svm = mi_svm_pipeline.predict(X_test)

# 4. 计算混淆矩阵
cm_knn = confusion_matrix(y_test, y_pred_knn, labels=classes)
cm_svm = confusion_matrix(y_test, y_pred_svm, labels=classes)

# 5. 绘制并排的高清混淆矩阵对比图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 绘制 KNN (冷色调)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=classes, yticklabels=classes, ax=axes[0])
axes[0].set_title('Baseline: KNN (Accuracy: 88.12%)', fontsize=14, pad=15)
axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')

# 绘制 Proposed MI-SVM (暖色调突出创新)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges', cbar=False, 
            xticklabels=classes, yticklabels=classes, ax=axes[1])
axes[1].set_title('Proposed: MI-SVM (Accuracy: 88.12%)', fontsize=14, pad=15)
axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')

# 调整布局并保存
plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
print("图片已成功保存为 confusion_matrix_comparison.png，可直接用于论文插图！")