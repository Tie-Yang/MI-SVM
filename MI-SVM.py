import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, f1_score, accuracy_score

# ============================================================================
# 1. 数据加载与预处理 (保持不变)
# ============================================================================
file_path = "ecoli.data" 
col_names = ["sequence", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"]

df = pd.read_csv(
    file_path, header=None, sep=r"\s+", names=col_names, 
    encoding="utf-8", on_bad_lines='skip'
)
df = df.drop("sequence", axis=1)

X = df.drop("class", axis=1).values
y = df["class"].values

print(f"Dataset Loaded: Samples: {X.shape[0]} | Features: {X.shape[1]}")

# 划分数据集 (分层抽样保证类别分布一致)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ============================================================================
# 2. 定义基线模型 (Baselines) - 涵盖基于距离、树模型和集成学习
# ============================================================================
baselines = {
    "KNN (k=5)": Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    "LightGBM (HistGBM)": HistGradientBoostingClassifier(
        random_state=42, max_iter=100
    )
}

# ============================================================================
# 3. 构建我们提出的框架：特征选择 + 代价敏感 SVM (The Proposed Pipeline)
# ============================================================================
# 将 标准化 -> 基于互信息的特征选择 -> SVM 封装为一个完整的 Pipeline
proposed_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=mutual_info_classif)),
    ('svm', SVC(class_weight='balanced', random_state=42))
])

# 针对 Pipeline 的超参数网格 (涵盖特征保留数量和 SVM 核心参数)
param_grid = {
    'feature_selection__k': [4, 5, 6, 'all'], # 探索最佳的特征保留数
    'svm__C': [0.1, 1, 10],                   # SVM 正则化参数
    'svm__gamma': [0.01, 0.1, 0.2, 1],        # SVM 核函数参数
    'svm__kernel': ['rbf']                    # RBF 核
}

print("Running Pipeline GridSearchCV on CPU...")
start_time = time.time()
# 优化目标仍设定为 f1_macro，兼顾小类别的召回与精确率
proposed_model = GridSearchCV(
    estimator=proposed_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1, # 充分利用 T530 的多核 CPU
    verbose=0
)
proposed_model.fit(X_train, y_train)
elapsed_time = time.time() - start_time
print(f"GridSearchCV Completed in {elapsed_time:.2f} seconds.")

# ============================================================================
# 4. 实验评估与结果对比
# ============================================================================
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, recall_macro, recall_weighted, f1_macro

results = {}

# 评估基线模型
for name, model in baselines.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = compute_metrics(y_test, y_pred)

# 评估提出的框架
y_pred_proposed = proposed_model.predict(X_test)
results["Proposed MI-SVM Pipeline"] = compute_metrics(y_test, y_pred_proposed)

# ============================================================================
# 5. 打印适合写入论文表格的输出格式
# ============================================================================
print("\n" + "="*85)
print("Pipeline Comparative Experimental Results (CPU Environment)")
print("="*85)
print(f"{'Method / Model':<25} | {'Accuracy':<10} | {'Macro Recall':<13} | {'Weighted Rec':<13} | {'Macro F1':<10}")
print("-" * 85)

for name, metrics in results.items():
    if name == "Proposed MI-SVM Pipeline":
        print("-" * 85) # 用分割线突出我们提出的方法
    print(f"{name:<25} | {metrics[0]:.4f}     | {metrics[1]:.4f}        | {metrics[2]:.4f}        | {metrics[3]:.4f}")

print("="*85)
best_params = proposed_model.best_params_
print(f"Optimal Pipeline Parameters: Features Retained (k)={best_params['feature_selection__k']}, " 
      f"SVM C={best_params['svm__C']}, Gamma={best_params['svm__gamma']}")