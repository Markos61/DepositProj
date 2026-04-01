##
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv("train.csv")
df = df.drop(columns=["id"])
X = df.drop("y", axis=1)
y = df["y"]

categorical_features = X.select_dtypes(
    include=["object", "string"]).columns.tolist()

params = {
    "iterations": 4000,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "cat_features": categorical_features,
    "auto_class_weights": "Balanced",
    "early_stopping_rounds": 50,
    "verbose": False,
    "task_type": "GPU",
    "devices": "0"
}

tscv = TimeSeriesSplit(n_splits=5)

cv_auc_scores = []
best_iterations = []

print("Начало обучения с Expanding Window...")
##
for i, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train_w, X_test_w = X.iloc[train_index], X.iloc[test_index]
    y_train_w, y_test_w = y.iloc[train_index], y.iloc[test_index]

    model_window = CatBoostClassifier(**params)

    model_window.fit(
        X_train_w, y_train_w,
        eval_set=(X_test_w, y_test_w),
        use_best_model=True
    )

    # Собираем метрики
    auc = model_window.get_best_score()['validation']['AUC']
    best_iter = model_window.get_best_iteration()

    cv_auc_scores.append(auc)
    best_iterations.append(best_iter)

    print(f"Фолд {i + 1}: AUC = {auc:.4f}, Лучшая итерация = {best_iter}")

print(f"\nСредний AUC по кросс-валидации: {np.mean(cv_auc_scores):.4f}")

# 4. ФИНАЛЬНЫЙ ШАГ: Обучение на 100% данных
# Используем среднее количество итераций из всех окон, чтобы не переобучиться
final_iterations = int(np.mean(best_iterations))
##
final_model = CatBoostClassifier(**params)
final_model.set_params(iterations=50000, early_stopping_rounds=200)

final_model.fit(X, y, verbose=500, use_best_model=True)

# 5. Сохранение
joblib.dump(final_model, "CatBoost_ExpandingWindow_100000.pkl")
print("Модель сохранена успешно!")
