##
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from train_functions import create_new_features
from config import name

df = pd.read_csv("train.csv")
df = create_new_features(df)
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

    auc = model_window.get_best_score()['validation']['AUC']
    best_iter = model_window.get_best_iteration()

    cv_auc_scores.append(auc)
    best_iterations.append(best_iter)

    print(f"Фолд {i + 1}: AUC = {auc:.4f}, Лучшая итерация = {best_iter}")

print(f"\nСредний AUC по кросс-валидации: {np.mean(cv_auc_scores):.4f}")

final_iterations = int(np.mean(best_iterations))
##

params = {
    'learning_rate': 0.08579479437504067,
    'depth': 8,
    'l2_leaf_reg': 0.1531536819683288,
    'bootstrap_type': 'Bernoulli',
    'random_strength': 1.9093184370082293e-05,
    'subsample': 0.3574694503073419,
    "task_type": "GPU",
    "devices": "0",
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "cat_features": categorical_features,
}

final_model = CatBoostClassifier(**params)
final_model.set_params(iterations=1200)

final_model.fit(X, y, use_best_model=True,
                verbose=True)

joblib.dump(final_model, f"models/CatBoost_{name}.pkl")
print("Модель сохранена успешно!")
