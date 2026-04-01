##
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


df = pd.read_csv("train.csv")
X = df.drop(columns=["id", "y"])
y = df["y"]
categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

# Диапазоны параметров
space = {
    'depth': hp.choice('depth', [4, 6, 8, 10]),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
    'border_count': hp.choice('border_count', [32, 64, 128]),
    'learning_rate': hp.choice('learning_rate', [4, 6, 8, 10]),
}


def objective(params):
    tscv = TimeSeriesSplit(n_splits=2)
    auc_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train_w, X_test_w = X.iloc[train_idx], X.iloc[test_idx]
        y_train_w, y_test_w = y.iloc[train_idx], y.iloc[test_idx]

        clf = CatBoostClassifier(
            **params,
            iterations=1000,
            eval_metric="AUC",
            cat_features=categorical_features,
            auto_class_weights="Balanced",
            early_stopping_rounds=50,
            verbose=False,
            task_type="GPU",
            devices="0"
        )

        clf.fit(X_train_w, y_train_w, eval_set=(X_test_w, y_test_w))
        score = clf.get_best_score()['validation']['AUC']
        auc_scores.append(score)

    return {'loss': -np.mean(auc_scores), 'status': STATUS_OK}


print("Запуск оптимизации параметров...")
trials = Trials()
best_params_idx = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=20,
    trials=trials
)

best_params = {
    'depth': [4, 6, 8, 10][best_params_idx['depth']],
    'learning_rate': best_params_idx['learning_rate'],
    'l2_leaf_reg': best_params_idx['l2_leaf_reg'],
    'bagging_temperature': best_params_idx['bagging_temperature'],
    'border_count': [32, 64, 128][best_params_idx['border_count']]
}

print("Лучшие параметры найдены:", best_params)
