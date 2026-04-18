##
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from train_functions import *


df = pd.read_csv("train.csv")
df = create_new_features(df)
df = df.drop(columns=["id"])
X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y)

categorical_features = X_train.select_dtypes(
    include=["object", "string"]).columns.tolist()

# TE mean
for cat_feature in categorical_features:
    X_train, X_test = mean_target_encoding(X_train, X_test, cat_feature, y_train)
X_train, X_test = mean_target_encoding(X_train, X_test, 'job_education', y_train)
X_train, X_test = mean_target_encoding(X_train, X_test, 'month_contact', y_train)


def objective(trial):
    params = {
        "iterations": 1200,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 6, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "od_type": "Iter",
        "od_wait": 50,
        "task_type": "GPU",
        "devices": "0",
        "eval_metric": "AUC",
        "verbose": False,
        "auto_class_weights": "Balanced"
    }

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1.0)

    current_cat_features = [f for f in categorical_features if f in selected_features]

    train_pool = Pool(X_train_final, y_train, cat_features=current_cat_features)
    test_pool = Pool(X_test_final, y_test, cat_features=current_cat_features)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=test_pool)

    preds = model.predict_proba(X_test_final)[:, 1]
    auc = roc_auc_score(y_test, preds)

    return auc


# ЗАПУСК ПОИСКА ----------------------------------------------------
##
with open('confirmed_features.txt', 'r') as f:
    selected_features = f.read()
    selected_features = selected_features.replace("' ", ' ').replace('[', '').replace(']', '').replace("'", '').replace(' ', '')
    selected_features = selected_features.split(',')
##
X_train_final = X_train[selected_features]
X_test_final = X_test[selected_features]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Лучшие параметры:", study.best_params)
print("Лучший AUC:", study.best_value)
