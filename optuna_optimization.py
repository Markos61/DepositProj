##
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import optuna.visualization as vis
from config import name
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
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        "learning_rate": trial.suggest_float("learning_rate", 5e-4, 0.05, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 20.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),

        # --- НОВЫЕ ПАРАМЕТРЫ ---
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "max_bin": trial.suggest_int("max_bin", 32, 254),
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 15),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        # -----------------------

        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
        "od_type": "Iter",
        "od_wait": 100,  # больше шансов не упасть раньше времени
        "task_type": "GPU",
        "devices": "0",
        "eval_metric": "AUC",
        "verbose": False,
        "iterations": 2000,
        # "auto_class_weights": "Balanced"
    }

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1.0)

    if params["grow_policy"] == "Lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 16, 64)

    current_cat_features = [f for f in categorical_features if f in selected_features]

    train_pool = Pool(X_train_final, y_train, cat_features=current_cat_features)
    test_pool = Pool(X_test_final, y_test, cat_features=current_cat_features)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=test_pool)

    preds = model.predict_proba(X_test_final)[:, 1]
    auc = roc_auc_score(y_test, preds)

    return auc


# ЗАПУСК ПОИСКА ----------------------------------------------------

with open('confirmed_features.txt', 'r') as f:
    selected_features = f.read()
    selected_features = selected_features.replace("' ", ' ').replace('[', '').replace(']', '').replace("'", '').replace(' ', '')
    selected_features = selected_features.split(',')
    selected_features.remove('month_cat_comb_day_TE')
    selected_features.remove('duration_ratio_age')
    selected_features.remove('balance_log_mul_duration')
    selected_features.remove('_duration_sqrt')
    selected_features.remove('poutcome_cat_comb_loan_TE')


X_train_final = X_train[selected_features]
X_test_final = X_test[selected_features]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Лучшие параметры:", study.best_params)
print("Лучший AUC:", study.best_value)

# 1. График важности гиперпараметров (Самое полезное!)
# Покажет, какие параметры сильнее всего влияли на рост AUC
fig_importances = vis.plot_param_importances(study)
# fig_importances.write_image(f"hyperparameters_analysis/param_importances_{name}.png")
fig_importances.show()

# 2. История оптимизации
# Красивый точечный график того, как рос AUC от попытки к попытке
fig_history = vis.plot_optimization_history(study)
# fig_history.write_image(f"hyperparameters_analysis/auc_history_{name}.png")
fig_history.show()

# 3. Графики зависимости (Parallel Coordinate Plot)
# Показывает, какие комбинации параметров работают лучше всего вместе
fig_parallel = vis.plot_parallel_coordinate(study, params=["learning_rate", "depth", "max_leaves", "scale_pos_weight"])
# fig_parallel.write_image(f"hyperparameters_analysis/params_combinations_{name}.png")
fig_parallel.show()

# 4. Сохранение всей истории в таблицу (Excel/CSV)
df_results = study.trials_dataframe()
df_results = df_results.sort_values(by='value', ascending=False)  # Сортируем от лучших к худшим
df_results.to_csv(f"hyperparameters_analysis/optuna_trials_history_{name}.csv", index=False)
print("История всех 50 попыток сохранена в optuna_trials_history.csv")
