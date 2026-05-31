##
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import optuna.visualization as vis
from config import name
from train_functions import *
from lightgbm.basic import LightGBMError

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

model_type = "TabM"


def objective(trial):
    try:
        current_categorical_features = [f for f in categorical_features if f in selected_features]
        params = get_optimization_params(trial, "TabM", current_categorical_features)
        # train_pool = Pool(X_train_final, y_train, cat_features=current_cat_features)
        # test_pool = Pool(X_test_final, y_test, cat_features=current_cat_features)
        # model = CatBoostClassifier(**params)
        # model.fit(train_pool, eval_set=test_pool)

        model, preds = optuna_fit_model(model_type, params, X_train_final, y_train, X_test_final)

        # preds = model.predict_proba(X_test_final)[:, 1]

        auc = roc_auc_score(y_test, preds)

        return auc

    except:
        print(f"Попытка {trial.number} упала из-за ошибки")
        raise optuna.TrialPruned()


# ЗАПУСК ПОИСКА ----------------------------------------------------

with open('confirmed_features.txt', 'r') as f:
    selected_features = f.read()
    selected_features = selected_features.replace("' ", ' ').replace('[', '').replace(']', '').replace("'", '').replace(
        ' ', '')
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
fig_importances.show()
##
# 2. История оптимизации
# Красивый точечный график того, как рос AUC от попытки к попытке
fig_history = vis.plot_optimization_history(study)
fig_history.write_image(f"hyperparameters_analysis/auc_history_{model_type}_{name}.png")
fig_history.show()
##
# 3. Графики зависимости (Parallel Coordinate Plot)
# Показывает, какие комбинации параметров работают лучше всего вместе
fig_parallel = vis.plot_parallel_coordinate(study,
                                            params=["learning_rate", "max_depth", 'subsample', 'scale_pos_weight'])
fig_parallel.write_image(f"hyperparameters_analysis/params_combinations_{model_type}_{name}.png")
fig_parallel.show()
##
# 4. Сохранение всей истории в таблицу (Excel/CSV)
df_results = study.trials_dataframe()
df_results = df_results.sort_values(by='value', ascending=False)  # Сортируем от лучших к худшим
df_results.to_csv(f"hyperparameters_analysis/optuna_trials_history_{model_type}_{name}.csv", index=False)
print("История всех 50 попыток сохранена в optuna_trials_history.csv")
