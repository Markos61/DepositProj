##
import joblib
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from train_functions import *
from config import name

# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ---------------------------------------
df = pd.read_csv("train.csv")
df = create_new_features(df)

if "id" in df.columns:
    df = df.drop(columns=["id"])

X = df.drop("y", axis=1)
y = df["y"]

# 1.1 ЗАГРУЗКА ПСЕВДО-РАЗМЕЧЕННЫХ ДАННЫХ ----------------------------------

df_pseudo = pd.read_csv("train_pseudo.csv")
df_pseudo = create_new_features(df_pseudo)

if "id" in df_pseudo.columns:
    df_pseudo = df_pseudo.drop(columns=["id"])

X_pseudo = df_pseudo.drop("y", axis=1)
y_pseudo = df_pseudo["y"]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# ЗАГРУЗКА ЗНАЧИМЫХ ПРИЗНАКОВ ---------------------------------------------------

with open('confirmed_features.txt', 'r') as f:
    selected_features = f.read()
    selected_features = selected_features.replace("' ", ' ').replace('[', '').replace(']', '').replace("'", '').replace(' ', '')
    selected_features = selected_features.split(',')

##
# 2. НАСТРОЙКИ КРОСС-ВАЛИДАЦИИ И ПАРАМЕТРЫ ------------------------------
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

params = {
    'iterations': 3000,
    'learning_rate': 0.0858,
    'depth': 8,
    'loss_function': "Logloss",
    'eval_metric': "AUC",
    'cat_features': categorical_features,
    'auto_class_weights': "Balanced",
    'l2_leaf_reg': 0.153154,
    'bootstrap_type': 'Bernoulli',
    'random_strength': 1.909318e-05,
    'subsample': 0.35747,
    'verbose': False,
    'task_type': "GPU",
    'devices': "0"
}

models = []
oof_preds = np.zeros(len(X))  # Массив для Out-of-Fold предсказаний
fold_aucs = []

print(f"Обучение {n_splits}-fold кросс-валидации...\n")

# 3. ЦИКЛ ОБУЧЕНИЯ ПО ФОЛДАМ --------------------------------------------
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # добавление псевдо-размеченных обучающих данных
    X_train, y_train = add_pseudo_data(X_train, X_pseudo, y_train, y_pseudo, add=True)

    # mean TE (Target Encoding)
    for cat_feature in categorical_features:
        X_train, X_val = mean_target_encoding(X_train, X_val, cat_feature, y_train)
    X_train, X_val = mean_target_encoding(X_train, X_val, 'job_education', y_train)
    X_train, X_val = mean_target_encoding(X_train, X_val, 'month_contact', y_train)

    X_train, X_val = X_train[selected_features], X_val[selected_features]
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    model = CatBoostClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=200,
        verbose=500  # Показывать прогресс каждые 500 итераций
    )

    # Предсказание вероятностей для текущего фолда
    fold_prob = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = fold_prob

    # Сохранение модели и метрики
    models.append(model)
    current_auc = roc_auc_score(y_val, fold_prob)
    fold_aucs.append(current_auc)

    print(f"Fold {fold + 1} завершен. AUC: {current_auc:.5f}")
    joblib.dump(model, fr"models/CatBoost_{name}_fold_{fold + 1}.pkl")

# 4. ИТОГОВЫЕ МЕТРИКИ (OOF) --------------------------------------------
mean_auc = np.mean(fold_aucs)
std_auc = np.std(fold_aucs)
total_oof_auc = roc_auc_score(y, oof_preds)

print("\n" + "=" * 30)
print(f"СРЕДНИЙ AUC ПО ФОЛДАМ: {mean_auc:.5f} (+/- {std_auc:.5f})")
print(f"ОБЩИЙ OOF ROC-AUC: {total_oof_auc:.5f}")
print("=" * 30 + "\n")

# 5. ВИЗУАЛИЗАЦИЯ И ОТЧЕТЫ (на основе OOF предсказаний) ----------------
# Для отчетов по метрикам используем порог 0.5 к OOF предсказаниям
oof_classes = (oof_preds > 0.5).astype(int)

print("Classification Report (OOF):")
print(classification_report(y, oof_classes))

# Используем последнюю обученную модель для визуализации важности фичей
# show_metrics(models[-1], name)

show_roc_auc_curve(oof_preds, y, name)

# Матрица ошибок
matrix = confusion_matrix(y, oof_classes)
show_matrix(matrix, name)

# Анализ взаимодействий на последней модели
get_features_importance(models[-1], X_train, name)

print(f"Все {n_splits} моделей сохранены успешно!")
