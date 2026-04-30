##
import joblib
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from train_functions import *
from config import name
import warnings
warnings.filterwarnings('ignore')

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

test_orig = pd.read_csv("test.csv")
test_orig = create_new_features(test_orig)
test_id = test_orig['id']

if "id" in df_pseudo.columns:
    df_pseudo = df_pseudo.drop(columns=["id"])

X_pseudo = df_pseudo.drop("y", axis=1)
y_pseudo = df_pseudo["y"]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# ЗАГРУЗКА ЗНАЧИМЫХ ПРИЗНАКОВ ---------------------------------------------------
selected_features = get_selected_features()


# 2. НАСТРОЙКИ КРОСС-ВАЛИДАЦИИ И ПАРАМЕТРЫ ------------------------------
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

params = {
    'iterations': 50000,
    'learning_rate': 0.0349,
    'depth': 7,
    'loss_function': "Logloss",
    'eval_metric': "AUC",
    'cat_features': categorical_features,
    'l2_leaf_reg': 0.45,
    'scale_pos_weight': 4,
    'bootstrap_type': 'Bernoulli',
    'random_strength': 0.826,
    'subsample': 0.208,
    'min_data_in_leaf': 12,
    'max_bin': 224,
    'leaf_estimation_iterations': 2,
    'grow_policy': 'Lossguide',
    'max_leaves': 54,
    'verbose': False,
    'task_type': "GPU",
    'devices': "0"
}

models = []
oof_preds = np.zeros(len(X))  # Массив для Out-of-Fold предсказаний
fold_aucs = []
all_preds = []  # Для предсказаний на тестовой выборке

print(f"Обучение ансамбля на {n_splits}-fold кросс-валидации...\n")

# 3. ЦИКЛ ОБУЧЕНИЯ ПО ФОЛДАМ --------------------------------------------
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test = test_orig.copy()

    # добавление псевдо-размеченных обучающих данных
    X_train, y_train = add_pseudo_data(X_train, X_pseudo, y_train, y_pseudo, add=True)

    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    # mean TE (Target Encoding)
    for cat_feature in categorical_features:
        X_train, X_val, X_test = mean_target_encoding(X_train, X_val, cat_feature, y_train, X_test)
    X_train, X_val, X_test = mean_target_encoding(X_train, X_val, 'job_education', y_train, X_test)
    X_train, X_val, X_test = mean_target_encoding(X_train, X_val, 'month_contact', y_train, X_test)

    X_train, X_val = X_train[selected_features], X_val[selected_features]
    X_test = X_test[selected_features]

    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    params['cat_features'] = categorical_features

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

    # Предсказания для тестовой выборки
    all_preds.append(model.predict_proba(X_test)[:, 1])


create_submission_file(all_preds, test_id, name)

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

# show_metrics(models[-1], name)

show_roc_auc_curve(oof_preds, y, name)

# Матрица ошибок
matrix = confusion_matrix(y, oof_classes)
show_matrix(matrix, name)

# Анализ взаимодействий на последней модели
get_features_importance(models[-1], X_train, name)

print(f"Все {n_splits} моделей сохранены успешно!")
