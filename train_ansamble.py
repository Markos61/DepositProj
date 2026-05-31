##
import joblib
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from pytabkit import TabM_D_Classifier
from train_functions import *
from config import name
import warnings

warnings.filterwarnings('ignore')

# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ---------------------------------------
df = pd.read_csv("train.csv")
df = add_original_dataset(df, add=False)

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

# params, model_type = get_best_params(categorical_features, "CatBoost")

models = []
oof_preds = np.zeros(len(X))  # Массив для Out-of-Fold предсказаний
all_preds = []  # Для предсказаний на тестовой выборке

model_types = ["TabM", "CatBoost", "LightGBM"]
oof_dict = {model: np.zeros(len(X)) for model in model_types}
test_preds_dict = {model: np.zeros(len(test_orig)) for model in model_types}

print(f"Обучение ансамбля на {n_splits}-fold кросс-валидации...\n")


# 3. ЦИКЛ ОБУЧЕНИЯ ПО ФОЛДАМ --------------------------------------------
for model_type in model_types:
    print(f"Обучение модели {model_type}...")
    model_test_preds = np.zeros(len(test_orig))
    fold_aucs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        params, model_type = get_best_params(categorical_features, model_type)

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_test = test_orig.copy()

        # добавление псевдо-размеченных обучающих данных
        X_train, y_train = add_pseudo_data(X_train, X_pseudo, y_train, y_pseudo, add=False)

        categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

        # mean TE (Target Encoding)
        for cat_feature in categorical_features:
            X_train, X_val, X_test = mean_target_encoding(X_train, X_val, cat_feature, y_train, X_test)
        X_train, X_val, X_test = mean_target_encoding(X_train, X_val, 'job_education', y_train, X_test)
        X_train, X_val, X_test = mean_target_encoding(X_train, X_val, 'month_contact', y_train, X_test)

        X_train, X_val = X_train[selected_features], X_val[selected_features]
        X_test = X_test[selected_features]

        categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

        if model_type == 'CatBoost':
            params['cat_features'] = categorical_features

        model, fold_prob, test_predict = fit_model(model_type, params, X_train, y_train, X_val, y_val, X_test)

        all_preds.append(test_predict)

        oof_preds[val_idx] = fold_prob

        oof_dict[model_type][val_idx] = fold_prob

        model_test_preds += test_predict / n_splits

        models.append(model)
        current_auc = roc_auc_score(y_val, fold_prob)
        fold_aucs.append(current_auc)

        print(f"Fold {fold + 1} завершен. AUC: {current_auc:.5f}")
        joblib.dump(model, fr"models/{model_type}_{name}_fold_{fold + 1}.pkl")

    test_preds_dict[model_type] = model_test_preds

    print(f"[{model_type}] СРЕДНИЙ OOF AUC: {roc_auc_score(y, oof_dict[model_type]):.5f}")

create_submission_file(all_preds, test_id, name, 'ensemble')

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

show_roc_auc_curve(oof_preds, y, name, model_type)

# Матрица ошибок
matrix = confusion_matrix(y, oof_classes)
show_matrix(matrix, name, model_type)

# Анализ взаимодействий на последней модели
# get_features_importance(models[-1], X_train, name)

print(f"Все {n_splits} моделей сохранены успешно!")

df_oof = pd.DataFrame(oof_dict)
df_test_preds = pd.DataFrame(test_preds_dict)

df_oof['target'] = y.values

df_oof.to_csv("oof_predictions.csv", index=False)
df_test_preds.to_csv("test_predictions.csv", index=False)

print("Предсказания успешно сохранены на диск!")

X_meta = pd.DataFrame(oof_dict)
X_test_meta = pd.DataFrame(test_preds_dict)

meta_model = LogisticRegression(C=0.01, penalty='l2', random_state=42)
meta_model.fit(X_meta, y)

final_oof_preds = meta_model.predict_proba(X_meta)[:, 1]
final_test_preds = meta_model.predict_proba(X_test_meta)[:, 1]

print("\n" + "=" * 30)
print(f"ФИНАЛЬНЫЙ META-AUC: {roc_auc_score(y, final_oof_preds):.5f}")
print("=" * 30 + "\n")

for model_name, weight in zip(model_types, meta_model.coef_[0]):
    print(f"Вес {model_name}: {weight:.4f}")

create_submission_file(final_test_preds, test_id, name, 'meta_ensemble2')

joblib.dump(meta_model, 'meta_model_logistic_regression.pkl')
