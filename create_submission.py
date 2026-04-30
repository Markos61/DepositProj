##
import joblib
from train_functions import *
from config import name

by_folds = True
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

# Создание фичей, как при обучении
test = create_new_features(test)
train = create_new_features(train)

y_train = train["y"]

categorical_features = train.select_dtypes(
    include=["object"]).columns.tolist()

# 1.1 ЗАГРУЗКА ПСЕВДО-РАЗМЕЧЕННЫХ ДАННЫХ ----------------------------------

df_pseudo = pd.read_csv("train_pseudo.csv")
df_pseudo = create_new_features(df_pseudo)

if "id" in df_pseudo.columns:
    df_pseudo = df_pseudo.drop(columns=["id"])

X_pseudo = df_pseudo.drop("y", axis=1)
y_pseudo = df_pseudo["y"]

train, y_train = add_pseudo_data(train, X_pseudo, y_train, y_pseudo, add=True)

# TE mean
for cat_feature in categorical_features:
    train, test = mean_target_encoding(train, test, cat_feature, y_train)
train, test = mean_target_encoding(train, test, 'job_education', y_train)
train, test = mean_target_encoding(train, test, 'month_contact', y_train)


# ЗАГРУЗКА ЗНАЧИМЫХ ПРИЗНАКОВ ---------------------------------------------------
selected_features = get_selected_features()

test_id = test['id']
train, test = train[selected_features], test[selected_features]
##

if by_folds:
    folds = 5
    all_preds = []
    for fold in range(folds):
        model = joblib.load(fr"models/CatBoost_{name}_fold_{fold + 1}.pkl")
        all_preds.append(model.predict_proba(test)[:, 1])
        print(f"Model {fold+1} prediction done.")
    # Усредняем
    final_submission_probs = np.mean(all_preds, axis=0)

    # Сохраняем результат
    submission = pd.DataFrame({
        "id": test_id,
        "y": final_submission_probs
    })
    submission.to_csv(fr"submissions/submission_{name}_folds.csv", index=False)

else:
    model = joblib.load(fr"models/CatBoost_{name}.pkl")
    y_pred_proba = model.predict_proba(test)[:, 1]
    submission = pd.DataFrame({
        "id": test_id,
        "y": y_pred_proba})
    submission.to_csv(fr"submissions/submission_{name}.csv", index=False)

##
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Применяем твою функцию создания фичей к обоим наборам
train_proc = create_new_features(train.copy())
test_proc = create_new_features(test.copy())

X_test = test_proc.drop(columns=["id"])

# 2. ПОЛУЧЕНИЕ ПСЕВДО-МЕТОК --------------------------------------------
all_preds = []
for fold in range(5):
    model = joblib.load(f"models/CatBoost_{name}_fold_{fold + 1}.pkl")
    all_preds.append(model.predict_proba(X_test)[:, 1])

# Усредняем вероятности
test_probs = np.mean(all_preds, axis=0)

confident_mask = (test_probs < 0.001) | (test_probs > 0.98)
pseudo_df = test_proc[confident_mask].copy()
pseudo_df['y'] = (test_probs[confident_mask] > 0.5).astype(int)

print(f"Добавлено псевдо-лейблов: {len(pseudo_df)}")
print(f"Из них класс 1: {pseudo_df['y'].sum()}")

# 3. СОЗДАНИЕ РАСШИРЕННОГО ТРЕНИРОВОЧНОГО НАБОРА -------------------------
# Убеждаемся, что колонки совпадают
cols_to_keep = [col for col in train_proc.columns if col != 'id']
combined_train = pd.concat([train_proc[cols_to_keep], pseudo_df[cols_to_keep]], axis=0).reset_index(drop=True)

X_combined = combined_train.drop("y", axis=1)
y_combined = combined_train["y"]
##
confident_test_original = test.loc[pseudo_df.index].copy()

# Добавляем предсказанные метки
confident_test_original['y'] = pseudo_df['y']

# Теперь у тебя есть чистый датасет с оригинальными колонками и метками
print(f"Форма сохраненного датасета: {confident_test_original.shape}")
confident_test_original.to_csv("pseudo_train.csv", index=False)
