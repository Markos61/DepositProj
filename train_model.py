##
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")
df = df.drop(columns=["id"])
X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y)

categorical_features = X_train.select_dtypes(
    include=["object", "string"]).columns.tolist()

# hyperopt библиотека
# window expand подход к обучению
# факторный анализ
# отбор фичей baruto
# исключение фичей
# prometeus graphana - отслеживание модели
model = CatBoostClassifier(
    iterations=50000,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",  # QueryCrossEntropy  "AUC"
    cat_features=categorical_features,
    max_ctr_complexity=2,
    auto_class_weights="Balanced",
    l2_leaf_reg=5,
    border_count=64,
    bagging_temperature=1,
    early_stopping_rounds=100,
    verbose=False,
    # random_seed=42,
    task_type="GPU",
    devices="0",
)

model.fit(X_train, y_train,
          eval_set=(X_test, y_test),
          use_best_model=True,
          early_stopping_rounds=200)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


print("ROC-AUC:", roc_auc_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
feature_importances = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.get_feature_importance()
}).sort_values(by="importance", ascending=False)

print(feature_importances.head(10))

matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
print(matrix)

joblib.dump(model, r"CatBoostClassifier_model.pkl")

print("Модель сохранена")
