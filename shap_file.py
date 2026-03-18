##
import numpy as np
import joblib
import pandas as pd
import shap
import catboost
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


model = joblib.load(r"CatBoostClassifier_model.pkl")

df = pd.read_csv("train.csv")
df = df.drop(columns=["id"])
X = df.drop("y", axis=1)
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

categorical_features = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

# Вклад фичей в результат
model.get_feature_importance(type="Interaction")

shap_values = model.get_feature_importance(
    data=catboost.Pool(X_test, cat_features=categorical_features),
    type="ShapValues"
)

params = ['age', 'job', 'marital', 'education', 'default', 'balance',
          'housing', 'loan', 'contact', 'day', 'month', 'duration',
          'campaign', 'pdays', 'previous', 'poutcome']

# Создаём Pool из данных
pool_test = catboost.Pool(
    X_test,
    cat_features=categorical_features,
    feature_names=params
)


shap_values = model.get_feature_importance(
    pool_test,
    type="ShapValues"
)

shap_values_features = shap_values[:, :-1]

# SHAP №1 для всех значений выборки
shap.summary_plot(
    shap_values_features,
    X_test,
    feature_names=params,
    show=False
)

plt.tight_layout()
plt.savefig("shap_test.png", dpi=300)

##
# SHAP №2 - Случайные объекты
sample_indices = np.random.choice(X_test.shape[0], 500, replace=False)

X_shap = X_test.iloc[sample_indices]
shap_values_shap = shap_values_features[sample_indices, :]

# Строим график
shap.summary_plot(
    shap_values_shap,
    X_shap,
    feature_names=params,
    show=False
)

plt.tight_layout()
plt.savefig("shap_random.png", dpi=300)

##
# SHAP №3 — Конкретный объект
i = 3

base_value = shap_values[i, -1]
shap_values_object = shap_values[i, :-1]

force_plot = shap.force_plot(
    base_value,
    shap_values_object,
    X_test.iloc[i],
    feature_names=params,
    show=False
)
shap.save_html("shap_1_object.html", force_plot)
