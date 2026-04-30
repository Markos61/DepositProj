##
import joblib
import shap
import catboost
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
from train_functions import *
from config import name

model = joblib.load(fr"models\CatBoost_{name}_fold_1.pkl")

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
    stratify=y
)

categorical_features = X_train.select_dtypes(
    include=["object"]).columns.tolist()

# TE mean
for cat_feature in categorical_features:
    X_train, X_test = mean_target_encoding(X_train, X_test, cat_feature, y_train)
X_train, X_test = mean_target_encoding(X_train, X_test, 'job_education', y_train)
X_train, X_test = mean_target_encoding(X_train, X_test, 'month_contact', y_train)


# ЗАГРУЗКА ЗНАЧИМЫХ ПРИЗНАКОВ ---------------------------------------------------
selected_features = get_selected_features()

X_train, X_test = X_train[selected_features], X_test[selected_features]
X_test = X_test[:5000]
categorical_features = X_train.select_dtypes(
    include=["object"]).columns.tolist()
# Вклад фичей в результат
model.get_feature_importance(type="Interaction")

shap_values = model.get_feature_importance(
    data=catboost.Pool(X_test, cat_features=categorical_features),
    type="ShapValues"
)

params = list(X_train.columns)

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

# SHAP для всех значений выборки
shap.summary_plot(
    shap_values_features,
    X_test,
    feature_names=params,
    show=False,
    max_display=len(params)
)

plt.tight_layout()
plt.savefig(f"shap_files/shap_test_{name}.png", dpi=300)
