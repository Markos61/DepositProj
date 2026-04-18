##
from BorutaShap import BorutaShap
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from train_functions import *
from config import name
matplotlib.use('Agg')


df = pd.read_csv("train.csv")

# ДОБАВЛЕНИЕ НОВЫХ ПРИЗНАКОВ ----------------------------------------

df = create_new_features(df)

# СОЗДАНИЕ ВЫБОРОК --------------------------------------------------

df = df.drop(columns=["id"])
X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y)

categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

# TE mean
for cat_feature in categorical_features:
    X_train, X_test = mean_target_encoding(X_train, X_test, cat_feature, y_train)
X_train, X_test = mean_target_encoding(X_train, X_test, 'job_education', y_train)
X_train, X_test = mean_target_encoding(X_train, X_test, 'month_contact', y_train)

##

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.0858,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    l2_leaf_reg=0.153154,
    early_stopping_rounds=100,
    bootstrap_type='Bernoulli',
    random_strength=1.909318e-05,
    subsample=0.35747,
    verbose=False,
    task_type="GPU",
    devices="0",
)
# Инициализация
Feature_Selector = BorutaShap(model=model, importance_measure='shap', classification=True)

Feature_Selector.fit(X=X_train, y=y_train, n_trials=20, random_state=42)

fig, ax = plt.subplots(figsize=(12, 15))
Feature_Selector.plot(X_size=12, figsize=(12, 15))

plt.savefig(f"boruta_result_{name}.png", dpi=600, bbox_inches='tight')

confirmed_features = Feature_Selector.accepted
tentative_features = Feature_Selector.tentative
all_selected_features = confirmed_features + tentative_features

print(f"Итого отобрано признаков: {len(all_selected_features)}")

X_train_filtered = X_train[all_selected_features]

with open('selected_features.txt', 'w') as f:
    f.write(f'{all_selected_features}')

with open('confirmed_features.txt', 'w') as f:
    f.write(f'{confirmed_features}')

with open('tentative_features.txt', 'w') as f:
    f.write(f'{tentative_features}')
