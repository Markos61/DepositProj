##
import joblib
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from train_functions import *
from config import name

# ÇÀÃĞÓÇÊÀ ÄÀÍÍÛÕ ---------------------------------------------------

df = pd.read_csv("train.csv")

# ÄÎÁÀÂËÅÍÈÅ ÍÎÂÛÕ ÏĞÈÇÍÀÊÎÂ ----------------------------------------

df = create_new_features(df)

# ÑÎÇÄÀÍÈÅ ÂÛÁÎĞÎÊ --------------------------------------------------

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
    include=["object"]).columns.tolist()

# TE mean
for cat_feature in categorical_features:
    X_train, X_test = mean_target_encoding(X_train, X_test, cat_feature, y_train)
X_train, X_test = mean_target_encoding(X_train, X_test, 'job_education', y_train)
X_train, X_test = mean_target_encoding(X_train, X_test, 'month_contact', y_train)

# ÇÀÃĞÓÇÊÀ ÇÍÀ×ÈÌÛÕ ÏĞÈÇÍÀÊÎÂ ---------------------------------------------------

with open('confirmed_features.txt', 'r') as f:
    selected_features = f.read()
    selected_features = selected_features.replace("' ", ' ').replace('[', '').replace(']', '').replace("'", '').replace(' ', '')
    selected_features = selected_features.split(',')

X_train, X_test = X_train[selected_features], X_test[selected_features]

categorical_features = X_train.select_dtypes(
    include=["object"]).columns.tolist()

##
model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.0858,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    cat_features=categorical_features,
    auto_class_weights="Balanced",
    l2_leaf_reg=0.153154,
    verbose=True,
    # task_type="GPU",
    # devices="0",
    bootstrap_type='Bernoulli',
    random_strength=1.909318e-05,
    subsample=0.35747)

# ÎÁÓ×ÅÍÈÅ ------------------------------------------------------------------

model.fit(X_train, y_train,
          eval_set=(X_test, y_test),
          use_best_model=True,
          early_stopping_rounds=250)

# ÏĞÅÄÑÊÀÇÀÍÈÅ ---------------------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ÂÈÇÓÀËÈÇÀÖÈß ÎØÈÁÊÈ È ÌÅÒĞÈÊÈ ÊÀ×ÅÑÒÂÀ --------

show_metrics(model, name)

# ÊĞÈÂÀß ROC-AUC

show_roc_auc_curve(y_prob, y_test, name)

print('\n', "AUC:", roc_auc_score(y_test, y_prob), '\n')

# ÎÒ×¨Ò Î ÊËÀÑÑÈÔÈÊÀÖÈÈ -------------------------------------------------------

print('\n', classification_report(y_test, y_pred), '\n')
fi = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.get_feature_importance()
}).sort_values(by="importance", ascending=False)

print('\n', fi.head(10), '\n')

#  ÌÀÒĞÈÖÀ ÎØÈÁÎÊ --------------------------------------------------------------
matrix = confusion_matrix(y_test, y_pred)
show_matrix(matrix, name)

joblib.dump(model, fr"models/CatBoost_{name}.pkl")

# Âûâîä interaction strength (ñèëà ñîâìåñòíîãî ñèãíàëà ïğèçíàêîâ) ---------------
get_features_importance(model, X_train, name)
