##
import pandas as pd
import joblib


model = joblib.load(r"CatBoost_ExpandingWindow_10000.pkl")
test = pd.read_csv("test.csv")
X_test = test.drop(columns=["id"])
y_pred_proba = model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({
    "id": test["id"],
    "y": y_pred_proba})
submission.to_csv("submission.csv", index=False)
