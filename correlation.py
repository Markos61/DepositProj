##
from sklearn.model_selection import train_test_split
from train_functions import *
from config import name

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
    test_size=0.01,
    random_state=42,
    stratify=y)

categorical_features = X_train.select_dtypes(
    include=["object"]).columns.tolist()

# TE mean
for cat_feature in categorical_features:
    X_train, X_test = mean_target_encoding(X_train, X_test, cat_feature, y_train)
X_train, X_test = mean_target_encoding(X_train, X_test, 'job_education', y_train)
X_train, X_test = mean_target_encoding(X_train, X_test, 'month_contact', y_train)


with open('confirmed_features.txt', 'r') as f:
    selected_features = f.read()
    selected_features = selected_features.replace("' ", ' ').replace('[', '').replace(']', '').replace("'", '').replace(' ', '')
    selected_features = selected_features.split(',')
    selected_features.remove('month_cat_comb_day_TE')
    selected_features.remove('duration_ratio_age')
    selected_features.remove('balance_log_mul_duration')
    selected_features.remove('_duration_sqrt')
    selected_features.remove('poutcome_cat_comb_loan_TE')


X_train, X_test = X_train[selected_features], X_test[selected_features]

correlation_plot(X_train, name)
