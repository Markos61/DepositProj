# -*- coding: cp1251 -*-
import streamlit as st
import numpy as np
import joblib
from shap_file_stremlit import *
from train_functions import create_new_features
from config import name


model = joblib.load(fr"models/CatBoost_{name}.pkl")
tab1, tab2 = st.tabs(['Прогноз', 'Аналитика'])

with tab1:
    st.title("Прогноз подписания договора на депозит")

    st.write("Введите параметры клиента:")

    params = ['age', 'job', 'marital', 'education', 'default', 'balance',
              'housing', 'loan', 'contact', 'day', 'month', 'duration',
              'campaign', 'pdays', 'previous', 'poutcome']
    numeric_features = [
        'age', 'balance', 'day', 'duration',
        'campaign', 'pdays', 'previous']
    categorical_features = [
        'job', 'marital', 'education', 'default',
        'housing', 'loan', 'contact', 'month', 'poutcome']

    age = st.slider("Возраст (age)", min_value=18, max_value=95, value=35)

    job = st.selectbox("Профессия (job)",
                       ['admin.', 'blue-collar', 'entrepreneur', 'housemaid',
                        'management', 'retired', 'self-employed',
                        'services', 'student', 'technician', 'unemployed', 'unknown'])

    marital = st.selectbox("Семейное положение (marital)",
                           ['single', 'married', 'divorced'])

    education = st.selectbox("Образование (education)",
                             ['primary', 'secondary', 'tertiary', 'unknown'])

    default = st.selectbox("Есть дефолт? (default)",
                           ['yes', 'no'])

    balance = st.slider("Баланс счета (balance)", min_value=-8019, max_value=99717,
                        value=0, step=1)

    housing = st.selectbox("Ипотека (housing)",
                           ['yes', 'no'])

    loan = st.selectbox("Потребительский кредит (loan)",
                        ['yes', 'no'])

    contact = st.selectbox("Тип контакта (contact)",
                           ['cellular', 'telephone', 'unknown'])

    day = st.slider("День месяца (day)", 1, 31, 15)

    month = st.selectbox("Месяц контакта (month)", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])

    duration = st.slider("Длительность звонка, сек (duration)", min_value=1, max_value=4918,
                         value=1, step=1)

    campaign = st.slider("Количество контактов (campaign)", 1, 63, 1)

    pdays = st.slider("Дней с последнего контакта (pdays)", -1, 871, 0)

    previous = st.slider("Количество предыдущих контактов (previous)", 0, 200, 0)

    poutcome = st.selectbox("Результат прошлой кампании (poutcome)",
                            ['success', 'failure', 'other', 'unknown'])

    if st.button("Прогноз"):
        features = np.array([[age, job, marital, education, default, balance,
                              housing, loan, contact, day, month, duration,
                              campaign, pdays, previous, poutcome]])

        df = pd.DataFrame([features[0]], columns=params)
        for col in numeric_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = create_new_features(df)
        new_features = df.values  # Для моделей, которые используют созданные фичи вместо features

        # prob = model.predict_proba(features)[0][1]
        prob = model.predict_proba(new_features)[0][1]
        st.success(f"Вероятность подписания: {prob:.2%}")

        # Создаём force plot и сохраняем HTML
        # html_file = save_force_plot(model, features, categorical_features, params)
        categorical_features = df.select_dtypes(
            include=["object", "string"]).columns.tolist()
        params = list(df.columns)
        html_file = save_force_plot(model, new_features, categorical_features, params)
        # Встраиваем интерактивный HTML
        st.components.v1.html(
            open(html_file, 'r', encoding='utf-8').read(),
            height=700
        )

    st.subheader("Массовая загрузка клиентов")
    uploaded = st.file_uploader("Загрузите CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        predictions = model.predict_proba(df)[:, 1]
        df["probability"] = predictions
        st.write(df)
        df.to_csv(r"prediction_results.csv", index=False)
        st.info("Результаты сохранены в prediction_results.csv")
        # streamlit run app.py

with tab2:
    st.title("Аналитика модели")
