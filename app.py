# -*- coding: cp1251 -*-
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from shap_file_stremlit import *
from train_functions import *
from config import inference_model_name
from streamlit_option_menu import option_menu
import torch

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    /* Стиль для всех кнопок в приложении */
    div.stButton > button:first-child {
        background-color: #e52d27; /* Цвет фона */
        color: white;               # Цвет текста
        border-radius: 10px;        /* Скругление углов */
        border: 2px solid #e52d27;  /* Граница */
        font-family: "Segoe UI", sans-serif;
        height: 3em;
        width: 100%;                /* Растянуть на всю ширину колонки */
        transition: 0.3s;
    }

    /* Эффект при наведении */
    div.stButton > button:first-child:hover {
        background-color: 	#bcbdbc;
        border-color: #bcbdbc;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Значения для интерфейса ------
marital_dict = {'не в браке': 'single', 'в браке': 'married', 'в разводе': 'divorced'}
job_dict = {
    'администратор': 'admin.', 'рабочий': 'blue-collar',
    'предприниматель': 'entrepreneur', 'домохозяйка': 'housemaid',
    'менеджмент / руководитель': 'management', 'пенсионер': 'retired',
    'самозанятый': 'self-employed', 'занят в сфере услуг': 'services',
    'студент': 'student', 'технический специалист': 'technician',
    'безработный': 'unemployed', 'неизвестно': 'unknown'}
education_dict = {'начальное': 'primary', 'среднее': 'secondary',
                  'высшее': 'tertiary', 'неизвестно': 'unknown'}
yes_or_no_dict = {'да, есть': 'yes', 'нет': 'no'}
contact_dict = {'мобильный телефон (смартфон)': 'cellular', 'стационарный телефон': 'telephone',
                'неизвестно': 'unknown'}
poutcome_dict = {'успех': 'success', 'отказ': 'failure', 'другое': 'other', 'неизвестно': 'unknown'}
months_dict = {'январь': 'jan', 'февраль': 'feb', 'март': 'mar', 'апрель': 'apr',
               'май': 'may', 'июнь': 'jun', 'июль': 'jul', 'август': 'aug', 'сентябрь': 'sep',
               'октябрь': 'oct', 'ноябрь': 'nov', 'декабрь': 'dec'}
# ---------
num_of_models = 15

original_torch_load = torch.load


def safe_torch_load(*args, **kwargs):
    kwargs['map_location'] = 'cpu'
    return original_torch_load(*args, **kwargs)


torch.load = safe_torch_load

models = []
for model_type in ["TabM", "CatBoost", "LightGBM"]:
    for idx in range(1, 6):
        model = joblib.load(fr"models/{model_type}_{inference_model_name}_fold_{idx}.pkl")
        models.append(model)

meta_model = joblib.load(fr"models/meta_model_logistic_regression.pkl")

params = ['age', 'job', 'marital', 'education', 'default', 'balance',
          'housing', 'loan', 'contact', 'day', 'month', 'duration',
          'campaign', 'pdays', 'previous', 'poutcome']

numeric_features = ['age', 'balance', 'day', 'duration',
                    'campaign', 'pdays', 'previous']

categorical_features = ['job', 'marital', 'education', 'default',
                        'housing', 'loan', 'contact', 'month', 'poutcome']

st.write("")
col1, col2 = st.columns([20, 1])

with col2:
    st.image('image.svg', width=40)

selected = option_menu(
    menu_title='МОДЕЛЬ ДЛЯ ОЦЕНКИ ВЕРОЯТНОСТИ ПОДПИСАНИЯ ДОГОВОРА НА ОТКРЫТИЕ ДЕПОЗИТА',
    options=['Одиночный прогноз', 'Аналитика', 'Множественный прогноз'],  # Названия вкладок
    icons=["file-binary-fill", "bar-chart-fill", "filetype-csv"],  # Иконки для вкладок
    menu_icon="stack",  # Иконка самого меню
    default_index=0,  # Какая вкладка открыта по умолчанию
    orientation="horizontal",  # Делает меню горизонтальным (как вкладки)
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"font-size": "20px"},
        "menu-title": {
            "color": "#FF4B4B",
            "font-family": "Segoe UI",
            "font-weight": "bold",
            "font-size": "18px",

        },
        "nav-link": {
            "font-family": "Segoe UI",
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#eee"}})

if selected == 'Одиночный прогноз':
    st.write("Введите параметры клиента:")
    col_1, col_2 = st.columns(2)

    with col_1:
        st.divider()
        age = st.slider("Возраст (age)", min_value=18, max_value=95, value=35)
        st.divider()
        job_rus = st.selectbox("Тип занятости (job)",
                               ['администратор', 'рабочий', 'предприниматель', 'домохозяйка',
                                'менеджер / руководитель', 'пенсионер', 'самозанятый', 'занят в сфере услуг',
                                'студент', 'технический специалист',
                                'безработный', 'неизвестно'])
        job = job_dict[job_rus]

        st.divider()
        marital_rus = st.selectbox("Семейное положение (marital)",
                                   ['не в браке', 'в браке', 'в разводе'])
        marital = marital_dict[marital_rus]
        st.divider()
        education_rus = st.selectbox("Уровень образования (education)",
                                     ['начальное', 'среднее', 'высшее', 'неизвестно'])
        education = education_dict[education_rus]

        st.divider()
        default_rus = st.selectbox("Есть ли просрочка платежа по кредиту? (default)",
                                   ['да, есть', 'нет'])
        default = yes_or_no_dict[default_rus]

        st.divider()
        balance = st.slider("Среднегодовой баланс в переводе на Евро (balance)", min_value=-8019, max_value=99717,
                            value=0, step=1)
        st.divider()
        housing_rus = st.selectbox("Есть ли жилищный кредит? (housing)",
                                   ['да, есть', 'нет'])
        housing = yes_or_no_dict[housing_rus]

        st.divider()

        loan_rus = st.selectbox("Есть ли потребительский кредит? (loan)",
                                ['да, есть', 'нет'])
        loan = yes_or_no_dict[loan_rus]

        st.divider()

    with col_2:

        st.divider()
        contact_rus = st.selectbox("Тип контакта (contact)",
                                   ['мобильный телефон (смартфон)', 'стационарный телефон', 'неизвестно'])
        contact = contact_dict[contact_rus]

        st.divider()
        day = st.slider("Последний день контакта за месяц (day)", 1, 31, 15)
        st.divider()
        month_rus = st.selectbox("Последний месяц контакта за год (month)",
                                 ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь',
                                  'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь'])
        month = months_dict[month_rus]

        st.divider()
        duration = st.slider("Продолжительность последнего контакта в секундах (duration)", min_value=1, max_value=4918,
                             value=1, step=1)
        st.divider()
        campaign = st.slider("Количество контактов, совершённых в ходе кампании (campaign)", 1, 63, 1)
        st.divider()
        pdays = st.slider("Количество дней с момента последнего обращения (pdays)",
                          -1, 871, 0)
        st.divider()
        previous = st.slider("Количество контактов, совершённых до этой кампании (previous)", 0, 200, 0)
        st.divider()
        poutcome_rus = st.selectbox("Результат предыдущей маркетинговой кампании (poutcome)",
                                    ['успех', 'отказ', 'другое', 'неизвестно'])
        poutcome = poutcome_dict[poutcome_rus]

        st.divider()

    if st.button("Сделать прогноз"):
        features = np.array([[age, job, marital, education, default, balance,
                              housing, loan, contact, day, month, duration,
                              campaign, pdays, previous, poutcome]])

        df = pd.DataFrame([features[0]], columns=params)
        for col in numeric_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = create_new_features(df)

        # подготовка данных
        df = preprocessing(df)

        new_features = df.values  # Для моделей, которые используют созданные фичи вместо features

        probs = {'TabM': [], 'CatBoost': [], 'LightGBM': []}

        for idx in range(num_of_models):
            new_features_copy = new_features.copy()
            if idx <= 4:
                probs['TabM'].append(models[idx].predict_proba(df)[0][1])
            elif 5 <= idx <= 9:
                probs['CatBoost'].append(models[idx].predict_proba(new_features)[0][1])
            else:
                cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
                for col in cat_cols:
                    df[col] = df[col].astype("category")
                probs['LightGBM'].append(models[idx].predict_proba(df)[0][1])

        probs = {'TabM': [np.mean(probs['TabM'])],
                 'CatBoost': [np.mean(probs['CatBoost'])],
                 'LightGBM': [np.mean(probs['LightGBM'])]}
        probs = pd.DataFrame(probs)

        prob = meta_model.predict_proba(probs)[0][1]

        if prob > 0.37:
            color1, color2 = '#32e111', '#2bc10f'
            text = 'Клиент сделает депозит'
        else:
            color1, color2 = '#e52d27', '#b31217'
            text = 'Клиент не сделает депозит'

        st.markdown(f"""
            <style>
                    .result-card {{
                        animation: fadeInUp 0.6s ease-out;
                    }}

                    @keyframes fadeInUp {{
                        from {{
                            opacity: 0;
                            transform: translateY(20px);
                        }}
                        to {{
                            opacity: 1;
                            transform: translateY(0);
                        }}
                    }}
                    </style>
            <div style="
                background-color: #ffffff; 
                padding: 20px; 
                border-radius: 15px; 
                border: 1px solid #eee;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
                text-align: center;">
                <h3 style="font-family: 'Segoe UI'; color: #31333F; margin: 0; font-size: 18px;
                background: -webkit-linear-gradient({color1}, {color2});
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;">
                    {text}
                </h3>
                <p style="
                    font-family: 'Segoe UI'; 
                    font-size: 70px; 
                    font-weight: 800; 
                    background: -webkit-linear-gradient({color1}, {color2});
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin: 10px 0;">
                    {str(round(prob * 100, 2)).replace('.', ',')}%
                </p>
            </div>
            """, unsafe_allow_html=True)

        categorical_features = df.select_dtypes(
            include=["object", "string", "category"]).columns.tolist()

        params = list(df.columns)
        for idx in range(num_of_models):
            if 5 <= idx <= 9:
                html_file = save_force_plot(models[idx], df, categorical_features, params,
                                            f"shap_force_{idx-5}.html")
            else:
                pass

if selected == 'Аналитика':
    for idx in range(5):
        st.write(f'Решающие признаки для {idx + 1}-й модели')
        st.components.v1.html(
            open(f"shap_force_{idx}.html", 'r', encoding='utf-8').read(),
            height=170,
            scrolling=True)

if selected == 'Множественный прогноз':
    uploaded = st.file_uploader("Загрузите данные в виде csv-файла", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        # подготовка данных
        df = create_new_features(df)
        df = preprocessing(df)
        new_features = df.values

        probs = {'TabM': [], 'CatBoost': [], 'LightGBM': []}

        for idx in range(num_of_models):
            new_features_copy = new_features.copy()
            if idx <= 4:
                probs['TabM'].append(models[idx].predict_proba(df)[:, 1])
            elif 5 <= idx <= 9:
                probs['CatBoost'].append(models[idx].predict_proba(new_features)[:, 1])
            else:
                cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
                for col in cat_cols:
                    df[col] = df[col].astype("category")
                probs['LightGBM'].append(models[idx].predict_proba(df)[:, 1])

        tabm_mean = np.mean(probs['TabM'], axis=0) if probs['TabM'] else np.zeros(len(df))
        cat_mean = np.mean(probs['CatBoost'], axis=0) if probs['CatBoost'] else np.zeros(len(df))
        lgb_mean = np.mean(probs['LightGBM'], axis=0) if probs['LightGBM'] else np.zeros(len(df))
        matrix_data = np.column_stack([tabm_mean, cat_mean, lgb_mean])

        probs_df = pd.DataFrame(matrix_data, columns=['TabM', 'CatBoost', 'LightGBM'])

        prob = meta_model.predict_proba(probs_df.values)[:, 1]

        df["probability"] = prob
        df['result'] = (df['probability'] >= 0.37).astype(int)

        st.write(df[['probability', 'result']])

        df.to_csv(r"prediction_results.csv", index=False)
        st.info("Результаты сохранены в prediction_results.csv")
