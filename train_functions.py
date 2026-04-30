import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from sklearn import metrics
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def show_roc_auc_curve(y_prob, y_test, name):
    """
    Функция для визуализации кривой ROC-AUC
    :param y_prob: предсказания
    :param y_test: тестовая выборка
    :param name: имя файла
    :return: None
    """
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    auc = metrics.roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.6f' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Reference Line')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Кривая ROC-AUC')
    plt.legend(loc="lower right")

    # Сохраняем график
    plt.savefig(f"metrics/ROC_AUC_{name}.png", dpi=500, bbox_inches='tight')
    plt.close()


def show_metrics(model, name):
    """
    Функция для визуализации ошибки и метрики качества
    :param model: обученная модель
    :param name: имя файла
    :return: None
    """
    results = model.get_evals_result()
    iterations = range(len(results['learn']['Logloss']))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Logloss', color='tab:red')
    ax1.plot(iterations, results['learn']['Logloss'], color='tab:red', alpha=0.5, label='Train Logloss')
    ax1.plot(iterations, results['validation']['Logloss'], color='red', label='Test Logloss')

    ax2 = ax1.twinx()
    ax2.set_ylabel('AUC', color='tab:blue')
    ax2.plot(iterations, results['validation']['AUC'], color='tab:blue', label='Test AUC')

    plt.title('Динамика обучения')
    plt.savefig(f"metrics/learning_curves_{name}.png", dpi=500)
    plt.close()


def get_features_importance(model, X_train, name):
    """
    Функция для вывода и сохранения силы совместного сигнала признаков.
    :param model - модель предобученная
    :param X_train - выборка
    :param name - дополнение к имени файла
    """
    interactions = model.get_feature_importance(type="Interaction")
    inter_df = pd.DataFrame(interactions, columns=[
        "feature_1", "feature_2", "importance"
    ])

    inter_df["feature_1"] = inter_df["feature_1"].astype(int).apply(lambda x: X_train.columns[x])
    inter_df["feature_2"] = inter_df["feature_2"].astype(int).apply(lambda x: X_train.columns[x])

    inter_df = inter_df.sort_values(by="importance", ascending=False)
    print('\n\nCила совместного сигнала признаков:\n')
    print(inter_df.head(20))
    print('\n\n')

    inter_df.to_excel(f'interaction_strength/interaction strength_{name}.xlsx', index=False)


def add_pseudo_data(X_train, X_pseudo, y_train, y_pseudo, add=True):
    """
    Функция для добавления псевдо-размеченных обучающих данных
    :param add: параметр для включения и отключения добавления псевдо-размеченных обучающих данных
    :param X_train: обучающая выборка
    :param X_pseudo: псевдо-размеченная обучающая выборка
    :param y_train: метки обучающей выборки
    :param y_pseudo: псевдо-метки
    :return: X_train, y_train
    """
    if add:
        X_train = pd.concat([X_train, X_pseudo], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_train, y_pseudo], axis=0).reset_index(drop=True)

    return X_train, y_train


def mean_target_encoding(X_train, X_val, column, y_train, X_test=None):
    """
    Функция для target encoding
    :param X_train: обучающая выборка
    :param X_val: валидационная выборка
    :param column: колонка для преобразования target encoding mean
    :param y_train: метки обучающей выборки
    :param X_test: тестовая выборка (опционально)
    :return: X_train, X_val
    """
    target_means = y_train.groupby(X_train[column]).mean()
    X_train[f'{column}_TE'] = X_train[column].map(target_means)
    X_val[f'{column}_TE'] = X_val[column].map(target_means)
    
    global_mean = y_train.mean()

    X_train[f'{column}_TE'] = X_train[f'{column}_TE'].fillna(global_mean)
    X_val[f'{column}_TE'] = X_val[f'{column}_TE'].fillna(global_mean)

    noise = np.random.normal(0, 0.001, X_train[f'{column}_TE'].shape)
    X_train[f'{column}_TE'] += noise

    if X_test is not None:
        X_test[f'{column}_TE'] = X_test[column].map(target_means)
        X_test[f'{column}_TE'] = X_test[f'{column}_TE'].fillna(global_mean)
        return X_train, X_val, X_test
    else:
        return X_train, X_val


def count_encoding(df, column1, column2=None):
    """
    Функция для частотного кодирования категориальных признаков (count encoding)
    :param df: набор данных
    :param column1: признак №1
    :param column2: признак №2
    :return: Набор данных с новой колонкой
    """
    if column2:
        df[f'{column1}_{column2}'] = df[column1].astype(str) + "_" + df[column2].astype(str)
        df[f'{column1}_{column2}_counts'] = df.groupby(f'{column1}_{column2}')[f'{column1}_{column2}'].transform('count')
    else:
        df[f'{column1}_counts'] = df.groupby(f'{column1}')[f'{column1}'].transform('count')
    return df


def create_new_features(df):
    """Функция для создания нескольких признаков для использования в обучении.
    :param df - исходный df
    :return df с новыми фичами
    """
    # дополнительные данные

    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

    # добавление фичей
    df = add_feature(df, 'balance', 'log')
    df = add_feature(df, 'duration', 'log')
    df = add_feature(df, 'pdays', 'log')
    df = add_feature(df, 'age', 'log')

    df = add_feature(df, 'balance_log', 'mul', 'duration')
    df = add_feature(df, 'balance', 'ratio', 'campaign')

    df = add_feature(df, 'age', 'mul', 'duration_log')
    df = add_feature(df, 'balance_log', 'ratio', 'age')
    df = add_feature(df, 'duration', 'ratio', 'day')
    df = add_feature(df, 'balance', 'ratio', 'duration')
    df = add_feature(df, 'duration', 'ratio', 'age')

    df = add_feature(df, 'month', 'cat_comb', 'day')
    df = add_feature(df, 'poutcome', 'cat_comb', 'housing')
    df = add_feature(df, 'poutcome', 'cat_comb', 'loan')
    df = add_feature(df, 'duration', 'ratio', 'campaign')

    df = add_feature(df, 'contact', 'cat_comb', 'month_cat_comb_day')

    df['balance_per_age'] = df['balance'] / (df['age_log'] + 1e-6)

    df['month_period'] = df['day'].apply(lambda x: 'early' if x <= 10 else ('mid' if x <= 20 else 'late'))
    df['job_balance_mean'] = df.groupby('job')['balance'].transform('mean')
    df['balance_vs_job_mean'] = df['balance'] / (df['job_balance_mean'] + 1e-6)

    # Цикличные фичи

    df['_balance_log'] = (np.sign(df['balance']) * np.log1p(np.abs(df['balance']))).astype('float32')

    df['_duration_sin'] = np.sin(2 * np.pi * df['duration'] / 540).astype('float32')
    df['_duration_cos'] = np.cos(2 * np.pi * df['duration'] / 540).astype('float32')
    df['_balance_sin'] = np.sin(2 * np.pi * df['balance'] / 1000).astype('float32')
    df['_balance_cos'] = np.cos(2 * np.pi * df['balance'] / 1000).astype('float32')
    df['_age_sin'] = np.sin(2 * np.pi * df['age'] / 10).astype('float32')
    df['_pdays_sin'] = np.sin(2 * np.pi * df['pdays'] / 7).astype('float32')
    df['month_num'] = df['month'].map(month_map)
    df['_month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12).astype('float32')

    # Преобразование duration
    # df['duration_long_'] = (df['duration'] > 300).astype('category')
    df['_duration_sqrt'] = np.sqrt(df['duration']).astype('float32')

    # CE (Count Encoding) для всех фичей
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    for cat_feature in categorical_features:
        df = count_encoding(df, cat_feature)

    df = count_encoding(df, 'job', 'education')
    df = count_encoding(df, 'month', 'contact')

    # исключение фичей
    df = df.drop(columns=['job_balance_mean', 'default', 'month_num',
                          'campaign', 'previous', 'age', 'housing', 'pdays'])

    return df


def add_feature(df, column1, operation, column2=None):
    """
        Функция для создания нового признака
        :param df - исходный DataFrame
        :param column1 - колонка 1 (используется в первую очередь)
        :param column2 - колонка 1 (используется при совмещении фичей)
        :param operation - операция для создания новой фичи
        :return df с новой колонкой
        """
    # имя новой колонки
    new_col_name = f"{column1}_{operation}" + (f"_{column2}" if column2 else "")

    if operation == "log":
        df[new_col_name] = np.log1p(df[column1].clip(lower=0))

    elif operation == "sqrt":
        df[new_col_name] = np.sqrt(df[column1].clip(lower=0))

    elif operation == "square":
        df[new_col_name] = df[column1] ** 2

    elif operation == "cube":
        df[new_col_name] = df[column1] ** 3

    elif operation == "abs":
        df[new_col_name] = np.abs(df[column1])

    elif operation == "add":
        df[new_col_name] = df[column1] + df[column2]

    elif operation == "sub":
        df[new_col_name] = df[column1] - df[column2]

    elif operation == "mul":
        df[new_col_name] = df[column1] * df[column2]

    elif operation == "ratio":
        df[new_col_name] = df[column1] / (df[column2] + 1e-6)

    elif operation == "poly3":
        df[new_col_name] = df[column1] * df[column2] * df[column1]

    elif operation == "cat_comb":
        df[new_col_name] = (df[column1].astype(str) + "_" + df[column2].astype(str))

    elif operation == "diff":
        df[new_col_name] = df[column1] - df[column2]

    elif operation == "pct_diff":
        df[new_col_name] = (df[column1] - df[column2]) / (df[column2] + 1e-6)

    else:
        raise ValueError(f"Unknown operation: {operation}")

    return df


def show_matrix(matrix, name):
    """
    Функция для визуализации матрицы ошибок
    :param matrix: созданная матрица ошибок
    :param name: дополнение к имени файла
    :return: None
    """
    matrix_norm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    fig, (ax2) = plt.subplots(1, 1, figsize=(15, 6))

    # Проценты
    sns.heatmap(matrix_norm,
                annot=True,
                fmt='.2%',
                cmap='YlGnBu',
                ax=ax2,
                cbar_kws={'label': 'Percentage'})
    ax2.set_title('Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(f'confusion_matrix/confusion_matrix_{name}.png', dpi=300, bbox_inches='tight')


def correlation_plot(df, name, title="Корреляционная матрица", figsize=(15, 12)):
    """
    Функция для построения тепловой карты корреляций.
    Отображает только нижний треугольник для лучшей читаемости.
    """
    # Выбираем только числовые колонки
    corr = df.select_dtypes(include=[np.number]).corr()

    # Создаем маску, чтобы скрыть верхний треугольник (он дублирует нижний)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=figsize, dpi=500)

    # Настройка цветовой схемы (от синего к красному через белый)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr,
                mask=mask,
                cmap=cmap,
                annot=False,  # Поставь True, если хочешь видеть цифры внутри ячеек
                fmt=".2f",
                linewidths=0.5,
                cbar_kws={"shrink": .8})

    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'correlation_matrix_{name}.png', dpi=300, bbox_inches='tight')


def preprocessing(df):
    """
    Функция для предобработки новых данных для инференса
    :param df: датафрейм для предсказания
    :return: подготовленный df
    """
    train = pd.read_csv("train.csv")
    train = create_new_features(train)

    y_train = train["y"]

    categorical_features = train.select_dtypes(
        include=["object"]).columns.tolist()

    # TE mean
    for cat_feature in categorical_features:
        train, df = mean_target_encoding(train, df, cat_feature, y_train)
    train, df = mean_target_encoding(train, df, 'job_education', y_train)
    train, df = mean_target_encoding(train, df, 'month_contact', y_train)

    selected_features = get_selected_features()

    df = df[selected_features]

    return df


def get_selected_features(file='confirmed_features.txt'):
    """
    :param file: Путь к txt файлу с признаками
    :return: list с признаками
    """
    with (open(file, 'r') as f):
        selected_features = f.read()
        selected_features = selected_features.replace("' ", ' ').replace('[', '').replace(']', '').replace("'",'').replace(' ', '')
        selected_features = selected_features.split(',')
        selected_features.remove('month_cat_comb_day_TE')
        selected_features.remove('duration_ratio_age')
        selected_features.remove('balance_log_mul_duration')
        selected_features.remove('_duration_sqrt')
        selected_features.remove('poutcome_cat_comb_loan_TE')
    return selected_features


def create_submission_file(all_preds, test_id, name):
    """
    Функция для создания файла с предсказаниями на тестовой выборке
    :param all_preds: список с предсказаниями
    :param test_id: индексы тестовой выборки
    :param name: имя итерации обучения
    :return: None
    """
    final_submission_probs = np.mean(all_preds, axis=0)

    # Сохраняем результат
    submission = pd.DataFrame({
        "id": test_id,
        "y": final_submission_probs
    })
    submission.to_csv(fr"submissions/submission_{name}_folds.csv", index=False)
