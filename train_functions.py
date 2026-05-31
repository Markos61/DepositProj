import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from sklearn import metrics
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from pytabkit import TabM_D_Classifier
from sklearn.impute import SimpleImputer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def show_roc_auc_curve(y_prob, y_test, name, model_type):
    """
    Функция для визуализации кривой ROC-AUC
    :param y_prob: предсказания
    :param y_test: тестовая выборка
    :param name: имя файла
    :param model_type: тип модели
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
    plt.savefig(f"metrics/ROC_AUC_{model_type}_{name}.png", dpi=500, bbox_inches='tight')
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
        df[f'{column1}_{column2}_counts'] = df.groupby(f'{column1}_{column2}')[f'{column1}_{column2}'].transform(
            'count')
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


def show_matrix(matrix, name, model_type):
    """
    Функция для визуализации матрицы ошибок
    :param matrix: созданная матрица ошибок
    :param name: дополнение к имени файла
    :param model_type: тип модели
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
    plt.savefig(f'confusion_matrix/confusion_matrix_{model_type}_{name}.png', dpi=300, bbox_inches='tight')


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
        selected_features = selected_features.replace("' ", ' ').replace('[', '').replace(']', '').replace("'",
                                                                                                           '').replace(
            ' ', '')
        selected_features = selected_features.split(',')
        selected_features.remove('month_cat_comb_day_TE')
        selected_features.remove('duration_ratio_age')
        selected_features.remove('balance_log_mul_duration')
        selected_features.remove('_duration_sqrt')
        selected_features.remove('poutcome_cat_comb_loan_TE')
    return selected_features


def create_submission_file1(all_preds, test_id, name, model_type):
    """
    Функция для создания файла с предсказаниями на тестовой выборке
    :param all_preds: список с предсказаниями
    :param test_id: индексы тестовой выборки
    :param name: имя итерации обучения
    :param model_type: тип модели
    :return: None
    """
    final_submission_probs = np.mean(all_preds, axis=0)

    # Сохраняем результат
    submission = pd.DataFrame({
        "id": test_id,
        "y": final_submission_probs
    })
    submission.to_csv(fr"submissions/submission_{model_type}_{name}_folds.csv", index=False)


def create_submission_file(all_preds, test_id, name, model_type):
    """
    Функция для создания файла с предсказаниями на тестовой выборке
    :param all_preds: список с предсказаниями (от фолдов) или 1D-массив (от мета-модели)
    :param test_id: индексы тестовой выборки
    :param name: имя итерации обучения
    :param model_type: тип модели
    :return: None
    """

    # Проверяем размерность: усредняем только если это список или 2D-массив
    if isinstance(all_preds, list) or (isinstance(all_preds, np.ndarray) and all_preds.ndim > 1):
        final_submission_probs = np.mean(all_preds, axis=0)
    else:
        # Если это уже готовый одномерный массив (например, от мета-модели), оставляем как есть
        final_submission_probs = all_preds

    # Сохраняем результат
    submission = pd.DataFrame({
        "id": test_id,
        "y": final_submission_probs
    })

    submission.to_csv(fr"submissions/submission_{model_type}_{name}_folds.csv", index=False)
    print(f"Файл submission_{model_type}_{name}_folds.csv успешно сохранен!")


def add_original_dataset(df, add=True):
    """
    :param df: начальный набор данных
    :param add: добавить оригинальный набор?
    :return: объединённый датасет
    """
    if add:
        orig_df = pd.read_csv("bank-full.csv", delimiter=';')
        orig_df['y'] = orig_df['y'].replace('yes', 1).replace('no', 0)
        df = pd.concat([df, orig_df], axis=0)
        df.index = list(range(0, len(df)))
        df = df.drop_duplicates()
        return df
    else:
        return df


def get_best_params(categorical_features, model_type):
    """
    Функция для загрузки лучшего набора параметров
    :param categorical_features: категориальные признаки
    :param model_type: тип модели
    :return: словарь с лучшим набором параметров
    """
    if model_type not in ["CatBoost", "TabM", "XGBoost", "LightGBM",
                          "RandomForest", "LogisticRegression", "MLP"]:
        print(f'Нет лучшего набора параметров для модели {model_type}')
        return {}

    if model_type == "CatBoost":
        params = {
            'iterations': 50000,
            'learning_rate': 0.0349,
            'depth': 7,
            'loss_function': "Logloss",
            'eval_metric': "AUC",
            'cat_features': categorical_features,
            'l2_leaf_reg': 0.45,
            'scale_pos_weight': 4,
            'bootstrap_type': 'Bernoulli',
            'random_strength': 0.826,
            'subsample': 0.208,
            'min_data_in_leaf': 12,
            'max_bin': 224,
            'leaf_estimation_iterations': 2,
            'grow_policy': 'Lossguide',
            'max_leaves': 54,
            'verbose': False,
            'task_type': "GPU",
            'devices': "0"
        }
        return params, model_type

    elif model_type == "TabM":
        params = {'device': 'cuda',
                  'val_metric_name': '1-auc_ovr',
                  'random_state': 100,
                  'verbosity': 2,
                  'arch_type': 'tabm-mini',
                  'tabm_k': 32,
                  'num_emb_type': 'pwl',
                  'd_embedding': 12,
                  'batch_size': 256,
                  'lr': 1e-3,
                  'n_epochs': 11,
                  'dropout': 0.1,
                  'd_block': 512,
                  'n_blocks': 3
                  }
        return params, model_type

    elif model_type == "XGBoost":
        params = {
            "n_estimators": 50000,
            'learning_rate': 0.0332,
            'max_depth': 8,
            'subsample': 0.9032,
            'colsample_bytree': 0.8068,
            'min_child_weight': 11,
            'gamma': 1.8518e-07,
            'reg_alpha': 9.5231e-05,
            'reg_lambda': 3.5892e-07,
            'scale_pos_weight': 1.8036,
            'random_state': 42,
            "tree_method": "hist",
            "device": "cuda",
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "enable_categorical": True}

        return params, model_type

    elif model_type == "LightGBM":
        params = {
            "n_estimators": 50000,
            'max_depth': 9,
            'learning_rate': 0.0395,
            'num_leaves': 113,
            'subsample': 0.8966,
            'colsample_bytree': 0.4617,
            'min_child_samples': 42,
            'reg_alpha': 3.8144,
            'reg_lambda': 1.4577e-07,
            'scale_pos_weight': 2.9033,
            "objective": "binary",
            "metric": "auc",
            "device_type": "gpu"}

        return params, model_type

    elif model_type == "RandomForest":
        params = {
            'n_estimators': 1000,  # Большое количество деревьев для стабильности ансамбля
            'max_depth': 12,  # Ограничение глубины, чтобы лес не переобучался
            'min_samples_leaf': 15,  # Защита от слишком мелких листьев (шума)
            'max_features': 'sqrt',  # Количество фичей для каждого сплита
            'class_weight': 'balanced',  # Автоматическая балансировка весов классов
            'n_jobs': -1,
            'random_state': 42,
            'verbose': True
        }
        return params, model_type

    elif model_type == "LogisticRegression":
        params = {
            'C': 0.1,  # Сила регуляризации (чем меньше, тем сильнее штраф за веса)
            'penalty': 'l2',  # L2-регуляризация (Ridge)
            'solver': 'lbfgs',  # Стабильный быстрый алгоритм оптимизации
            'max_iter': 1000,  # Чтобы модель точно успела сойтись
            'class_weight': 'balanced',  # Штраф за ошибки на редком классе
            'random_state': 42
        }
        return params, model_type

    elif model_type == "MLP":
        params = {
            'hidden_layer_sizes': (128, 64),  # Двухслойная архитектура (128 нейронов на первом слое, 64 на втором)
            'activation': 'relu',  # Стандартная и эффективная функция активации
            'solver': 'adam',  # Оптимайзер с адаптивным шагом градиента
            'alpha': 0.001,  # Коэффициент L2-регуляризации для весов нейросети
            'batch_size': 256,  # Размер батча для обучения
            'learning_rate_init': 0.005,  # Начальная скорость обучения
            'max_iter': 300,  # Максимальное количество эпох
            'early_stopping': True,  # Остановить обучение, если лосс на валидации перестал падать
            'validation_fraction': 0.1,  # 10% от train пойдет на внутреннюю валидацию для early_stopping
            'random_state': 42
        }
        return params, model_type


def fit_model(model_type, params, X_train, y_train, X_val, y_val, X_test):
    """
    Функция для загрузки модели с параметрами и обучения
    :param model_type: тип модели
    :param params: лучшие параметры
    :param X_train: обучающий датасет
    :param y_train: метки обучающего датасета
    :param X_val: валидационный датасет
    :param y_val: метки валидационного датасета
    :param X_test: тестовый датасет
    :return: model
    """
    if model_type not in ["CatBoost", "TabM", "XGBoost", "LightGBM",
                          "RandomForest", "LogisticRegression", "MLP"]:
        print(f'Нет модели {model_type}!')
        return None

    if model_type == "CatBoost":
        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=200,
            verbose=500)

        fold_prob = model.predict_proba(X_val)[:, 1]
        test_predict = model.predict_proba(X_test)[:, 1]

        return model, fold_prob, test_predict

    elif model_type == "TabM":
        X_train_w = X_train.copy()
        X_val_w = X_val.copy()
        X_test_w = X_test.copy()

        cat_cols = X_train_w.select_dtypes(include=["object"]).columns.tolist()

        model = TabM_D_Classifier(**params)
        model.fit(X_train_w, y_train, X_val_w, y_val, cat_col_names=cat_cols)

        fold_prob = model.predict_proba(X_val_w)[:, 1]
        test_predict = model.predict_proba(X_test_w)[:, 1]

        return model, fold_prob, test_predict

    elif model_type == "XGBoost":
        X_train_w = X_train.copy()
        X_val_w = X_val.copy()
        X_test_w = X_test.copy()

        # находим object-колонки и переводим в category
        cat_cols = X_train_w.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            X_train_w[col] = X_train_w[col].astype("category")
            X_val_w[col] = X_val_w[col].astype("category")
            X_test_w[col] = X_test_w[col].astype("category")

        xgb_params = params.copy()
        xgb_params['enable_categorical'] = True

        model = XGBClassifier(**xgb_params, early_stopping_rounds=200)
        model.fit(
            X_train_w, y_train,
            eval_set=[(X_val_w, y_val)],
            verbose=500)

        fold_prob = model.predict_proba(X_val_w)[:, 1]
        test_predict = model.predict_proba(X_test_w)[:, 1]

        return model, fold_prob, test_predict

    elif model_type == "LightGBM":
        X_train_w = X_train.copy()
        X_val_w = X_val.copy()
        X_test_w = X_test.copy()

        cat_cols = X_train_w.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            X_train_w[col] = X_train_w[col].astype("category")
            X_val_w[col] = X_val_w[col].astype("category")
            X_test_w[col] = X_test_w[col].astype("category")

        model = LGBMClassifier(**params)
        model.fit(
            X_train_w, y_train,
            eval_set=[(X_val_w, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, first_metric_only=True),
                lgb.log_evaluation(period=500)])

        fold_prob = model.predict_proba(X_val_w)[:, 1]
        test_predict = model.predict_proba(X_test_w)[:, 1]

        return model, fold_prob, test_predict

    elif model_type in ["RandomForest", "LogisticRegression", "MLP"]:

        X_train_num = X_train.select_dtypes(exclude=['object', 'category']).fillna(0)
        X_val_num = X_val.select_dtypes(exclude=['object', 'category']).fillna(0)
        X_test_num = X_test.select_dtypes(exclude=['object', 'category']).fillna(0)

        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            X_train_num = pd.concat([X_train_num, pd.get_dummies(X_train[cat_cols], drop_first=True)], axis=1)
            X_val_num = pd.concat([X_val_num, pd.get_dummies(X_val[cat_cols], drop_first=True)], axis=1)
            X_test_num = pd.concat([X_test_num, pd.get_dummies(X_test[cat_cols], drop_first=True)], axis=1)

            X_val_num = X_val_num.reindex(columns=X_train_num.columns, fill_value=0)
            X_test_num = X_test_num.reindex(columns=X_train_num.columns, fill_value=0)

        if model_type in ["LogisticRegression", "MLP"]:
            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train_num)
            X_val_processed = scaler.transform(X_val_num)
            X_test_processed = scaler.transform(X_test_num)
        else:
            X_train_processed, X_val_processed, X_test_processed = X_train_num, X_val_num, X_test_num

        if model_type == "RandomForest":

            model = RandomForestClassifier(**params)
            model.fit(X_train_processed, y_train)

        elif model_type == "LogisticRegression":
            model = LogisticRegression(**params)
            model.fit(X_train_processed, y_train)

        elif model_type == "MLP":

            model = MLPClassifier(**params)
            model.fit(X_train_processed, y_train)

        # 5. Предсказание
        fold_prob = model.predict_proba(X_val_processed)[:, 1]
        test_predict = model.predict_proba(X_test_processed)[:, 1]

        return model, fold_prob, test_predict


def get_optimization_params(trial, model_type, categorical_features):
    """
    Генерирует пространство поиска гиперпараметров в зависимости от типа модели.
    """
    if model_type == "CatBoost":
        params = {
            "iterations": 2000,  # Для подбора ставим меньше, чем для финала (например, 2000 вместо 50000)
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),

            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "max_bin": trial.suggest_int("max_bin", 32, 254),
            "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 15),

            "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),

            "task_type": "GPU",
            "devices": "0",
            "eval_metric": "AUC",
            "verbose": False,
            "od_type": "Iter",
            "od_wait": 100
        }

        # Условные параметры CatBoost
        if params["grow_policy"] == "Lossguide":
            params["max_leaves"] = trial.suggest_int("max_leaves", 16, 64)

        if params["bootstrap_type"] == "Bernoulli":
            params["subsample"] = trial.suggest_float("subsample", 0.1, 1.0)
        else:  # Bayesian
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)

        params['cat_features'] = categorical_features

        return params

    elif model_type == "XGBoost":
        params = {
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),

            "tree_method": "hist",
            "device": "cuda",
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "enable_categorical": True
        }
        return params

    elif model_type == "LightGBM":
        max_depth = trial.suggest_int("max_depth", 3, 9)
        max_leaves_limit = int(2 ** max_depth - 1)
        params = {
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "max_depth": max_depth,
            "num_leaves": trial.suggest_int("num_leaves", 7, min(256, max_leaves_limit)),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),

            "objective": "binary",
            "metric": "auc",
            "device_type": "gpu",
            "verbose": -1
        }
        return params

    elif model_type == "TabM":
        params = {
            # Фиксированные инфраструктурные параметры
            'device': 'cuda',
            'val_metric_name': '1-auc_ovr',
            'random_state': 100,
            'verbosity': 1,  # Отрезаем лишний спам в консоль во время подбора
            'arch_type': 'tabm-mini',
            'num_emb_type': 'pwl',
            # Параметры, которые оптимизирует Optuna
            # 1. Оптимизация обучения
            'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [512, 1024, 2048]),
            'n_epochs': 4,  # количество эпох
            # 2. Архитектура сети
            'd_block': trial.suggest_categorical('d_block', [256, 512, 1024]),
            'n_blocks': trial.suggest_int('n_blocks', 2, 5),
            'd_embedding': trial.suggest_int('d_embedding', 8, 32),
            # 3. Регуляризация и ансамблирование
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'tabm_k': trial.suggest_categorical('tabm_k', [16, 32, 64])  # Размер внутреннего ансамбля TabM
        }

        return params

    # === НОВЫЕ МОДЕЛИ ===

    elif model_type == "RandomForest":
        params = {
            # Деревьев можно зафиксировать на 500-1000, тюнить их количество в Optuna не очень эффективно
            "n_estimators": 500,
            "max_depth": trial.suggest_int("max_depth", 5, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            # Какую долю фичей брать для сплита. log2 и sqrt работают лучше всего
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42
        }
        return params

    elif model_type == "LogisticRegression":
        params = {
            # C - обратный коэффициент регуляризации. Чем меньше, тем сильнее штраф
            "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
            "penalty": "l2",  # L2 работает с lbfgs, для L1 нужен другой solver (saga), который сильно медленнее
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42
        }
        return params

    elif model_type == "MLP":
        params = {
            # Тестируем разную глубину и ширину слоев
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes",
                [(64,), (128, 64), (256, 128, 64)]
            ),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),  # L2 штраф
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),

            "solver": "adam",
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "random_state": 42
        }
        return params


def optuna_fit_model(model_type, params, X_train, y_train, X_test):
    """
    Функция для загрузки модели с параметрами и обучения
    :param model_type: тип модели
    :param params: лучшие параметры
    :param X_train: обучающий датасет
    :param y_train: метки обучающего датасета
    :param X_test: тестовый датасет
    :return: model
    """
    if model_type not in ["CatBoost", "TabM", "XGBoost", "LightGBM",
                          "RandomForest", "LogisticRegression", "MLP"]:
        print(f'Нет модели {model_type}!')
        return None

    if model_type == "CatBoost":
        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            use_best_model=True,
            early_stopping_rounds=200,
            verbose=500)

        test_predict = model.predict_proba(X_test)[:, 1]

        return model, test_predict

    elif model_type == "TabM":
        X_train_w = X_train.copy()
        X_test_w = X_test.copy()

        cat_cols = X_train_w.select_dtypes(include=["object"]).columns.tolist()

        model = TabM_D_Classifier(**params)
        model.fit(X_train_w, y_train, cat_col_names=cat_cols)

        test_predict = model.predict_proba(X_test_w)[:, 1]

        return model, test_predict

    elif model_type == "XGBoost":
        X_train_w = X_train.copy()
        X_test_w = X_test.copy()

        cat_cols = X_train_w.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            X_train_w[col] = X_train_w[col].astype("category")
            X_test_w[col] = X_test_w[col].astype("category")

        xgb_params = params.copy()
        xgb_params['enable_categorical'] = True

        model = XGBClassifier(**xgb_params)  # early_stopping_rounds=200
        model.fit(
            X_train_w, y_train,
            verbose=500)

        test_predict = model.predict_proba(X_test_w)[:, 1]

        return model, test_predict

    elif model_type == "LightGBM":
        X_train_w = X_train.copy()
        X_test_w = X_test.copy()

        cat_cols = X_train_w.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            X_train_w[col] = X_train_w[col].astype("category")
            X_test_w[col] = X_test_w[col].astype("category")

        model = LGBMClassifier(**params)
        model.fit(
            X_train_w, y_train,
            callbacks=[
                # lgb.early_stopping(stopping_rounds=200, first_metric_only=True),
                lgb.log_evaluation(period=500)])

        test_predict = model.predict_proba(X_test_w)[:, 1]

        return model, test_predict

    elif model_type in ["RandomForest", "LogisticRegression", "MLP"]:

        X_train_num = X_train.select_dtypes(exclude=['object', 'category']).fillna(0)
        X_test_num = X_test.select_dtypes(exclude=['object', 'category']).fillna(0)

        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            X_train_num = pd.concat([X_train_num, pd.get_dummies(X_train[cat_cols], drop_first=True)], axis=1)
            X_test_num = pd.concat([X_test_num, pd.get_dummies(X_test[cat_cols], drop_first=True)], axis=1)

            X_test_num = X_test_num.reindex(columns=X_train_num.columns, fill_value=0)

        if model_type in ["LogisticRegression", "MLP"]:
            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train_num)
            X_test_processed = scaler.transform(X_test_num)
        else:
            X_train_processed, X_test_processed = X_train_num, X_test_num

        if model_type == "RandomForest":

            model = RandomForestClassifier(**params)
            model.fit(X_train_processed, y_train)

        elif model_type == "LogisticRegression":
            model = LogisticRegression(**params)
            model.fit(X_train_processed, y_train)

        elif model_type == "MLP":

            model = MLPClassifier(**params)
            model.fit(X_train_processed, y_train)

        test_predict = model.predict_proba(X_test_processed)[:, 1]

        return model, test_predict
