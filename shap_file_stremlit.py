import catboost
import shap
import pandas as pd


def save_force_plot(model, x, cat_features, params, filename="shap_force.html"):
    """
    Сохраняет force_plot для одного объекта в HTML.
    :param filename: - название для сохранения файла
    :param params: - все названия фичей (list)
    :param x: - входные данные модели (вектор-строка)
    :param cat_features: - категориальные фичи (list)
    :param model - предварительно обученная модель;
    :return название сохранённого файла

    """
    # Преобразуем features_array в DataFrame, если это np.array
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x, columns=params)

    pool_test = catboost.Pool(x, cat_features=cat_features,
                              feature_names=params)

    shap_values = model.get_feature_importance(
        pool_test,
        type="ShapValues")
    # Индекс объекта (только 1 строка)
    i = 0
    # SHAP для конкретного объекта
    base_value = shap_values[i, -1]
    shap_values_object = shap_values[i, :-1]

    force_plot = shap.force_plot(
        base_value,
        shap_values_object,
        x.iloc[i],
        feature_names=params,
        show=False
    )

    shap.save_html(filename, force_plot)
    return filename
