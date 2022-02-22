import pandas as pd
import numpy as np

from src.metrics import precision_at_k, recall_at_k


def prefilter_items(data, take_n_popular=5000, item_features=None, n_last_week=12, popularity_limit=0.01,\
                    min_price=1, max_price=100):
    # Уберем не интересные для рекоммендаций категории (department)
    # if item_features is not None:
    # department_size = pd.DataFrame(item_features. \
    # groupby('department')['item_id'].nunique(). \
    # sort_values(ascending=False)).reset_index()

    # department_size.columns = ['department', 'n_items']
    # rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
    # items_in_rare_departments = item_features[
    # item_features['department'].isin(rare_departments)].item_id.unique().tolist()

    # data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем товары, у которых quantity == 0
    # data = data[data['quantity'] > 0]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    # data['price'] = data['sales_value'] / data['quantity']
    data = data[data['price'] > min_price]

    # Уберем слишком дорогие товары
    data = data[data['price'] < max_price]

    # Уберем самые НЕ популярные товары (их итак НЕ купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_notpopular = popularity[popularity['share_unique_users'] < popularity_limit].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые НЕ покупались за последние n_last_week
    item_max_week = data.groupby('item_id')['week_no'].max().reset_index()
    item_max_week.columns = ['item_id', 'max_week']
    fresh_items = item_max_week[item_max_week['max_week'] >= item_max_week['max_week'].max() - n_last_week].\
        item_id.unique().tolist()
    data = data[data['item_id'].isin(fresh_items)]

    # Возьмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data


def postfilter_items(recommendations, item_features, n=5):
    """Пост-фильтрация товаров
    Input
    -----
    recommendations: list
        Ранжированный список item_id для рекомендаций
    item_features: pd.DataFrame
        Датафрейм с информацией о товарах
    n: int
        Количество рекомендаций, которые требуется получить на выходе
    """

    # Уникальность
    # recommendations = list(set(recommendations)) - неверно! так теряется порядок
    unique_recommendations = []
    [unique_recommendations.append(item) for item in recommendations if item not in unique_recommendations]

    # Разные категории
    categories_used = []
    final_recommendations = []

    CATEGORY_NAME = 'sub_commodity_desc'
    for item in unique_recommendations:
        category = item_features.loc[item_features['item_id'] == item, CATEGORY_NAME].values[0]

        if category not in categories_used:
            final_recommendations.append(item)

        unique_recommendations.remove(item)
        categories_used.append(category)

    n_rec = len(final_recommendations)
    if n_rec < n:
        final_recommendations.extend(unique_recommendations[:n - n_rec])
    else:
        final_recommendations = final_recommendations[:n]

    assert len(final_recommendations) == n, 'Количество рекомендаций != {}'.format(n)
    return final_recommendations


def calc_precision(df_data, top_k, actual_col):
    for col_name in df_data.columns[2:]:
        yield col_name, df_data.apply(lambda row: precision_at_k(row[col_name], row[actual_col], k=top_k),
                                      axis=1).mean()


def calc_recall(df_data, top_k, actual_col):
    for col_name in df_data.columns[2:]:
        yield col_name, df_data.apply(lambda row: recall_at_k(row[col_name], row[actual_col], k=top_k), axis=1).mean()


def eval_recall(df_result, target_col_name, models, n=5):
    for key, v in models.items():
        df_result[key] = df_result[target_col_name].apply(lambda x: v(x, n))
        print('Model {} has done'.format(key))
    return df_result


def rerank(user_col, user_id, df_predict, proba_col, n=5):
    return df_predict[df_predict[user_col] == user_id].sort_values(proba_col, ascending=False).\
        head(n).item_id.tolist()
