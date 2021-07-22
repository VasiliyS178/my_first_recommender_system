import numpy as np


def hit_rate_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    flags = np.isin(bought_list, recommended_list)
    hit_rate = (flags.sum() > 0) * 1
    return hit_rate


def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    bought_list = bought_list
    recommended_list = recommended_list[:k]
    flags = np.isin(bought_list, recommended_list)
    precision = flags.sum() / len(recommended_list)
    return precision


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    prices_list = np.array(prices_recommended[:k])
    bought_list = bought_list
    flags = np.isin(recommended_list, bought_list)
    precision = np.dot(flags, prices_list) / prices_list.sum()
    return precision


def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)
    return recall


def money_recall_at_k(recommended_list, bought_list, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    prices_bought_list = np.array(prices_bought)
    flags = np.isin(bought_list, recommended_list)
    recall = np.dot(flags, prices_bought_list) / prices_bought_list.sum()
    return recall


def ap_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(recommended_list, bought_list)

    if sum(flags) == 0:
        return 0
    sum_ = 0

    for i in range(1, k + 1):
        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i)
            sum_ += p_k
    result = sum_ / k
    return result
