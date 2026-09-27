import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (Корень из среднеквадратичной ошибки)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def precision_at_k(recommended_ids: list, actual_ids: list, k: int = 10) -> float:
    """Precision@K: Доля релевантных фильмов среди первых K рекомендованных."""
    recommended_at_k = recommended_ids[:k]
    if not recommended_at_k:
        return 0.0
    relevant_retrieved = set(recommended_at_k).intersection(set(actual_ids))
    return len(relevant_retrieved) / k


def recall_at_k(recommended_ids: list, actual_ids: list, k: int = 10) -> float:
    """Recall@K: Доля найденных релевантных фильмов от общего числа любимых фильмов пользователя."""
    if not actual_ids:
        return 0.0
    recommended_at_k = recommended_ids[:k]
    relevant_retrieved = set(recommended_at_k).intersection(set(actual_ids))
    return len(relevant_retrieved) / len(actual_ids)