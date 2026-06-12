"""
Статистический анализ разброса сроков службы (распределение Вейбулла).

Реальные модули даже одной партии имеют разброс скорости деградации из-за
производственных допусков. Распределение Вейбулла — стандартный инструмент
для описания времени наступления отказа при старении (раздел 2.7 / 3.4 ВКР).
"""

import numpy as np
from scipy.stats import weibull_min


def weibull_lifetime_analysis(t_eol_det, shape_k=3.0, percentile_levels=(10, 50, 90)):
    """
    Расчёт квантилей распределения Вейбулла по детерминированному сроку службы.

    Детерминированный срок службы принимается за медиану (50-й перцентиль)
    распределения. Для распределения Вейбулла медиана связана с масштабом λ:

        t_50 = λ · (ln 2)^(1/k)   ⟹   λ = t_50 / (ln 2)^(1/k)

    Параметры:
        t_eol_det         — детерминированный срок службы (медиана), лет
        shape_k           — параметр формы k (k=3 для стареющих отказов)
        percentile_levels — требуемые квантили, %

    Возвращает словарь:
        quantiles — {p: t_p} квантили срока службы
        lambda    — масштабный параметр λ
        t_array   — ось времени для графика
        survival  — функция выживания S(t) = 1 − F(t)
    """
    if not np.isfinite(t_eol_det):
        return None

    lambda_param = t_eol_det / (np.log(2) ** (1 / shape_k))

    quantiles = {}
    for p in percentile_levels:
        quantiles[p] = weibull_min.ppf(p / 100, c=shape_k, scale=lambda_param)

    t_max = max(quantiles.values()) * 1.6
    t_array = np.linspace(0, t_max, 300)
    survival = weibull_min.sf(t_array, c=shape_k, scale=lambda_param)

    return {
        "quantiles": quantiles,
        "lambda": lambda_param,
        "shape_k": shape_k,
        "t_array": t_array,
        "survival": survival,
    }


def survival_probability_at(t_eol_det, year, shape_k=3.0):
    """
    Вероятность того, что модуль сохранит > 80 % мощности к заданному году.

        S(year) = exp[ −(year/λ)^k ]
    """
    if not np.isfinite(t_eol_det):
        return 1.0
    lambda_param = t_eol_det / (np.log(2) ** (1 / shape_k))
    return float(weibull_min.sf(year, c=shape_k, scale=lambda_param))
