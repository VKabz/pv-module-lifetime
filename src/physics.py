"""
Физическая модель деградации фотоэлектрического модуля.

Реализует методологию из главы 2 ВКР:
  - расчёт инсоляции на наклонную плоскость (упрощённая модель);
  - расчёт температуры ячейки по модели Номера–Пирса (NOCT);
  - расчёт почасовой деградации по моделям Аррениуса и Пэка;
  - определение срока службы до порога 80 % мощности;
  - калибровка параметров модели по тесту DH1000.
"""

import numpy as np

# --- Физические константы ---
R = 8.314           # универсальная газовая постоянная, Дж/(моль·К)
EV_TO_J_PER_MOL = 96485.0   # 1 эВ = 96485 Дж/моль (постоянная Фарадея)
HOURS_PER_YEAR = 8760.0


def ev_to_joules(ea_ev):
    """Перевод энергии активации из эВ в Дж/моль."""
    return ea_ev * EV_TO_J_PER_MOL


def calculate_plane_of_array_irradiance(df, lat, beta):
    """
    Упрощённый перевод горизонтальной инсоляции G_h в инсоляцию на наклонную
    плоскость G_inc.

    В стандартной почасовой выдаче NASA POWER нет компонент DNI и DHI,
    необходимых для строгой модели Hay–Davies (pvlib). Поэтому, как и в
    основном скрипте ВКР, применяется геометрическое приближение:

        G_inc ≈ G_h · cos(β − φ)

    где β — угол наклона панели, φ — широта. Отрицательные значения (ночь)
    обнуляются.

    Параметры:
        df   — DataFrame с колонкой G_h
        lat  — широта, град
        beta — угол наклона панели, град
    """
    df = df.copy()
    df["G_inc"] = df["G_h"] * np.cos(np.radians(beta - lat))
    df["G_inc"] = df["G_inc"].clip(lower=0)
    return df


def calculate_cell_temperature(df, NOCT=45, Ta_NOCT=20, G_ref=800, wind_factor=0.06):
    """
    Расчёт температуры ячейки по модели Номера–Пирса (уравнение 1.6 ВКР).

        Tc = Ta + (G_inc / G_ref) · (NOCT − Ta_NOCT) / (1 + f_w · v_w)

    Параметры:
        df          — DataFrame с колонками Ta, G_inc, wind_speed
        NOCT        — номинальная рабочая температура ячейки, °C
        Ta_NOCT     — температура воздуха при измерении NOCT, °C
        G_ref       — опорная инсоляция, Вт/м²
        wind_factor — коэффициент влияния ветра
    """
    df = df.copy()
    G_inc = df["G_inc"].clip(lower=0)
    df["Tc"] = df["Ta"] + (G_inc / G_ref) * (NOCT - Ta_NOCT) / (
        1 + wind_factor * df["wind_speed"]
    )
    # Ночью (G_inc ≈ 0) температура ячейки равна температуре воздуха
    df.loc[df["G_inc"] < 1, "Tc"] = df.loc[df["G_inc"] < 1, "Ta"]
    return df


def calculate_degradation(df, d_ref, Ea_joules, n, T_ref=298.15, RH_ref=0.5):
    """
    Расчёт почасовой доли деградации по моделям Аррениуса и Пэка.

    Коэффициент ускорения:
        AF = exp[(Ea/R)·(1/T_ref − 1/Tc)] · (RH/RH_ref)^n

    Параметры:
        df        — DataFrame с колонками Tc (°C) и RH (доли от 1)
        d_ref     — годовая скорость деградации при опорных условиях
        Ea_joules — энергия активации, Дж/моль
        n         — показатель степени влажности (модель Пэка)
        T_ref     — опорная температура, К (по умолчанию 298.15 = 25 °C)
        RH_ref    — опорная влажность, доли (по умолчанию 0.5 = 50 %)

    Добавляет колонки:
        AF       — коэффициент ускорения
        d_hourly — часовая доля деградации
    """
    df = df.copy()
    Tc_K = df["Tc"] + 273.15

    # Температурный (Аррениус) и влажностный (Пэк) факторы
    arrhenius_factor = np.exp((Ea_joules / R) * (1 / T_ref - 1 / Tc_K))
    rh_factor = (df["RH"] / RH_ref) ** n

    df["AF"] = arrhenius_factor * rh_factor
    # Годовую скорость d_ref·AF переводим в часовую делением на 8760
    df["d_hourly"] = d_ref * df["AF"] / HOURS_PER_YEAR
    return df


def find_eol_time(df, degradation_target=0.20):
    """
    Нахождение срока службы — времени достижения порога деградации.

    Порог 80 % мощности соответствует накопленной деградации 20 % (0.20).
    Предполагается, что климатический год повторяется (типовой метеогод TMY),
    поэтому накопление продлевается во времени тайлингом одного года.

    Параметры:
        df                — DataFrame с колонкой d_hourly (за 1 год)
        degradation_target— целевая деградация (0.20 = 20 %)

    Возвращает:
        t_eol_years — срок службы в годах
        cumulative  — массив накопленной деградации (продлённый до EOL)
        years_axis  — соответствующая ось времени в годах
    """
    d_year = df["d_hourly"].sum()           # деградация за один год
    if d_year <= 0:
        return np.inf, np.array([0.0]), np.array([0.0])

    # Сколько лет нужно, чтобы накопить degradation_target (+ запас 1 год)
    n_years = int(np.ceil(degradation_target / d_year)) + 1
    n_years = min(n_years, 200)             # страховка от бесконечного цикла

    hourly = df["d_hourly"].values
    full = np.tile(hourly, n_years)         # повторяем год n_years раз
    cumulative = np.cumsum(full)
    years_axis = np.arange(len(full)) / HOURS_PER_YEAR

    # Первый час, где деградация превысила порог
    idx = np.argmax(cumulative >= degradation_target)
    if cumulative[idx] < degradation_target:
        t_eol_years = np.inf                # порог не достигнут
    else:
        t_eol_years = idx / HOURS_PER_YEAR

    return t_eol_years, cumulative, years_axis


def calibrate_d_ref_from_dh1000(
    dP_DH=0.05, T_DH=85, RH_DH=0.85, t_DH_hours=1000,
    Ea_ev=0.55, n=2.5, T_ref=298.15, RH_ref=0.5,
):
    """
    Калибровка опорной скорости деградации d_ref по результатам теста DH1000.

    Тест DH1000: 1000 часов при 85 °C и 85 % RH, измеряется падение мощности
    ΔP_DH. Зная ΔP_DH, находим эквивалентную скорость деградации d_ref при
    опорных условиях (25 °C, 50 % RH).

    Алгоритм:
        1. Коэффициент ускорения теста относительно опорных условий:
               AF_DH = exp[(Ea/R)(1/T_ref − 1/T_DH)] · (RH_DH/RH_ref)^n
        2. Эквивалентное время эксплуатации в опорных условиях:
               t_eq = (t_DH / 8760) · AF_DH   [лет]
        3. Опорная скорость деградации:
               d_ref = ΔP_DH / t_eq           [год⁻¹]

    Возвращает словарь с d_ref, AF_DH, t_eq_years.
    """
    Ea_joules = ev_to_joules(Ea_ev)
    T_DH_K = T_DH + 273.15

    arrhenius_factor_DH = np.exp((Ea_joules / R) * (1 / T_ref - 1 / T_DH_K))
    rh_factor_DH = (RH_DH / RH_ref) ** n
    AF_DH = arrhenius_factor_DH * rh_factor_DH

    t_eq_years = (t_DH_hours / HOURS_PER_YEAR) * AF_DH
    d_ref = dP_DH / t_eq_years

    return {
        "d_ref": d_ref,           # год⁻¹
        "AF_DH": AF_DH,           # коэффициент ускорения DH1000
        "t_eq_years": t_eq_years, # эквивалентное время, лет
        "Ea_ev": Ea_ev,
        "Ea_joules": Ea_joules,
    }
