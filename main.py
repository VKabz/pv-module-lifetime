"""
==============================================================================
  Прогнозирование срока службы фотоэлектрического модуля по климатическим
  данным региона (расчётная часть ВКР, глава 3).
==============================================================================

Скрипт выполняет полный цикл расчёта:
  1. Калибровка опорной скорости деградации d_ref по тесту DH1000.
  2. Загрузка почасовых климатических данных NASA POWER для нескольких регионов.
  3. Расчёт температуры ячейки, почасовой деградации и срока службы (EOL).
  4. Статистический анализ разброса (распределение Вейбулла).
  5. Сценарный анализ влияния качества модуля (DH1000).
  6. Построение всех графиков и сводных таблиц.

Запуск:
    python main.py
Результаты (графики и таблицы) сохраняются в папку output/.
"""

import os
import pandas as pd

from src.nasa_power import fetch_nasa_power_data
from src import physics
from src.weibull import weibull_lifetime_analysis, survival_probability_at
from src import plots

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# --- Параметры исследуемого модуля (таблица 2.1 ВКР) ---
MODULE = {
    "NOCT": 45,          # номинальная рабочая температура ячейки, °C
    "Ea_ev": 0.55,       # энергия активации, эВ (калибруется по DH1000)
    "n": 2.5,            # показатель степени влажности (модель Пэка)
    "beta": 30,          # угол наклона панели, град
    "dP_DH": 0.05,       # деградация в тесте DH1000 (5 %)
}

# --- Климатические зоны (таблица 2.2 / 3.3 ВКР) ---
REGIONS = [
    {"name": "Москва",    "lat": 55.75, "lon": 37.62, "climate": "Умеренный"},
    {"name": "Сочи",      "lat": 43.58, "lon": 39.73, "climate": "Субтропический"},
    {"name": "Сингапур",  "lat": 1.35,  "lon": 103.82, "climate": "Тропический"},
    {"name": "Дубай",     "lat": 25.20, "lon": 55.27, "climate": "Пустынный"},
]

YEAR = "2023"  # год климатических данных


def line(char="="):
    print(char * 78)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ======================================================================
    # ШАГ 1. Калибровка d_ref по тесту DH1000
    # ======================================================================
    line()
    print("ШАГ 1. Калибровка опорной скорости деградации по тесту DH1000")
    line()
    cal = physics.calibrate_d_ref_from_dh1000(
        dP_DH=MODULE["dP_DH"], Ea_ev=MODULE["Ea_ev"], n=MODULE["n"]
    )
    d_ref = cal["d_ref"]
    Ea_joules = cal["Ea_joules"]
    print(f"  Энергия активации Ea     = {MODULE['Ea_ev']} эВ = {Ea_joules:.0f} Дж/моль")
    print(f"  Коэф. ускорения DH1000   = {cal['AF_DH']:.1f}")
    print(f"  Эквивалентное время      = {cal['t_eq_years']:.2f} лет в опорных условиях")
    print(f"  --> Опорная d_ref        = {d_ref:.5f} год⁻¹ ({d_ref*100:.3f} %/год)")

    start = f"{YEAR}0101"
    end = f"{YEAR}1231"

    # ======================================================================
    # ШАГ 2-4. Расчёт по каждому региону
    # ======================================================================
    line()
    print("ШАГ 2-4. Расчёт срока службы по климатическим зонам")
    line()

    results = []           # для графика деградации
    weibull_results = []   # для графика Вейбулла
    summary_rows = []      # для сводной таблицы

    for reg in REGIONS:
        df = fetch_nasa_power_data(reg["lat"], reg["lon"], start, end, reg["name"])

        # Инсоляция на наклонную плоскость и температура ячейки
        df = physics.calculate_plane_of_array_irradiance(df, reg["lat"], MODULE["beta"])
        df = physics.calculate_cell_temperature(df, NOCT=MODULE["NOCT"])

        # Нормализация влажности: % -> доли
        df["RH"] = df["RH"] / 100.0

        # Почасовая деградация и срок службы
        df = physics.calculate_degradation(df, d_ref, Ea_joules, MODULE["n"])
        t_eol, cumulative, years_axis = physics.find_eol_time(df)

        # Сводные показатели за год
        Tc_avg = df["Tc"].mean()
        RH_avg = df["RH"].mean() * 100
        AF_avg = df["AF"].mean()
        d_year = df["d_hourly"].sum()

        # Распределение Вейбулла
        w = weibull_lifetime_analysis(t_eol, shape_k=3.0)
        surv25 = survival_probability_at(t_eol, 25, shape_k=3.0) * 100

        eol_str = f"{t_eol:.1f}" if pd.notna(t_eol) and t_eol != float("inf") else ">200"
        print(f"\n  {reg['name']} ({reg['climate']}):")
        print(f"     Tc средн. = {Tc_avg:5.1f} °C | RH средн. = {RH_avg:4.1f} % "
              f"| AF = {AF_avg:5.2f} | d_год = {d_year*100:5.2f} %/год")
        print(f"     Срок службы (до 80 %) = {eol_str} лет | "
              f"выживание к 25 году = {surv25:.0f} %")

        results.append({
            "region": f"{reg['name']} ({reg['climate']})",
            "years_axis": years_axis,
            "cumulative": cumulative,
            "t_eol": t_eol,
        })
        weibull_results.append({
            "region": reg["name"],
            "weibull": w,
        })
        summary_rows.append({
            "Регион": reg["name"],
            "Климат": reg["climate"],
            "Tc средн., °C": round(Tc_avg, 1),
            "RH средн., %": round(RH_avg, 1),
            "AF (почасовой)": round(AF_avg, 2),
            "d_год, %/год": round(d_year * 100, 2),
            "Срок службы, лет": round(t_eol, 1) if t_eol != float("inf") else None,
            "Выживание к 25 г., %": round(surv25, 1),
        })

    # ======================================================================
    # ШАГ 5. Сценарный анализ DH1000 (умеренный климат — Москва)
    # ======================================================================
    line()
    print("\nШАГ 5. Сценарный анализ качества модулей (DH1000), климат: Москва")
    line()

    moscow = REGIONS[0]
    df_m = fetch_nasa_power_data(moscow["lat"], moscow["lon"], start, end, moscow["name"])
    df_m = physics.calculate_plane_of_array_irradiance(df_m, moscow["lat"], MODULE["beta"])
    df_m = physics.calculate_cell_temperature(df_m, NOCT=MODULE["NOCT"])
    df_m["RH"] = df_m["RH"] / 100.0

    SCENARIOS = [
        {"name": "Плохой",  "dP_DH": 0.070},
        {"name": "Базовый", "dP_DH": 0.050},
        {"name": "Хороший", "dP_DH": 0.030},
        {"name": "Премиум", "dP_DH": 0.015},
    ]
    scenario_results = []
    for sc in SCENARIOS:
        cal_sc = physics.calibrate_d_ref_from_dh1000(
            dP_DH=sc["dP_DH"], Ea_ev=MODULE["Ea_ev"], n=MODULE["n"]
        )
        df_sc = physics.calculate_degradation(
            df_m, cal_sc["d_ref"], Ea_joules, MODULE["n"]
        )
        t_eol_sc, _, _ = physics.find_eol_time(df_sc)
        print(f"  {sc['name']:8s}: ΔP_DH = {sc['dP_DH']*100:4.1f} % "
              f"-> d_ref = {cal_sc['d_ref']*100:.3f} %/год -> срок службы = {t_eol_sc:.1f} лет")
        scenario_results.append({
            "name": sc["name"],
            "dP_DH_percent": sc["dP_DH"] * 100,
            "t_eol": t_eol_sc,
        })

    # ======================================================================
    # ШАГ 6. Построение графиков и сохранение таблиц
    # ======================================================================
    line()
    print("\nШАГ 6. Построение графиков и сохранение таблиц")
    line()

    plots.plot_degradation_curves(results)
    plots.plot_weibull_survival(weibull_results)
    plots.plot_scenario_dh1000(scenario_results)
    plots.plot_monthly_temperature(df_m, "Москва")

    # Сводные таблицы в CSV (легко открыть в Excel / вставить в работу)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "table_3_3_regions.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  сохранена таблица: output/table_3_3_regions.csv")

    scenario_df = pd.DataFrame([{
        "Сценарий": s["name"],
        "ΔP_DH, %": s["dP_DH_percent"],
        "Срок службы, лет": round(s["t_eol"], 1),
    } for s in scenario_results])
    scenario_path = os.path.join(OUTPUT_DIR, "table_3_5_scenarios.csv")
    scenario_df.to_csv(scenario_path, index=False, encoding="utf-8-sig")
    print(f"  сохранена таблица: output/table_3_5_scenarios.csv")

    line()
    print("\nГОТОВО. Все результаты — в папке output/")
    print("\nСводная таблица по регионам:")
    print(summary_df.to_string(index=False))
    line()


if __name__ == "__main__":
    main()
