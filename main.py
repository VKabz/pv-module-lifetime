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

# --- Параметры исследуемого модуля (таблица 2.1 / 3.1 ВКР) ---
# Значения взяты НАПРЯМУЮ из таблиц ВКР как итоговые параметры модели:
#   d_ref и Ea приведены в табл. 3.1 как откалиброванные по DH1000 величины.
MODULE = {
    "NOCT": 45,          # номинальная рабочая температура ячейки, °C
    "d_ref": 0.005,      # опорная скорость деградации при 25 °C / 50 % RH, год⁻¹ (табл. 2.1, 3.1)
    "Ea_ev": 0.55,       # энергия активации, эВ (табл. 3.1, откалибрована по DH1000)
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
    print("ШАГ 1. Параметры модели (табл. 2.1 / 3.1 ВКР) и проверка по DH1000")
    line()
    # Итоговые параметры модели берём напрямую из таблиц ВКР
    d_ref = MODULE["d_ref"]
    Ea_joules = physics.ev_to_joules(MODULE["Ea_ev"])
    print(f"  Опорная скорость d_ref   = {d_ref:.4f} год⁻¹ ({d_ref*100:.2f} %/год)  [табл. 2.1/3.1]")
    print(f"  Энергия активации Ea     = {MODULE['Ea_ev']} эВ = {Ea_joules:.0f} Дж/моль  [табл. 3.1]")
    # Проверка согласованности с тестом DH1000 (обоснование, не источник d_ref)
    cal = physics.calibrate_d_ref_from_dh1000(
        dP_DH=MODULE["dP_DH"], Ea_ev=MODULE["Ea_ev"], n=MODULE["n"]
    )
    print(f"  Проверка: коэф. ускорения DH1000 = {cal['AF_DH']:.1f}, "
          f"эквивалентное время = {cal['t_eq_years']:.2f} лет")

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

        # Среднегодовой (упрощённый) подход — для сравнения методов
        t_eol_avg, AF_avg_annual, _ = physics.annual_average_lifetime(
            df, d_ref, Ea_joules, MODULE["n"]
        )

        # Сводные показатели за год
        Tc_avg = df["Tc"].mean()
        RH_avg = df["RH"].mean() * 100
        AF_avg = df["AF"].mean()
        d_year = df["d_hourly"].sum()

        # Распределение Вейбулла
        w = weibull_lifetime_analysis(t_eol, shape_k=3.0)
        surv25 = survival_probability_at(t_eol, 25, shape_k=3.0) * 100

        eol_str = f"{t_eol:.1f}" if pd.notna(t_eol) and t_eol != float("inf") else ">200"
        # Разница почасового и среднегодового методов, %
        delta_method = (t_eol_avg - t_eol) / t_eol_avg * 100 if t_eol > 0 else 0
        print(f"\n  {reg['name']} ({reg['climate']}):")
        print(f"     Tc средн. = {Tc_avg:5.1f} °C | RH средн. = {RH_avg:4.1f} % "
              f"| AF = {AF_avg:5.2f} | d_год = {d_year*100:5.2f} %/год")
        print(f"     Срок службы: почасовой = {eol_str} лет | "
              f"среднегодовой = {t_eol_avg:.1f} лет | разница = {delta_method:.0f} %")
        print(f"     Выживание к 25 году = {surv25:.0f} %")

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
            "Срок службы (почас.), лет": round(t_eol, 1) if t_eol != float("inf") else None,
            "Срок службы (среднегод.), лет": round(t_eol_avg, 1) if t_eol_avg != float("inf") else None,
            "Разница методов, %": round(delta_method, 0),
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
        # Связь качества модуля и d_ref — линейная (табл. 3.5 ВКР):
        #   базовый случай ΔP_DH = 5 % соответствует d_ref = 0,5 %/год,
        #   то есть d_ref = 0,1 · ΔP_DH.
        d_ref_sc = sc["dP_DH"] * 0.1
        df_sc = physics.calculate_degradation(
            df_m, d_ref_sc, Ea_joules, MODULE["n"]
        )
        t_eol_sc, _, _ = physics.find_eol_time(df_sc)
        print(f"  {sc['name']:8s}: ΔP_DH = {sc['dP_DH']*100:4.1f} % "
              f"-> d_ref = {d_ref_sc*100:.3f} %/год -> срок службы = {t_eol_sc:.1f} лет")
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
