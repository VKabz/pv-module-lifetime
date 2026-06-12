"""
==============================================================================
  ПРОВЕРКА КОРРЕКТНОСТИ РАСЧЁТОВ
==============================================================================

Скрипт независимо пересчитывает ключевые величины модели «вручную»
(по элементарным формулам) и сверяет их с результатами основного кода.
Каждая проверка завершается отметкой [OK] или [ОШИБКА].

Дополнительно проверяются физические свойства модели:
  - выпуклость Аррениуса (неравенство Йенсена): почасовой AF > AF по средней T;
  - монотонность: чем жарче/влажнее климат, тем короче срок службы;
  - совпадение примеров из текста ВКР с расчётом кода.

Запуск:  python verify.py
"""

import math
import numpy as np

from src import physics
from src.weibull import weibull_lifetime_analysis, survival_probability_at
from src.nasa_power import fetch_nasa_power_data

passed = 0
failed = 0


def check(name, got, expected, tol=1e-2, rel=False):
    """Сравнение значения с эталоном с заданной точностью."""
    global passed, failed
    if rel:
        ok = abs(got - expected) <= tol * abs(expected)
    else:
        ok = abs(got - expected) <= tol
    status = "[OK]    " if ok else "[ОШИБКА]"
    print(f"  {status} {name}")
    print(f"            расчёт = {got:.4f} | эталон = {expected:.4f}")
    if ok:
        passed += 1
    else:
        failed += 1
    return ok


def check_true(name, condition, detail=""):
    global passed, failed
    status = "[OK]    " if condition else "[ОШИБКА]"
    print(f"  {status} {name}")
    if detail:
        print(f"            {detail}")
    if condition:
        passed += 1
    else:
        failed += 1
    return condition


print("=" * 78)
print("ПРОВЕРКА 1. Перевод энергии активации эВ -> Дж/моль")
print("=" * 78)
# 1 эВ = 96485 Дж/моль (постоянная Фарадея). 0.55 эВ должно дать ~53067.
ev = physics.ev_to_joules(0.55)
check("0.55 эВ в Дж/моль (текст ВКР: ≈53000)", ev, 53067, tol=5)

print()
print("=" * 78)
print("ПРОВЕРКА 2. Температура ячейки (модель Номера–Пирса)")
print("=" * 78)
# Пример из раздела 2.4 ВКР: Ta=25, Ginc=850, vw=2 -> Tc=48.7 °C
import pandas as pd
df_demo = pd.DataFrame({
    "Ta": [25.0],
    "G_inc": [850.0],
    "wind_speed": [2.0],
})
df_demo = physics.calculate_cell_temperature(
    df_demo, NOCT=45, Ta_NOCT=20, G_ref=800, wind_factor=0.06
)
tc = df_demo["Tc"].iloc[0]
# Ручной расчёт: 25 + (850/800)*(45-20)/(1+0.06*2) = 25 + 1.0625*25/1.12 = 48.72
manual_tc = 25 + (850 / 800) * (45 - 20) / (1 + 0.06 * 2)
check("Tc для летнего полудня (текст ВКР: 48.7 °C)", tc, 48.72, tol=0.1)
check("совпадение с ручной формулой", tc, manual_tc, tol=1e-9)

print()
print("=" * 78)
print("ПРОВЕРКА 3. Калибровка по тесту DH1000")
print("=" * 78)
cal = physics.calibrate_d_ref_from_dh1000(dP_DH=0.05, Ea_ev=0.55, n=2.5)
# Ручной расчёт AF_DH:
Ea = 0.55 * 96485
R = 8.314
arr = math.exp((Ea / R) * (1 / 298.15 - 1 / 358.15))
rh = (0.85 / 0.5) ** 2.5
AF_DH_manual = arr * rh
t_eq_manual = (1000 / 8760) * AF_DH_manual
d_ref_manual = 0.05 / t_eq_manual
check("коэффициент ускорения AF_DH", cal["AF_DH"], AF_DH_manual, tol=1e-6)
check("эквивалентное время, лет", cal["t_eq_years"], t_eq_manual, tol=1e-6)
check("опорная скорость d_ref, год⁻¹", cal["d_ref"], d_ref_manual, tol=1e-9)
check_true(
    "AF_DH в разумных пределах (теста ускоряет старение в ~100+ раз)",
    100 < cal["AF_DH"] < 200,
    f"AF_DH = {cal['AF_DH']:.1f}",
)

print()
print("=" * 78)
print("ПРОВЕРКА 4. Коэффициент ускорения Аррениуса при нагреве")
print("=" * 78)
# При Tc=45 °C относительно 25 °C, только температура (RH=RHref):
df_af = pd.DataFrame({"Tc": [45.0], "RH": [0.5]})
df_af = physics.calculate_degradation(df_af, d_ref=1.0, Ea_joules=Ea, n=2.5)
af45 = df_af["AF"].iloc[0]
af45_manual = math.exp((Ea / R) * (1 / 298.15 - 1 / 318.15)) * (0.5 / 0.5) ** 2.5
check("AF при Tc=45 °C (только температура)", af45, af45_manual, tol=1e-9)
check_true(
    "нагрев на 20 °C ускоряет деградацию в 3-5 раз (диапазон для PV)",
    3.0 < af45 < 5.0,
    f"AF(45°C) = {af45:.2f}",
)

print()
print("=" * 78)
print("ПРОВЕРКА 5. Распределение Вейбулла")
print("=" * 78)
# Для t_eol=44.6, k=3: lambda = 44.6/(ln2)^(1/3), S(25)=exp(-(25/lambda)^3)
w = weibull_lifetime_analysis(44.6, shape_k=3.0)
lam_manual = 44.6 / (math.log(2) ** (1 / 3))
s25_manual = math.exp(-((25 / lam_manual) ** 3))
check("масштаб λ", w["lambda"], lam_manual, tol=1e-6)
check("выживание к 25 году, доли", survival_probability_at(44.6, 25), s25_manual, tol=1e-6)
# Медиана распределения должна равняться исходному сроку службы
check("медиана = детерминированный срок службы", w["quantiles"][50], 44.6, tol=0.1)
check_true(
    "квантили упорядочены: t10 < t50 < t90",
    w["quantiles"][10] < w["quantiles"][50] < w["quantiles"][90],
    f"t10={w['quantiles'][10]:.1f}, t50={w['quantiles'][50]:.1f}, t90={w['quantiles'][90]:.1f}",
)

print()
print("=" * 78)
print("ПРОВЕРКА 6. Физическая корректность на реальных данных")
print("=" * 78)
# Считаем для двух контрастных регионов и проверяем физику
regions = {
    "Москва (холодный)": {"lat": 55.75, "lon": 37.62},
    "Сингапур (жаркий влажный)": {"lat": 1.35, "lon": 103.82},
}
eol = {}
af_hourly = {}
af_avgtemp = {}
for name, c in regions.items():
    df = fetch_nasa_power_data(c["lat"], c["lon"], "20230101", "20231231",
                               name.split()[0])
    df = physics.calculate_plane_of_array_irradiance(df, c["lat"], 30)
    df = physics.calculate_cell_temperature(df, NOCT=45)
    df["RH"] = df["RH"] / 100.0
    df = physics.calculate_degradation(df, cal["d_ref"], Ea, 2.5)
    t, _, _ = physics.find_eol_time(df)
    t_avg, af_a, _ = physics.annual_average_lifetime(df, cal["d_ref"], Ea, 2.5)
    eol[name] = t
    af_hourly[name] = df["AF"].mean()
    af_avgtemp[name] = af_a

m = "Москва (холодный)"
s = "Сингапур (жаркий влажный)"
check_true(
    "жаркий влажный климат деградирует быстрее холодного (срок короче)",
    eol[s] < eol[m],
    f"Сингапур {eol[s]:.1f} лет < Москва {eol[m]:.1f} лет",
)
check_true(
    "неравенство Йенсена: почасовой AF > AF по средней T (выпуклость Аррениуса)",
    af_hourly[m] > af_avgtemp[m] and af_hourly[s] > af_avgtemp[s],
    f"Москва: почас {af_hourly[m]:.2f} > среднегод {af_avgtemp[m]:.2f}; "
    f"Сингапур: почас {af_hourly[s]:.2f} > среднегод {af_avgtemp[s]:.2f}",
)
check_true(
    "срок службы положителен и конечен для обоих регионов",
    0 < eol[m] < 200 and 0 < eol[s] < 200,
    f"Москва {eol[m]:.1f} лет, Сингапур {eol[s]:.1f} лет",
)

print()
print("=" * 78)
print("ПРОВЕРКА 7. Сохранение баланса деградации")
print("=" * 78)
# Сумма часовых деградаций за год = d_ref * средний AF (тождество модели)
df_bal = fetch_nasa_power_data(55.75, 37.62, "20230101", "20231231", "Москва")
df_bal = physics.calculate_plane_of_array_irradiance(df_bal, 55.75, 30)
df_bal = physics.calculate_cell_temperature(df_bal, NOCT=45)
df_bal["RH"] = df_bal["RH"] / 100.0
df_bal = physics.calculate_degradation(df_bal, cal["d_ref"], Ea, 2.5)
d_year_sum = df_bal["d_hourly"].sum()
d_year_identity = cal["d_ref"] * df_bal["AF"].mean()
check("Σ часовой деградации = d_ref · среднее(AF)", d_year_sum, d_year_identity, tol=1e-9)
# Срок службы из суммы должен совпадать с 0.20 / годовая деградация
eol_from_sum = 0.20 / d_year_sum
t_eol_code, _, _ = physics.find_eol_time(df_bal)
check("срок службы = 0.20 / годовая деградация", t_eol_code, eol_from_sum, tol=0.2)

print()
print("=" * 78)
print(f"ИТОГ: пройдено {passed}, ошибок {failed}")
print("=" * 78)
if failed == 0:
    print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ. Расчёты корректны.")
else:
    print("ЕСТЬ ОШИБКИ — см. отметки [ОШИБКА] выше.")
