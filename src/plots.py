"""
Построение графиков для ВКР.

Все рисунки сохраняются в папку output/ с разрешением 300 dpi (пригодно для
вставки в текст работы). Подписи на русском языке.
"""

import os
import matplotlib
matplotlib.use("Agg")  # рендеринг в файл без графического окна
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

# Единый стиль оформления
plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 110,
})


def _ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_degradation_curves(results, save_name="fig_3_1_degradation.png"):
    """
    Рисунок 3.1 — Накопленная деградация во времени по климатическим зонам.

    results — список словарей с ключами:
        region, years_axis, cumulative, t_eol
    """
    _ensure_output()
    plt.figure(figsize=(11, 7))

    for r in results:
        label = f"{r['region']} (EOL = {r['t_eol']:.1f} лет)"
        plt.plot(r["years_axis"], r["cumulative"] * 100, linewidth=2, label=label)

    plt.axhline(y=20, color="red", linestyle="--", linewidth=1.5,
                label="Порог 80 % мощности (деградация 20 %)")
    plt.axhline(y=0, color="black", linewidth=0.5)

    plt.xlabel("Время эксплуатации, годы")
    plt.ylabel("Накопленная деградация, %")
    plt.title("Прогнозируемая деградация фотоэлектрических модулей\nпо климатическим зонам")
    plt.legend(loc="lower right")
    plt.xlim(0, 30)
    plt.ylim(0, 35)

    path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  сохранён график: output/{save_name}")
    return path


def plot_weibull_survival(weibull_results, save_name="fig_3_2_weibull.png"):
    """
    Рисунок 3.2 — Кривые выживания (распределение Вейбулла) по зонам.

    weibull_results — список словарей с ключами:
        region, weibull (результат weibull_lifetime_analysis)
    """
    _ensure_output()
    plt.figure(figsize=(11, 7))

    for r in weibull_results:
        w = r["weibull"]
        if w is None:
            continue
        t_50 = w["quantiles"][50]
        plt.plot(w["t_array"], w["survival"] * 100, linewidth=2,
                 label=f"{r['region']} (медиана {t_50:.1f} лет)")

    plt.axhline(y=50, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Время эксплуатации, годы")
    plt.ylabel("Доля модулей, сохранивших > 80 % мощности, %")
    plt.title("Кривые выживания модулей по климатическим зонам\n(распределение Вейбулла, k = 3)")
    plt.legend(loc="upper right")
    plt.xlim(0, 35)
    plt.ylim(0, 105)

    path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  сохранён график: output/{save_name}")
    return path


def plot_scenario_dh1000(scenarios, save_name="fig_3_3_scenarios.png"):
    """
    Рисунок 3.3 — Влияние качества модуля (ΔP_DH в тесте DH1000) на срок службы.

    scenarios — список словарей с ключами: name, dP_DH_percent, t_eol
    """
    _ensure_output()
    plt.figure(figsize=(10, 6))

    names = [s["name"] for s in scenarios]
    eols = [s["t_eol"] for s in scenarios]
    dph = [s["dP_DH_percent"] for s in scenarios]

    bars = plt.bar(names, eols, color=["#c0392b", "#e67e22", "#27ae60", "#2980b9"])
    for bar, e, d in zip(bars, eols, dph):
        plt.text(bar.get_x() + bar.get_width() / 2, e + 0.5,
                 f"{e:.1f} лет\n(ΔP={d:.1f}%)", ha="center", va="bottom", fontsize=10)

    plt.ylabel("Прогнозируемый срок службы, годы")
    plt.title("Влияние стойкости к влажному теплу (DH1000)\nна срок службы модуля (умеренный климат)")
    plt.ylim(0, max(eols) * 1.25)

    path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  сохранён график: output/{save_name}")
    return path


def plot_monthly_temperature(df, region, save_name=None):
    """
    Вспомогательный рисунок — среднемесячная температура ячейки Tc и воздуха Ta.
    """
    _ensure_output()
    if save_name is None:
        save_name = f"fig_temp_{region}.png"

    monthly = df.resample("ME").mean(numeric_only=True)
    plt.figure(figsize=(10, 6))
    plt.plot(monthly.index, monthly["Ta"], "o-", label="Температура воздуха Ta")
    plt.plot(monthly.index, monthly["Tc"], "s-", label="Температура ячейки Tc")
    plt.xlabel("Месяц")
    plt.ylabel("Температура, °C")
    plt.title(f"Среднемесячная температура — {region}")
    plt.legend()

    path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  сохранён график: output/{save_name}")
    return path
