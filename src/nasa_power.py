"""
Загрузка почасовых климатических данных из открытого API NASA POWER.

NASA POWER (Prediction Of Worldwide Energy Resources) — спутниковый реанализ
климатических данных с глобальным покрытием и почасовой дискретностью.
Документация API: https://power.larc.nasa.gov/docs/services/api/

Данные кэшируются в папку data/, чтобы не скачивать их повторно при каждом
запуске (и чтобы расчёты были воспроизводимыми).
"""

import os
import json
import pandas as pd
import requests

# Папка для кэша скачанных данных
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Параметры, запрашиваемые у NASA POWER
#   T2M               — температура воздуха на высоте 2 м, °C
#   RH2M              — относительная влажность на высоте 2 м, %
#   PS                — приземное давление, кПа
#   WS2M              — скорость ветра на высоте 2 м, м/с
#   ALLSKY_SFC_SW_DWN — суммарная нисходящая коротковолновая радиация, Вт/м²
PARAMETERS = "T2M,RH2M,PS,WS2M,ALLSKY_SFC_SW_DWN"

BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"


def fetch_nasa_power_data(lat, lon, start_date, end_date, region_name="region"):
    """
    Загрузка почасовых метеоданных из API NASA POWER.

    Параметры:
        lat, lon            : float — широта и долгота
        start_date, end_date: str   — даты в формате 'YYYYMMDD'
        region_name         : str   — имя региона (для имени файла кэша)

    Возвращает:
        DataFrame с временным индексом и колонками:
            Ta         — температура воздуха, °C
            RH         — относительная влажность, %
            PS         — давление, кПа
            wind_speed — скорость ветра, м/с
            G_h        — горизонтальная инсоляция, Вт/м²
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_file = os.path.join(
        DATA_DIR, f"{region_name}_{start_date}_{end_date}.json"
    )

    # 1. Пытаемся взять данные из кэша
    if os.path.exists(cache_file):
        print(f"  [{region_name}] данные взяты из кэша: {os.path.basename(cache_file)}")
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # 2. Иначе — запрашиваем у API NASA POWER
        print(f"  [{region_name}] загрузка из NASA POWER (lat={lat}, lon={lon})...")
        payload = {
            "parameters": PARAMETERS,
            "community": "RE",          # Renewable Energy
            "longitude": lon,
            "latitude": lat,
            "start": start_date,
            "end": end_date,
            "format": "JSON",
        }
        response = requests.get(BASE_URL, params=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        # сохраняем в кэш
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    # 3. Извлечение данных в DataFrame
    parameter_block = data["properties"]["parameter"]
    df = pd.DataFrame()
    for param in PARAMETERS.split(","):
        # ключи NASA POWER имеют вид 'YYYYMMDDHH'
        series = pd.Series(parameter_block[param])
        df[param] = series

    # 4. Формирование корректного временного индекса из ключей API
    #    (это надёжнее, чем pd.date_range — длина гарантированно совпадает)
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    df.index.name = "timestamp"
    df = df.sort_index()

    # 5. NASA POWER отдаёт пропуски как -999 — заменяем их интерполяцией
    df = df.replace(-999.0, pd.NA)
    df = df.astype(float).interpolate(limit_direction="both")

    # 6. Переименование колонок для удобства
    df = df.rename(
        columns={
            "T2M": "Ta",                  # температура воздуха, °C
            "RH2M": "RH",                 # относительная влажность, %
            "WS2M": "wind_speed",         # скорость ветра, м/с
            "ALLSKY_SFC_SW_DWN": "G_h",   # горизонтальная инсоляция, Вт/м²
            # PS оставляем как есть
        }
    )

    return df
