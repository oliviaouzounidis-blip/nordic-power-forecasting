import requests
import pandas as pd
from entsoe import EntsoePandasClient

# ── Place ENTSO-E API key here ──────────────────
ENTSOE_API_KEY = "5eccbc4f-dd07-4768-938c-afb4e0743708"

def fetch_spot_prices():
    """
    Collects SE3 day-ahead spotprices from ENTSO-E for the period 2022–2024.
    Saves the result as data/spot_prices.csv with columns datetime (one line per hour) and price (EUR/MWh)
    """
    client = EntsoePandasClient(api_key=ENTSOE_API_KEY)

    start = pd.Timestamp("2022-01-01", tz="Europe/Stockholm")
    end   = pd.Timestamp("2024-12-31", tz="Europe/Stockholm")

    # Collects prices
    prices = client.query_day_ahead_prices("10Y1001A1001A46L", start=start, end=end)

    # Converts into table with columns datetime and price
    df = prices.reset_index()
    df.columns = ["datetime", "price"]

    # Removes timeszone information
    df["datetime"] = df["datetime"].dt.tz_localize(None)

    df.to_csv("data/spot_prices.csv", index=False)
    print(f"Spot prices saved. ({len(df)} rows)")
    return df


def fetch_weather_data():
    """
    Collects hourly weather data for Stockholm from Open-Meteo (no key needed).
    Saves the result as data/weather.csv with columns time, temperature windspeed
    """
    url = "https://archive-api.open-meteo.com/v1/archive"

    # Params to the API
    params = {
        "latitude":   59.33,           # Stockholm
        "longitude":  18.07,
        "start_date": "2022-01-01",
        "end_date":   "2024-12-31",
        "hourly":     ["temperature_2m", "windspeed_10m"],
        "timezone":   "Europe/Stockholm"
    }

    response = requests.get(url, params=params)
    data = response.json()   # Response comes as JSON, we convcert to table

    df = pd.DataFrame({
        "datetime":    pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"],
        "windspeed":   data["hourly"]["windspeed_10m"]
    })

    df.to_csv("data/weather.csv", index=False)
    print(f"Weather data saved. ({len(df)} rows)")
    return df


if __name__ == "__main__":
    fetch_weather_data()   
    fetch_spot_prices()    # Requires ENTSO-E API-key!