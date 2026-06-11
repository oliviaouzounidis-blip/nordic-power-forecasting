import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def load_and_merge_data():
    """
    Reads the two CSV-files and merges them into a table
    """
    prices = pd.read_csv("data/spot_prices.csv")
    prices["datetime"] = pd.to_datetime(prices["datetime"])

    weather = pd.read_csv("data/weather.csv")
    weather["datetime"] = pd.to_datetime(weather["datetime"])

    # how="inner" means we only save deata for rows where both tables have values for that hour.
    df = pd.merge(prices, weather, on="datetime", how="inner")
    return df


def clean_data(df):
    """
    Cleans the data:
    1. Removes rows with missing values (NaN)
    2. Removes extreme price spikes using the IQR-method (Interquartile Range)
    """
    df = df.dropna()

    Q1  = df["price"].quantile(0.25)
    Q3  = df["price"].quantile(0.75)
    IQR = Q3 - Q1

    df = df[(df["price"] >= Q1 - 1.5 * IQR) & (df["price"] <= Q3 + 1.5 * IQR)]

    return df.reset_index(drop=True)


def feature_engineering(df):
    """
    Extracts time patterns (hour, weekday, month) and creates price history variables (lagged prices)
    to give the model more useful information than just raw timestamps.

    """
    df = df.copy()

    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"]       = df["datetime"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int) 

    df["price_lag_24h"]  = df["price"].shift(24)
    df["price_lag_48h"]  = df["price"].shift(48)
    df["price_lag_168h"] = df["price"].shift(168)   

    # Used to smooth out large variations
    df["price_rolling_24h"]  = df["price"].rolling(24).mean()
    df["price_rolling_168h"] = df["price"].rolling(168).mean()

    df = df.dropna().reset_index(drop=True)

    return df


def plot_price_distribution(df):
    fig = px.histogram(
        df, x="price",
        title="Distribution of spot-prices SE3 (2022–2024)",
        labels={"price": "Price (EUR/MWh)"},
        nbins=50,
        color_discrete_sequence=["#2196F3"]
    )
    fig.show()


def plot_price_over_time(df):
    daily = df.resample("D", on="datetime")["price"].mean().reset_index()
    fig = px.line(
        daily, x="datetime", y="price",
        title="Daily average price SE3 (2022–2024)",
        labels={"price": "Price (EUR/MWh)", "datetime": "Date"}
    )
    fig.show()


def plot_hourly_pattern(df):
    hourly = df.groupby("hour")["price"].mean().reset_index()
    fig = px.bar(
        hourly, x="hour", y="price",
        title="Average price per hour of the day",
        labels={"price": "Average price (EUR/MWh)", "hour": "Hour"},
        color_discrete_sequence=["#FF5722"]
    )
    fig.show()


def plot_correlations(df):
    """
    Creates a correlation matrix.
    """
    corr_cols    = ["price", "temperature", "windspeed", "hour", "month"]
    corr_matrix  = df[corr_cols].corr()

    fig = px.imshow(
        corr_matrix,
        title="Correlation matrix",
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1
    )
    fig.show()


if __name__ == "__main__":
    print("Loadin and mergeing data...")
    df = load_and_merge_data()

    print("Cleaning data...")
    df = clean_data(df)

    print("Creating features...")
    df = feature_engineering(df)

    print(f"\nSize of the dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\n Price statistics:\n{df['price'].describe().round(2)}")

    print("\nCreating charts...")
    plot_price_distribution(df)
    plot_price_over_time(df)
    plot_hourly_pattern(df)
    plot_correlations(df)

    # Saves the cleaned and processed data – model.py reads it from here
    df.to_csv("data/processed_data.csv", index=False)
    print("\nProcessed data saved in data/processed_data.csv")