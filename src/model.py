import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# The columns we give to our model 
FEATURE_COLS = [
    "temperature", "windspeed", "hour", "day_of_week", "month", "is_weekend", 
    "price_lag_24h", "price_lag_48h", "price_lag_168h", "price_rolling_24h", "price_rolling_168h"
]


def load_processed_data():
    df = pd.read_csv("data/processed_data.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def prepare_features(df):
    """Divides the data in features (X) and target (y)."""
    X = df[FEATURE_COLS]   
    y = df["price"]        
    return X, y


def train_model(X_train, y_train):
    """
    Trains the XGBoost-model on the training data.
    """
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using the test data and calculating MAE, RMSE and R2.
    """
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("Model perfomance on test data:")
    print(f"   MAE:  {mae:.2f} EUR/MWh")
    print(f"   RMSE: {rmse:.2f} EUR/MWh")
    print(f"   R²:   {r2:.4f}")

    return y_pred, {"MAE": mae, "RMSE": rmse, "R2": r2}


def get_feature_importance(model):
    """
    Returns the features which contribute to reducing error the most.
    """
    importance = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    return importance


if __name__ == "__main__":
    print("Loading data...")
    df = load_processed_data()
    X, y = prepare_features(df)

    # We don't shuffle when ditributing the training and test data! 
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Trains on {len(X_train):,} hours, tests on {len(X_test):,} hours...")
    model = train_model(X_train, y_train)

    y_pred, metrics = evaluate_model(model, X_test, y_test)

    importance = get_feature_importance(model)
    print(f"\n5 most important features:")
    print(importance.head(5).to_string(index=False))

    # Saves the model as a file to use in the dashboard
    with open("data/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Saves the test results to be used in the dashboard
    test_results = df.iloc[split_idx:].copy()
    test_results["predicted_price"] = y_pred
    test_results.to_csv("data/test_results.csv", index=False)

    print("\nModel saved in data/model.pkl")
    print("Test resultats saved in data/test_results.csv")