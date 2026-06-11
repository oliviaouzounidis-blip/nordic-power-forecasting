import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("data/processed_data.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

test_results = pd.read_csv("data/test_results.csv")
test_results["datetime"] = pd.to_datetime(test_results["datetime"])

with open("data/model.pkl", "rb") as f:
    model = pickle.load(f)

FEATURE_COLS = [
    "temperature", "windspeed",
    "hour", "day_of_week", "month", "is_weekend",
    "price_lag_24h", "price_lag_48h", "price_lag_168h",
    "price_rolling_24h", "price_rolling_168h"
]

# Metrics
mae  = mean_absolute_error(test_results["price"], test_results["predicted_price"])
rmse = np.sqrt(mean_squared_error(test_results["price"], test_results["predicted_price"]))
r2   = r2_score(test_results["price"], test_results["predicted_price"])

# Feature importance
importance = pd.DataFrame({
    "feature":    FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True)

# Cleaner feature labels
label_map = {
    "price_rolling_24h":  "24h Rolling Avg",
    "hour":               "Hour of Day",
    "price_lag_24h":      "Price 24h Ago",
    "is_weekend":         "Weekend",
    "day_of_week":        "Day of Week",
    "price_rolling_168h": "7d Rolling Avg",
    "windspeed":          "Wind Speed",
    "temperature":        "Temperature",
    "month":              "Month",
    "price_lag_48h":      "Price 48h Ago",
    "price_lag_168h":     "Price 7d Ago",
}
importance["label"] = importance["feature"].map(label_map)

# Colour palette
BG        = "#0D1117"   
SURFACE   = "#161B22"   
BORDER    = "#21262D"
ACCENT    = "#00D4AA"
ACCENT2   = "#FF6B35"
TEXT      = "#E6EDF3"
MUTED     = "#7D8590"

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'Inter', 'Helvetica Neue', sans-serif", color=TEXT, size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED)),
)

# Plot 1: Actual vs Predicted 
fig_predictions = go.Figure()
fig_predictions.add_trace(go.Scatter(
    x=test_results["datetime"].iloc[-500:],
    y=test_results["price"].iloc[-500:],
    name="Actual price",
    line=dict(color=ACCENT2, width=1.5)
))
fig_predictions.add_trace(go.Scatter(
    x=test_results["datetime"].iloc[-500:],
    y=test_results["predicted_price"].iloc[-500:],
    name="Predicted price",
    line=dict(color=ACCENT, width=1.5, dash="dot")
))
fig_predictions.update_layout(
    **PLOT_LAYOUT,
    title=dict(text="Actual vs Predicted — SE3 Day-Ahead Price", font=dict(size=13, color=TEXT)),
    yaxis_title="EUR / MWh",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED))
)

# Plot 2: Feature importance
fig_importance = go.Figure(go.Bar(
    x=importance["importance"],
    y=importance["label"],
    orientation="h",
    marker=dict(
        color=importance["importance"],
        colorscale=[[0, BORDER], [1, ACCENT]],
        showscale=False
    )
))
fig_importance.update_layout(
    **PLOT_LAYOUT,
    title=dict(text="Feature Importance", font=dict(size=13, color=TEXT)),
    xaxis_title="Relative weight",
)

# Plot 3: Price history
daily = df.resample("D", on="datetime")["price"].mean().reset_index()
fig_history = go.Figure(go.Scatter(
    x=daily["datetime"],
    y=daily["price"],
    mode="lines",
    line=dict(color=ACCENT, width=1.5),
    fill="tozeroy",
    fillcolor=f"rgba(0, 212, 170, 0.07)"
))
fig_history.update_layout(
    **PLOT_LAYOUT,
    title=dict(text="Historical SE3 Spot Price — Daily Average", font=dict(size=13, color=TEXT)),
    yaxis_title="EUR / MWh",
)

# Plot 4: Hourly pattern
hourly = df.groupby("hour")["price"].mean().reset_index()
fig_hourly = go.Figure(go.Bar(
    x=hourly["hour"],
    y=hourly["price"],
    marker=dict(
        color=hourly["price"],
        colorscale=[[0, SURFACE], [1, ACCENT2]],
        showscale=False
    )
))
fig_hourly.update_layout(
    **PLOT_LAYOUT,
    title=dict(text="Average Price by Hour of Day", font=dict(size=13, color=TEXT)),
    xaxis_title="Hour (CET)",
    yaxis_title="EUR / MWh",
)

# Styles
CARD = {
    "backgroundColor": SURFACE,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "padding": "20px 24px",
    "flex": "1",
}

GRAPH_CARD = {
    "backgroundColor": SURFACE,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "padding": "4px",
}

# App 
app = Dash(__name__)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Nordic Power Forecasting</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { background-color: ''' + BG + '''; font-family: 'Inter', sans-serif; }
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: ''' + BG + '''; }
            ::-webkit-scrollbar-thumb { background: ''' + BORDER + '''; border-radius: 3px; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([

    # Header 
    html.Div([
        html.Div([
            html.Div([
                html.Span("NORDIC POWER", style={
                    "fontSize": "11px", "fontWeight": "600", "letterSpacing": "3px",
                    "color": ACCENT, "display": "block", "marginBottom": "6px"
                }),
                html.H1("Day-Ahead Price Forecasting", style={
                    "fontSize": "26px", "fontWeight": "600", "color": TEXT,
                    "letterSpacing": "-0.5px"
                }),
            ]),
            html.Div([
                html.Span("SE3 Bidding Zone", style={
                    "fontSize": "12px", "color": MUTED, "display": "block", "textAlign": "right"
                }),
                html.Span("XGBoost · 2022–2024", style={
                    "fontSize": "12px", "color": MUTED, "display": "block", "textAlign": "right"
                }),
            ])
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"})
    ], style={
        "backgroundColor": SURFACE,
        "borderBottom": f"1px solid {BORDER}",
        "padding": "24px 32px",
        "marginBottom": "24px"
    }),

    # KPI cards
    html.Div([
        html.Div([
            html.Span("Mean Absolute Error", style={"fontSize": "11px", "color": MUTED,
                      "letterSpacing": "1px", "textTransform": "uppercase"}),
            html.Div(f"{mae:.2f}", style={"fontSize": "32px", "fontWeight": "600",
                     "color": TEXT, "margin": "6px 0 2px"}),
            html.Span("EUR / MWh", style={"fontSize": "12px", "color": MUTED}),
        ], style=CARD),
        html.Div([
            html.Span("Root Mean Sq. Error", style={"fontSize": "11px", "color": MUTED,
                      "letterSpacing": "1px", "textTransform": "uppercase"}),
            html.Div(f"{rmse:.2f}", style={"fontSize": "32px", "fontWeight": "600",
                     "color": TEXT, "margin": "6px 0 2px"}),
            html.Span("EUR / MWh", style={"fontSize": "12px", "color": MUTED}),
        ], style=CARD),
        html.Div([
            html.Span("R² Score", style={"fontSize": "11px", "color": MUTED,
                      "letterSpacing": "1px", "textTransform": "uppercase"}),
            html.Div(f"{r2:.3f}", style={"fontSize": "32px", "fontWeight": "600",
                     "color": ACCENT, "margin": "6px 0 2px"}),
            html.Span("1.0 = perfect fit", style={"fontSize": "12px", "color": MUTED}),
        ], style=CARD),
        html.Div([
            html.Span("Training Hours", style={"fontSize": "11px", "color": MUTED,
                      "letterSpacing": "1px", "textTransform": "uppercase"}),
            html.Div(f"{len(df):,}", style={"fontSize": "32px", "fontWeight": "600",
                     "color": TEXT, "margin": "6px 0 2px"}),
            html.Span("hourly observations", style={"fontSize": "12px", "color": MUTED}),
        ], style=CARD),
    ], style={"display": "flex", "gap": "12px", "padding": "0 32px", "marginBottom": "16px"}),

    # Row 1: Predictions + Feature importance 
    html.Div([
        html.Div(dcc.Graph(figure=fig_predictions, config={"displayModeBar": False}),
                 style={**GRAPH_CARD, "flex": "2"}),
        html.Div(dcc.Graph(figure=fig_importance, config={"displayModeBar": False}),
                 style={**GRAPH_CARD, "flex": "1"}),
    ], style={"display": "flex", "gap": "12px", "padding": "0 32px", "marginBottom": "12px"}),

    # Row 2: History + Hourly pattern 
    html.Div([
        html.Div(dcc.Graph(figure=fig_history, config={"displayModeBar": False}),
                 style={**GRAPH_CARD, "flex": "1"}),
        html.Div(dcc.Graph(figure=fig_hourly, config={"displayModeBar": False}),
                 style={**GRAPH_CARD, "flex": "1"}),
    ], style={"display": "flex", "gap": "12px", "padding": "0 32px", "marginBottom": "32px"}),

], style={"minHeight": "100vh", "backgroundColor": BG})


if __name__ == "__main__":
    app.run(debug=True)