from prefect import flow, task
import pandas as pd
from sklearn.cluster import KMeans
import xgboost as xgb
from prophet import Prophet
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

@task
def load_data():
    df = pd.read_csv("your_data.csv")
    logging.info("[INFO] Loaded CSV file.")
    return df

@task
def run_kmeans(df):
    # Auto-handle missing feature1, feature2
    numeric_cols = df.select_dtypes(include=['number']).columns
    if 'feature1' not in df.columns or 'feature2' not in df.columns:
        logging.info("[INFO] Creating 'feature1' and 'feature2'.")
        if len(numeric_cols) >= 2:
            df['feature1'] = df[numeric_cols[0]]
            df['feature2'] = df[numeric_cols[1]]
        else:
            df['feature1'] = pd.Series(range(len(df)))
            df['feature2'] = pd.Series(range(len(df)))
    model = KMeans(n_clusters=3, n_init='auto')
    df['cluster'] = model.fit_predict(df[['feature1', 'feature2']])
    logging.info("[INFO] KMeans clustering completed.")
    return df

@task
def run_xgboost(df):
    if 'target' not in df.columns:
        logging.warning("[WARNING] 'target' column not found. Skipping XGBoost.")
        return None
    X = df.drop(columns=['target'])
    y = df['target']
    model = xgb.XGBClassifier()
    model.fit(X, y)
    logging.info("[INFO] XGBoost model trained.")
    return model

@task
def run_prophet(df):
    if not {'date', 'value'}.issubset(df.columns):
        logging.warning("[WARNING] 'date' and 'value' columns are required for Prophet.")
        return None
    prophet_df = df.rename(columns={'date': 'ds', 'value': 'y'})
    model = Prophet()
    model.fit(prophet_df)
    forecast = model.predict(prophet_df)
    logging.info("[INFO] Prophet forecasting complete.")
    return forecast[['ds', 'yhat']]

@task
def save_outputs(df, forecast):
    df.to_csv("clustered_output.csv", index=False)
    if forecast is not None:
        forecast.to_csv("prophet_forecast.csv", index=False)

@flow(name="AI Dashboard Flow")
def ai_dashboard_flow():
    df = load_data()
    clustered_df = run_kmeans(df)
    xgb_model = run_xgboost(df)
    forecast = run_prophet(df)
    logging.info("[INFO] Flow execution complete.")
    save_outputs(clustered_df, forecast)
    return clustered_df, xgb_model, forecast

if __name__ == "__main__":
    ai_dashboard_flow()
