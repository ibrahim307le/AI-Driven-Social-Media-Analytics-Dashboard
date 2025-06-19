import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(layout="wide", page_title="AI Dashboard", page_icon="📊")
st.title("📊 AI Insights Dashboard")

# Load datasets
df = pd.read_csv("clustered_output.csv") if os.path.exists("clustered_output.csv") else None
forecast_df = pd.read_csv("prophet_forecast.csv") if os.path.exists("prophet_forecast.csv") else None

# Convert 'date' to datetime
if df is not None and 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# ========== Time Slider & Metric Filters ==========
if df is not None and 'date' in df.columns:
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.slider("📅 Select Date Range:", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    df = df[(df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])]

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    default_metrics = ['impressions', 'clicks', 'conversions']
    selected_metrics = st.multiselect("📌 Filter Metrics to Compare:", options=numeric_cols, default=[col for col in default_metrics if col in numeric_cols])

    if selected_metrics:
        st.line_chart(df[['date'] + selected_metrics].set_index("date"))

# ========== KMeans Section ==========
if df is not None:
    st.subheader("🧠 KMeans Clustering")
    st.dataframe(df.head())

    if {'feature1', 'feature2', 'cluster'}.issubset(df.columns):
        fig, ax = plt.subplots()
        for cluster_id in df['cluster'].unique():
            subset = df[df['cluster'] == cluster_id]
            ax.scatter(subset['feature1'], subset['feature2'], label=f"Cluster {cluster_id}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title("KMeans Clusters")
        ax.legend()
        st.pyplot(fig)

        st.subheader("📊 Correlation Heatmap")
        numeric_df = df.select_dtypes(include=['number'])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ========== Prophet Forecast ==========
if forecast_df is not None:
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

    st.subheader("📈 Prophet Forecasting")
    st.line_chart(forecast_df.rename(columns={"ds": "Date", "yhat": "Forecast"}).set_index("Date"))

    st.subheader("📊 Prophet Trend Analysis")
    fig, ax = plt.subplots()
    ax.plot(forecast_df['ds'], forecast_df['yhat'], color="green", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Forecast Value")
    ax.set_title("Forecast Over Time")
    st.pyplot(fig)

# ========== KPI Calculations ==========
if df is not None and {'clicks', 'impressions', 'total_spent', 'revenue', 'likes', 'shares', 'comments', 'reach'}.issubset(df.columns):
    st.subheader("📌 KPI Metrics: CTR, CPC, ROI, Engagement Rate")
    df['CTR'] = df['clicks'] / df['impressions'].replace(0, 1)
    df['CPC'] = df['total_spent'] / df['clicks'].replace(0, 1)
    df['ROI'] = df['revenue'] / df['total_spent'].replace(0, 1)
    df['Engagement Rate'] = (df['likes'] + df['shares'] + df['comments']) / df['reach'].replace(0, 1)

    st.dataframe(df[['CTR', 'CPC', 'ROI', 'Engagement Rate']].describe().T.style.format("{:.2f}"))

    fig, ax = plt.subplots(figsize=(12, 5))
    for metric in ['CTR', 'CPC', 'ROI', 'Engagement Rate']:
        ax.plot(df['date'], df[metric], label=metric)
    ax.set_title("KPI Trends Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Metric Value")
    ax.legend()
    st.pyplot(fig)

# ========== Ad Performance Metrics ==========
if df is not None and {'impressions', 'clicks', 'conversions'}.issubset(df.columns):
    st.subheader("📣 Ad Performance Metrics")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(df.index, df['impressions'], color='lightblue')
    ax[0].set_title("Impressions")

    ax[1].bar(df.index, df['clicks'], color='orange')
    ax[1].set_title("Clicks")

    ax[2].bar(df.index, df['conversions'], color='green')
    ax[2].set_title("Conversions")

    for a in ax:
        a.set_xlabel("Samples")
        a.set_ylabel("Count")

    st.pyplot(fig)

# ========== Top Performing Content ==========
if df is not None:
    st.subheader("🏆 Top Performing Content")
    if 'conversion_rate' in df.columns:
        top_content = df.sort_values('conversion_rate', ascending=False).head(5)
        st.dataframe(top_content)

# ========== XGBoost Prediction Comparison ==========
if df is not None and 'predicted_target' in df.columns and 'target' in df.columns:
    st.subheader("🧪 XGBoost Prediction Comparison")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['target'], label="Actual", linestyle='--')
    ax.plot(df.index, df['predicted_target'], label="Predicted", linestyle='-')
    ax.set_title("Actual vs Predicted Target")
    ax.legend()
    st.pyplot(fig)

# ========== Downloads ==========
st.subheader("📥 Download Processed Files")
if df is not None:
    st.download_button("Download Filtered Data", data=df.to_csv(index=False), file_name="filtered_output.csv", mime="text/csv")
if forecast_df is not None:
    st.download_button("Download Prophet Forecast", data=forecast_df.to_csv(index=False), file_name="prophet_forecast.csv", mime="text/csv")

# ========== Footer ==========
st.markdown("---")
st.caption("Developed with ❤️ using Prefect, Streamlit, Prophet, KMeans, XGBoost")
