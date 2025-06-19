# 📊 AI-Driven Social Media Analytics Dashboard

This project is a fully automated, AI-powered Streamlit dashboard that gives data-driven insights into social media content performance. It combines **Prefect**, **KMeans**, **XGBoost**, and **Prophet** models to provide:

## 🚀 Features
- 🔄 **Automated AI Flow** using Prefect
- 📊 **KMeans Clustering** for content segmentation
- 📈 **Prophet Forecasting** for predicting future trends
- 🤖 **XGBoost Predictions** for content performance
- 📌 **Key Metrics** like CTR, CPC, ROI, Engagement Rate
- 🖱️ **Interactive Filters**: Date range, metric selection, and dynamic charts
- 📥 **Download Options** for filtered data and forecasts
- 📡 Ready for **Meta's Instagram API** and **Power BI**

## 📂 Folder Structure
📁 your-repo/
├── dashboard.py
├── flow.py
├── requirements.txt
├── README.md
└── data/
├── sample_clustered_output.csv
└── sample_prophet_forecast.csv

bash
Copy
Edit

## 📦 Installation
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
pip install -r requirements.txt
running dashboard
bash
Copy
Edit
# Run Prefect flow first (if needed)
python flow.py

# Launch Streamlit dashboard
streamlit run dashboard.py
📊 Data Source
