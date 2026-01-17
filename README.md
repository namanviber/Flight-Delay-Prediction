# ✈️ Flight Delay Prediction

A sophisticated Machine Learning application that predicts potential flight delays by analyzing airline schedules, historical data, and real-time weather conditions. By leveraging advanced data processing and predictive modeling, it provides travelers with insights into the likelihood and duration of delays.

## 🌟 Overview

The application takes into account several critical factors:
- **Airline & Route**: Flight-specific patterns across major US carriers.
- **Temporal Factors**: Quarter, month, day of month, and day of week.
- **Real-time Weather**: Live weather data (wind, precipitation, snow, temperature) fetched via the **NOAA (NCEI) API** for both origin and destination airports.
- **Predictive Categories**: On-time, >15 min delay, >30 min delay, >45 min delay, or >1 hour delay.

## 🛠️ Tech Stack

- **Framework**: [Streamlit](https://streamlit.io/)
- **Data Science**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (Random Forest)
- **External API**: [NOAA NCEI Data Services](https://www.ncei.noaa.gov/)
- **Data Format**: [Parquet](https://parquet.apache.org/)

## 📁 Project Structure

```text
Flight-Delay-Prediction/
├── app.py                  # Streamlit app with real-time weather and inference
├── Data_preprocessing.ipynb # Notebook for data cleaning and preparation
├── airlineDelay.ipynb      # Notebook for model training and evaluation
├── eda-bda.ipynb           # Exploratory Data Analysis
├── newData.parquet         # Optimized processed dataset
├── requirements.txt        # List of Python dependencies
└── README.md               # Project documentation
```

## 🚀 Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/namanviber/Flight-Delay-Prediction.git
   cd Flight-Delay-Prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📊 Methodology

1. **Data Ingestion**: Processes large-scale flight datasets stored in Parquet format.
2. **Feature Engineering**: Calculates distances, extracts temporal features, and maps airport codes for weather lookups.
3. **Weather Integration**: Dynamically fetches daily weather summaries for the journey date using the NOAA API.
4. **Model Inference**: One-hot encodes the inputs and passes them to a pre-trained Random Forest classifier.