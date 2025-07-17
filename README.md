# Retail Sales Forecasting App 📈

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sales-forecasting-app-kfetpdtysqqpcnyzoj6m.streamlit.app/)

An end-to-end data science project that forecasts daily retail sales. This interactive web application was built using Python, trained on historical data with LightGBM, and deployed as a live tool using Streamlit.

This project simulates a real-world scenario where a business needs to predict future demand to optimize inventory, staffing, and marketing efforts, similar to challenges faced in an SAP data environment.


## Features

-   **Interactive UI:** Users can input various features like inventory levels, pricing, promotions, and contextual data via a clean sidebar interface.
-   **Live Predictions:** Leverages a pre-trained LightGBM model to deliver sales forecasts in real-time based on user inputs.
-   **Data Visualization:** Includes key charts from the exploratory data analysis (EDA) phase, such as the impact of promotions on sales.

## Tech Stack

-   **Language:** Python 3
-   **Data Manipulation:** Pandas
-   **Machine Learning:** Scikit-learn, LightGBM
-   **Web Framework:** Streamlit
-   **Plotting:** Altair

## Project Structure

```
sales-forecasting-project/
├── data/
│   └── retail_store_inventory.csv  # Raw dataset
├── app.py                          # Main Streamlit application script
├── lgbm_model.txt                  # Pre-trained LightGBM model
└── requirements.txt                # Required Python libraries
```

## Setup and Local Installation

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Key Learnings & Insights

-   **Model Performance:** The LightGBM model serves as a strong baseline, successfully capturing the general trends and seasonality of sales data.
-   **Feature Importance:** The model identified that **`Inventory_Level`**, **`Units_Sold_Lag_1`** (yesterday's sales), and **`Units_Ordered`** were the most significant predictors. This highlights that operational data and recent sales momentum are more critical for daily forecasting than high-level categorical features.
-   **Business Insights:** Exploratory analysis confirmed that **promotional activities** provide a significant lift in sales, whereas daily weather conditions have a negligible impact on overall sales volume.

## Future Improvements

-   **Incorporate More Granular Data:** Add specific holiday names (e.g., "Christmas," "Black Friday") instead of a simple 0/1 flag.
-   **Advanced Modeling:** Experiment with deep learning models (like LSTMs) to potentially capture more complex temporal patterns.
-   **CI/CD Pipeline:** Implement a GitHub Actions workflow to automatically retrain and deploy the model when new data becomes available.
