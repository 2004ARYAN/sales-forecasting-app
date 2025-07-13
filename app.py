# app.py (Final, Working Version)

import streamlit as st
import pandas as pd
import lightgbm as lgb
import numpy as np
import altair as alt

# --- Page Configuration ---
st.set_page_config(
    page_title="Retail Sales Forecaster",
    page_icon="📈",
    layout="wide"
)

# --- Caching Functions for Performance ---
@st.cache_data
def load_data(path):
    """Loads and provides a clean copy of the raw data."""
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained LightGBM model."""
    bst = lgb.Booster(model_file=model_path)
    return bst

# --- Feature Engineering Function ---
def engineer_features(df):
    """Creates all features needed for the model from a raw dataframe."""
    df_copy = df.copy()
    df_copy['Year'] = df_copy['Date'].dt.year
    df_copy['Month'] = df_copy['Date'].dt.month
    df_copy['Day'] = df_copy['Date'].dt.day
    df_copy['DayOfWeek'] = df_copy['Date'].dt.dayofweek
    df_copy['WeekOfYear'] = df_copy['Date'].dt.isocalendar().week.astype(int)
    df_copy.sort_values(['Store ID', 'Product ID', 'Date'], inplace=True)
    df_copy['Units_Sold_Lag_1'] = df_copy.groupby(['Store ID', 'Product ID'])['Units Sold'].shift(1)
    df_copy['Units_Sold_Rolling_7'] = df_copy.groupby(['Store ID', 'Product ID'])['Units Sold'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    df_copy.fillna(0, inplace=True)
    return df_copy

# --- Load Data and Model ---
raw_df = load_data('data/retail_store_inventory.csv')
model = load_model('lgbm_model.txt')

# --- UI Layout ---
st.title("📈 Retail Sales Forecasting Tool")
st.markdown("This app uses a LightGBM model to predict daily units sold. Use the sidebar to input product and store features.")

# --- Sidebar Inputs (No changes needed here) ---
st.sidebar.header("Forecasting Inputs")
categories = sorted(raw_df['Category'].unique())
regions = sorted(raw_df['Region'].unique())
weather_conditions = sorted(raw_df['Weather Condition'].unique())
seasons = sorted(raw_df['Seasonality'].unique())
inventory_level = st.sidebar.number_input("Current Inventory Level", min_value=0, value=200, step=10)
units_ordered = st.sidebar.number_input("Units Ordered Recently", min_value=0, value=50, step=5)
price = st.sidebar.slider("Product Price ($)", min_value=10.0, max_value=100.0, value=55.0, step=0.01)
competitor_price = st.sidebar.slider("Competitor's Price ($)", min_value=5.0, max_value=105.0, value=55.0, step=0.01)
discount = st.sidebar.slider("Discount (%)", min_value=0, max_value=20, value=10)
holiday_promotion = st.sidebar.selectbox("Is there a Holiday/Promotion?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
st.sidebar.markdown("---")
st.sidebar.markdown("##### Time & Context Features")
day_of_week = st.sidebar.selectbox("Day of the Week", range(7), format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
month = st.sidebar.selectbox("Month", range(1, 13))
units_sold_lag_1 = st.sidebar.number_input("Units Sold Yesterday (Lag 1)", min_value=0, value=100)
units_sold_rolling_7 = st.sidebar.number_input("Avg. Sales Last 7 Days (Rolling)", min_value=0.0, value=100.0, step=1.0)
st.sidebar.markdown("---")
st.sidebar.markdown("##### Categorical Features")
category = st.sidebar.selectbox("Product Category", categories)
region = st.sidebar.selectbox("Store Region", regions)
weather = st.sidebar.selectbox("Weather Condition", weather_conditions)
season = st.sidebar.selectbox("Seasonality", seasons)


# --- Prediction Logic ---
if st.sidebar.button("Forecast Sales", type="primary"):
    current_year = pd.Timestamp.now().year
    
    input_data = {
        'Inventory Level': [inventory_level], 'Units Ordered': [units_ordered],
        'Price': [price], 'Discount': [discount], 'Holiday/Promotion': [holiday_promotion],
        'Competitor Pricing': [competitor_price], 'Year': [current_year], 'Month': [month],
        'Day': [15], 'DayOfWeek': [day_of_week],
        'WeekOfYear': [pd.Timestamp(f'{current_year}-{month}-15').isocalendar().week],
        'Units_Sold_Lag_1': [units_sold_lag_1], 'Units_Sold_Rolling_7': [units_sold_rolling_7],
        'Category': [category], 'Region': [region],
        'Weather Condition': [weather], 'Seasonality': [season]
    }
    input_df = pd.DataFrame(input_data)

    # Create a perfect template
    full_template_df = engineer_features(raw_df)
    categorical_cols = ['Category', 'Region', 'Weather Condition', 'Seasonality']
    final_template = pd.get_dummies(full_template_df, columns=categorical_cols, drop_first=True)

    # One-hot encode user input
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    # Align user input to the template
    final_input_df = input_df_encoded.reindex(columns=final_template.columns, fill_value=0)
    
    # --- THIS IS THE FIX ---
    # Sanitize the column names to match the model's expected format (replace spaces with underscores)
    final_input_df.columns = [col.replace(' ', '_') for col in final_input_df.columns]
    
    # Select and order columns exactly as the model expects
    model_features = model.feature_name()
    final_input_df = final_input_df[model_features]

    # Make Prediction
    prediction = model.predict(final_input_df)
    predicted_sales = int(round(prediction[0]))

    # Display the result
    st.subheader("Forecast Result")
    st.metric(label="Predicted Units Sold for the Day", value=f"{predicted_sales} units")
    st.info("Note: This prediction is based on the features provided in the sidebar and the patterns learned by the model from historical data.", icon="ℹ️")

# --- EDA Charts (No changes needed here) ---
st.markdown("---")
st.subheader("Exploratory Data Analysis Insights")
if st.checkbox("Show EDA Charts?", value=True):
    st.markdown("#### Impact of Promotions on Sales")
    promo_chart_df = raw_df.copy()
    promo_chart_df['Promotion_Status'] = promo_chart_df['Holiday/Promotion'].map({0: 'No Promotion', 1: 'Promotion'})
    chart = alt.Chart(promo_chart_df).mark_boxplot(extent='min-max').encode(
        x=alt.X('Promotion_Status:N', title='Promotion Status'),
        y=alt.Y('Units Sold:Q', title='Units Sold'),
        color=alt.Color('Promotion_Status:N', legend=None)
    ).properties(title='Impact of Promotions on Daily Units Sold')
    st.altair_chart(chart, use_container_width=True)
