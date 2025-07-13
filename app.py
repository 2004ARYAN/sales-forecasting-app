# app.py

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
    """Loads the raw data from a CSV file."""
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
def load_model(model_path):
    """Loads a pre-trained LightGBM model."""
    bst = lgb.Booster(model_file=model_path)
    return bst

# --- Load Data and Model ---
raw_df = load_data('data/retail_store_inventory.csv')
model = load_model('lgbm_model.txt')

# --- UI Layout ---
st.title("📈 Retail Sales Forecasting Tool")
st.markdown("This app uses a LightGBM model to predict daily units sold for retail products. Use the sidebar to input product and store features.")

# --- Sidebar for User Inputs ---
st.sidebar.header("Forecasting Inputs")

# Get unique values for dropdowns from the raw data
categories = sorted(raw_df['Category'].unique())
regions = sorted(raw_df['Region'].unique())
weather_conditions = sorted(raw_df['Weather Condition'].unique())
seasons = sorted(raw_df['Seasonality'].unique())

# Create input widgets in the sidebar
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
    # 1. Create a DataFrame from the inputs
    current_year = pd.Timestamp.now().year
    
    input_data = {
        'Inventory Level': [inventory_level],
        'Units Ordered': [units_ordered],
        'Price': [price],
        'Discount': [discount],
        'Holiday/Promotion': [holiday_promotion],
        'Competitor Pricing': [competitor_price],
        'Year': [current_year],
        'Month': [month],
        'Day': [15], # Placeholder day, as day of month was not a top feature
        'DayOfWeek': [day_of_week],
        'WeekOfYear': [pd.Timestamp(f'{current_year}-{month}-15').isocalendar().week],
        'Units_Sold_Lag_1': [units_sold_lag_1],
        'Units_Sold_Rolling_7': [units_sold_rolling_7],
        'Category': [category],
        'Region': [region],
        'Weather Condition': [weather],
        'Seasonality': [season]
    }
    input_df = pd.DataFrame(input_data)

    # 2. One-Hot Encode the new data in the same way as the training data
    categorical_cols = ['Category', 'Region', 'Weather Condition', 'Seasonality']

    # We use the full raw_df to ensure all possible categorical values are represented in the template
    template_df = pd.get_dummies(raw_df.drop('Units Sold', axis=1), columns=categorical_cols, drop_first=True)
    
    # One-hot encode our single input row
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    # Align columns: ensures our input has the exact same columns as the model was trained on
    final_input_df = input_df_encoded.reindex(columns=template_df.columns, fill_value=0)
    
    # Ensure the order of columns matches the model's training order
    model_features = model.feature_name()
    final_input_df = final_input_df[model_features]

    # 3. Make Prediction
    prediction = model.predict(final_input_df)
    predicted_sales = int(round(prediction[0]))

    # 4. Display the result in the main panel
    st.subheader("Forecast Result")
    st.metric(label="Predicted Units Sold for the Day", value=f"{predicted_sales} units")
    st.info("Note: This prediction is based on the features provided in the sidebar and the patterns learned by the model from historical data.", icon="ℹ️")


# --- Optional: Display some of the original EDA charts ---
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
    ).properties(
        title='Impact of Promotions on Daily Units Sold'
    )
    st.altair_chart(chart, use_container_width=True)
