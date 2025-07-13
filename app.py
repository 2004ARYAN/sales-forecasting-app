# app script.

You can copy and paste this entire block of code to replace the existing `app.py` file in your GitHub repository.py

import streamlit as st
import pandas as pd
import lightgbm as lgb
import numpy as np. This version includes the bug fix for the `NameError`.

---

```python
# app.py

import streamlit
import altair as alt

# --- Page Configuration ---
st.set_page_config(
    page_ as st
import pandas as pd
import lightgbm as lgb
import numpy as np
import altair astitle="Retail Sales Forecaster",
    page_icon="📈",
    layout="wide"
)

 alt

# --- Page Configuration ---
st.set_page_config(
    page_title="Retail Sales# --- Caching Functions for Performance ---
@st.cache_data
def load_data(path):
 Forecaster",
    page_icon="📈",
    layout="wide"
)

# --- Caching Functions for Performance    """Loads the raw data from a CSV file."""
    df = pd.read_csv(path)
 ---
@st.cache_data
def load_data(path):
    """Loads the raw data from a CSV file    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.."""
    df = pd.read_csv(path)
    df['Date'] = pd.to_cache_resource
def load_model(model_path):
    """Loads a pre-trained LightGBM model."""datetime(df['Date'])
    return df

@st.cache_resource
def load_model(model
    bst = lgb.Booster(model_file=model_path)
    return bst

# --- Load Data and Model ---
raw_df = load_data('data/retail_store_inventory._path):
    """Loads a pre-trained LightGBM model."""
    bst = lgb.Booster(model_file=model_path)
    return bst

# --- Load Data and Model ---
csv')
model = load_model('lgbm_model.txt')

# --- UI Layout ---
st.title("📈 Retail Sales Forecasting Tool")
st.markdown("This app uses a LightGBM model to predictraw_df = load_data('data/retail_store_inventory.csv')
model = load_model('lgbm_model.txt')

# --- UI Layout ---
st.title("📈 Retail Sales Forecasting Tool daily units sold for retail products. Use the sidebar to input product and store features.")

# --- Sidebar for User Inputs ---
st")
st.markdown("This app uses a LightGBM model to predict daily units sold for retail products. Use.sidebar.header("Forecasting Inputs")

# Get unique values for dropdowns from the raw data
categories = the sidebar to input product and store features.")

# --- Sidebar for User Inputs ---
st.sidebar.header("Fore raw_df['Category'].unique()
regions = raw_df['Region'].unique()
weather_conditions =casting Inputs")

# Get unique values for dropdowns from the raw data
categories = sorted(raw_df['Category'].unique raw_df['Weather Condition'].unique()
seasons = raw_df['Seasonality'].unique()

#())
regions = sorted(raw_df['Region'].unique())
weather_conditions = sorted(raw_df Create input widgets in the sidebar
inventory_level = st.sidebar.number_input("Current Inventory Level", min['Weather Condition'].unique())
seasons = sorted(raw_df['Seasonality'].unique())

# Create_value=0, value=200)
units_ordered = st.sidebar.number_input("Units Ordered Recently", min_value=0, value=50)
price = st.sidebar.slider(" input widgets in the sidebar
inventory_level = st.sidebar.number_input("Current Inventory Level", min_value=0Product Price", min_value=10.0, max_value=100.0, value=, value=200, step=10)
units_ordered = st.sidebar.number_input("Units Ordered Recently", min_value=0, value=50, step=5)
price = st.sidebar55.0, step=0.01)
competitor_price = st.sidebar.slider("Competitor's Price", min_value=5.0, max_value=105.0, value=.slider("Product Price ($)", min_value=10.0, max_value=100.0, value=55.0, step=0.01)
competitor_price = st.sidebar.55.0, step=0.01)
discount = st.sidebar.slider("Discount (%)", min_value=0, max_value=20, value=10)
holiday_promotion = stslider("Competitor's Price ($)", min_value=5.0, max_value=105.0, value=55.0, step=0.01)
discount = st.sidebar..sidebar.selectbox("Is there a Holiday/Promotion?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

st.sidebar.markdown("--- Time & Context Features ---")slider("Discount (%)", min_value=0, max_value=20, value=10)
holiday_promotion = st.sidebar.selectbox("Is there a Holiday/Promotion?", [0, 1], format
month = st.sidebar.selectbox("Month", range(1, 13))
day_of_week = st.sidebar.selectbox("Day of the Week", range(7), format_func=lambda x: ['_func=lambda x: 'Yes' if x == 1 else 'No')

st.sidebar.markdown("---")Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
units
st.sidebar.markdown("##### Time & Context Features")
day_of_week = st.sidebar._sold_lag_1 = st.sidebar.number_input("Units Sold Yesterday (Lag 1)", minselectbox("Day of the Week", range(7), format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
month = st.sidebar._value=0, value=100)
units_sold_rolling_7 = st.sidebar.number_input("Avg. Sales Last 7 Days (Rolling)", min_value=0.0, value=100selectbox("Month", range(1, 13))
units_sold_lag_1 = st.sidebar.number_input("Units Sold Yesterday (Lag 1)", min_value=0, value=100)
units_sold.0, step=0.1)

st.sidebar.markdown("--- Categorical Features ---")
category = st._rolling_7 = st.sidebar.number_input("Avg. Sales Last 7 Days (Rolling)", minsidebar.selectbox("Product Category", categories)
region = st.sidebar.selectbox("Store Region", regions)
weather = st.sidebar.selectbox("Weather Condition", weather_conditions)
season = st.sidebar.selectbox("Seasonality",_value=0.0, value=100.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.markdown("##### Categorical Features")
category = st.sidebar.selectbox("Product seasons)


# --- Prediction Logic ---
if st.sidebar.button("Forecast Sales", type="primary"):
    # 1. Create a DataFrame from the inputs
    # Using the current year for predictions
    current_year = pd. Category", categories)
region = st.sidebar.selectbox("Store Region", regions)
weather = st.sidebar.selectbox("Weather Condition", weather_conditions)
season = st.sidebar.selectbox("Seasonality", seasons)Timestamp.now().year
    
    input_data = {
        'Inventory Level': [inventory_level],
        'Units Ordered': [units_ordered],
        'Price': [price],
        'Discount


# --- Prediction Logic ---
if st.sidebar.button("Forecast Sales", type="primary"):
    # 1. Create a DataFrame from the inputs
    current_year = pd.Timestamp.now().year
    ': [discount],
        'Holiday/Promotion': [holiday_promotion],
        'Competitor Pricing': [input_data = {
        'Inventory Level': [inventory_level],
        'Units Ordered': [unitscompetitor_price],
        'Year': [current_year],
        'Month': [month],
        'Day_ordered],
        'Price': [price],
        'Discount': [discount],
        'Holiday/Promotion': [holiday_promotion],
        'Competitor Pricing': [competitor_price],
        'Year': [15], # Using a placeholder day, as Day was not a top feature
        'DayOfWeek': [day_of_week],
        'WeekOfYear': [pd.Timestamp(f'{current_year}-{month': [current_year],
        'Month': [month],
        'Day': [15], #}-15').isocalendar().week],
        'Units_Sold_Lag_1': [units_sold_lag_1],
        'Units_Sold_Rolling_7': [units_sold_rolling_7],
 Placeholder day, as day of month was not a top feature
        'DayOfWeek': [day_of_week],
        'WeekOfYear': [pd.Timestamp(f'{current_year}-{month}-15').isocalendar().        'Category': [category],
        'Region': [region],
        'Weather Condition': [weather],
        'Seasonality': [season]
    }
    input_df = pd.DataFrame(input_week],
        'Units_Sold_Lag_1': [units_sold_lag_1],
        data)

    # 2. One-Hot Encode the new data in the same way as the training data
'Units_Sold_Rolling_7': [units_sold_rolling_7],
        'Category': [category],
        'Region': [region],
        'Weather Condition': [weather],
        'Seasonality    # THIS IS THE FIX: Define the list of categorical columns before using it.
    categorical_cols = ['Category', 'Region', 'Weather Condition', 'Seasonality']

    # Create a template from the original data to get all possible encoded': [season]
    }
    input_df = pd.DataFrame(input_data)

    # 2. One-Hot Encode the new data in the same way as the training data
    # --- THIS IS columns
    template_df = pd.get_dummies(raw_df.drop('Units Sold', axis=1), columns=categorical_cols, drop_first=True)
    
    # One-hot encode our THE FIX ---
    # Define the list of categorical columns before using it.
    categorical_cols = ['Category', 'Region', 'Weather Condition', 'Seasonality']

    # Create a template from the original data to get all single row of input data
    input_df_encoded = pd.get_dummies(input_df, columns possible columns
    # We use the full raw_df to ensure all possible categorical values are represented
    template_df = pd.get_dummies(raw_df.drop('Units Sold', axis=1), columns=categorical=categorical_cols, drop_first=True)
    
    # Align columns: ensures our input has the exact same columns as the model was trained on
    # Any missing columns in our input will be added and filled with 0_cols, drop_first=True)
    
    # One-hot encode our single input row
    input.
    final_input_df = input_df_encoded.reindex(columns=template_df._df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    # Align columns: ensures our input has the exact same columns as the model was trainedcolumns, fill_value=0)
    
    # Ensure the order of columns matches the model's expectation
    model_features = model.feature_name()
    final_input_df = final_input_ on
    # Any missing columns in our input will be added and filled with 0.
    final_input_dfdf[model_features]

    # 3. Make Prediction
    prediction = model.predict(final_ = input_df_encoded.reindex(columns=template_df.columns, fill_value=0)input_df)
    predicted_sales = int(round(prediction[0]))

    # 4.
    
    # Ensure the order of columns matches the model's training order
    model_features = model. Display the result
    st.subheader("Forecast Result")
    st.metric(label="Predicted Units Sold for the Day", value=f"{predicted_sales} units")
    st.info("Note: This prediction isfeature_name()
    final_input_df = final_input_df[model_features]

    # 3. Make Prediction
    prediction = model.predict(final_input_df)
    predicted_ based on the features provided in the sidebar and the patterns learned by the model from historical data.", icon="ℹ️")

#sales = int(round(prediction[0]))

    # 4. Display the result in the main panel
    st. --- Optional: Display some of the original EDA charts ---
st.markdown("---")
st.subheader("Explsubheader("Forecast Result")
    st.metric(label="Predicted Units Sold for the Day", value=f"{predicted_salesoratory Data Analysis Insights")
agree = st.checkbox("Show EDA Charts?")
if agree:
    st.markdown("} units")
    st.info("Note: This prediction is based on the features provided in the sidebar and the#### Impact of Promotions on Sales")
    promo_chart_df = raw_df.copy()
    promo_chart patterns learned by the model from historical data.", icon="ℹ️")


# --- Optional: Display some of the original EDA charts_df['Promotion_Status'] = promo_chart_df['Holiday/Promotion'].map({0: 'No Promotion', 1: 'Promotion'})
    
    chart = alt.Chart(promo_chart_df). ---
st.markdown("---")
st.subheader("Exploratory Data Analysis Insights")
if st.checkbox("mark_boxplot(extent='min-max').encode(
        x='Promotion_Status:N',
        Show EDA Charts?", value=True):
    st.markdown("#### Impact of Promotions on Sales")
    
y='Units Sold:Q'
    ).properties(
        title='Impact of Promotions on Daily Units Sold'    chart = alt.Chart(raw_df).mark_boxplot(extent='min-max').encode(
        x
    )
    st.altair_chart(chart, use_container_width=True)
