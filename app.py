# streamlit_reservoir_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# ------------------------------------------------
# Backend: Load data and train model (hidden from front-end)
# ------------------------------------------------
@st.cache_data
def load_and_train():
    # Load dataset
    combined_data = pd.read_csv('/content/finaldata.csv')
    df = combined_data.copy()
    
    # Split features & target
    X = df.drop(columns=['ID', 'Production (MMcfge)'])
    y = df['Production (MMcfge)']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(
        loss='absolute_error',
        learning_rate=0.1,
        n_estimators=600,
        max_depth=1,
        random_state=42,
        max_features=5
    )
    gbr.fit(X_train_scaled, y_train)

    # Predict test set
    y_pred = gbr.predict(X_test_scaled)

    return df, X, y, scaler, gbr

df, X, y, scaler, gbr = load_and_train()

# ------------------------------------------------
# Streamlit App Layout
# ------------------------------------------------
st.set_page_config(page_title="Reservoir Analysis App", layout="wide")
st.title("Reservoir Analysis & Prediction App")

# Sidebar Navigation
pages = ["Spatial Visualization", "Economic Analysis", "Reservoir Engineering Dashboard", "Reservoir Prediction"]
page = st.sidebar.radio("Select Page", pages)

# ------------------------------------------------
# Page 1: Spatial Visualization
# ------------------------------------------------
if page == "Spatial Visualization":
    st.header("Spatial Visualization of Wells & Production Zones")

    option = st.radio("Choose Visualization", ["Map Visualization", "Cluster Analysis / Zonation"])

    if option == "Map Visualization":
        fig = px.scatter_mapbox(
            df,
            lat="Surface Latitude",
            lon="Surface Longitude",
            color="Production (MMcfge)",
            color_continuous_scale=["green","yellow","red"],
            size="Production (MMcfge)",
            size_max=15,
            zoom=5
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

    else:  # Cluster Analysis
        n_clusters = st.slider("Select number of clusters", 2, 10, 4)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_cols = ['Depth (feet)', 'Thickness (feet)', 'Porosity (decimal)', 'Production (MMcfge)']
        df['Cluster'] = kmeans.fit_predict(df[cluster_cols])
        fig = px.scatter(
            df, x='Depth (feet)', y='Production (MMcfge)',
            color='Cluster', hover_data=['ID', 'Porosity (decimal)']
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# Page 2: Economic Analysis
# ------------------------------------------------
elif page == "Economic Analysis":
    st.header("Economic Analysis: CAPEX, OPEX, Revenue & Profit")

    st.subheader("Adjust Base Costs")
    base_drilling_cost = st.number_input("Base Drilling Cost ($/ft)", 500, 5000, 1000)
    base_completion_cost = st.number_input("Base Completion Cost ($/ft)", 100, 2000, 500)
    proppant_cost_per_lb = st.number_input("Proppant Cost ($/lb)", 0.01, 1.0, 0.1)
    water_cost_per_bbl = st.number_input("Water Cost ($/bbl)", 0.1, 10.0, 1.5)
    additive_cost_per_bbl = st.number_input("Additive Cost ($/bbl)", 0.1, 10.0, 2.0)
    base_maintenance_cost = st.number_input("Maintenance Cost ($/yr)", 10000, 100000, 30000)
    base_pump_cost = st.number_input("Pump/Energy Cost ($/yr)", 10000, 100000, 20000)
    gas_price = st.number_input("Gas Price ($/MMcfge)", 1, 100, 5)

    # CAPEX
    df['CAPEX'] = (
        base_drilling_cost * df['Depth (feet)'] +
        base_completion_cost * df['Gross Perforated Interval (ft)'] +
        proppant_cost_per_lb * df['Proppant per foot (lbs)'] * df['Gross Perforated Interval (ft)'] +
        water_cost_per_bbl * df['Water per foot (bbls)'] * df['Gross Perforated Interval (ft)'] +
        additive_cost_per_bbl * df['Additive per foot (bbls)'] * df['Gross Perforated Interval (ft)']
    )

    # OPEX
    df['OPEX'] = (
        base_maintenance_cost +
        base_pump_cost +
        (proppant_cost_per_lb * df['Proppant per foot (lbs)'] * df['Gross Perforated Interval (ft)']) +
        (water_cost_per_bbl * df['Water per foot (bbls)'] * df['Gross Perforated Interval (ft)']) +
        (additive_cost_per_bbl * df['Additive per foot (bbls)'] * df['Gross Perforated Interval (ft)'])
    )

    # Revenue & Profit
    df['Revenue'] = df['Production (MMcfge)'] * gas_price
    df['Profit'] = df['Revenue'] - df['CAPEX'] - df['OPEX']

    st.subheader("Production vs Economic Metrics")
    st.write(px.scatter(df, x='Production (MMcfge)', y='Revenue', title='Production vs Revenue'))
    st.write(px.scatter(df, x='Production (MMcfge)', y='Profit', title='Production vs Profit'))

# ------------------------------------------------
# Page 3: Reservoir Engineering Dashboard
# ------------------------------------------------
elif page == "Reservoir Engineering Dashboard":
    st.header("Reservoir Engineering Insights")

    st.write(px.scatter(df, x='Depth (feet)', y='Production (MMcfge)', trendline="ols", title="Production vs Depth"))
    st.write(px.scatter(df, x='Porosity (decimal)', y='Production (MMcfge)', trendline="ols", title="Production vs Porosity"))
    st.write(px.scatter(df, x='Proppant per foot (lbs)', y='Production (MMcfge)', color='Water per foot (bbls)',
                        size='Additive per foot (bbls)', title="Stimulation Effectiveness"))

# ------------------------------------------------
# Page 4: Reservoir Prediction
# ------------------------------------------------
elif page == "Reservoir Prediction":
    st.header("Predict Production & Economic Outcomes for a New Well")

    # Collect inputs via sliders
    input_dict = {}
    feature_cols = list(X.columns)
    for col in feature_cols:
        input_dict[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    pred_production = gbr.predict(input_scaled)[0]

    st.success(f"Predicted Production: {pred_production:.2f} MMcfge")

    # Optional: Calculate economic outcomes based on predicted production
    predicted_capex = (
        base_drilling_cost * input_dict['Depth (feet)'] +
        base_completion_cost * input_dict['Gross Perforated Interval (ft)'] +
        proppant_cost_per_lb * input_dict['Proppant per foot (lbs)'] * input_dict['Gross Perforated Interval (ft)'] +
        water_cost_per_bbl * input_dict['Water per foot (bbls)'] * input_dict['Gross Perforated Interval (ft)'] +
        additive_cost_per_bbl * input_dict['Additive per foot (bbls)'] * input_dict['Gross Perforated Interval (ft)']
    )
    predicted_opex = (
        base_maintenance_cost +
        base_pump_cost +
        (proppant_cost_per_lb * input_dict['Proppant per foot (lbs)'] * input_dict['Gross Perforated Interval (ft)']) +
        (water_cost_per_bbl * input_dict['Water per foot (bbls)'] * input_dict['Gross Perforated Interval (ft)']) +
        (additive_cost_per_bbl * input_dict['Additive per foot (bbls)'] * input_dict['Gross Perforated Interval (ft)'])
    )
    predicted_revenue = pred_production * gas_price
    predicted_profit = predicted_revenue - predicted_capex - predicted_opex

    st.write(f"Predicted CAPEX: ${predicted_capex:,.2f}")
    st.write(f"Predicted OPEX: ${predicted_opex:,.2f}")
    st.write(f"Predicted Revenue: ${predicted_revenue:,.2f}")
    st.write(f"Predicted Profit: ${predicted_profit:,.2f}")
