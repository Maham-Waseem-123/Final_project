import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# ------------------------------------------------
# Backend: Load data & train model
# ------------------------------------------------
@st.cache_data
def load_and_train():
    csv_url = "https://raw.githubusercontent.com/Maham-Waseem-123/Final_project/main/Shale_Test.csv"
    combined_data = pd.read_csv(csv_url)

    df = combined_data.copy()

    X = df.drop(columns=['ID', 'Production (MMcfge)'])
    y = df['Production (MMcfge)']

    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    gbr = GradientBoostingRegressor(
        loss='absolute_error',
        learning_rate=0.1,
        n_estimators=600,
        max_depth=1,
        random_state=42,
        max_features=5
    )

    gbr.fit(X_train_scaled, y)

    return df, X, y, scaler, gbr


df, X, y, scaler, gbr = load_and_train()

# ------------------------------------------------
# App Layout
# ------------------------------------------------
st.set_page_config(page_title="Reservoir Analysis App", layout="wide")
st.title("Reservoir Analysis & Prediction App")

# ------------------------------------------------
# GLOBAL ECONOMIC PARAMETERS
# ------------------------------------------------
st.sidebar.subheader("Global Economic Parameters")

base_drilling_cost = st.sidebar.number_input("Base Drilling Cost ($/ft)", 500, 5000, 1000)
base_completion_cost = st.sidebar.number_input("Base Completion Cost ($/ft)", 100, 2000, 500)
proppant_cost_per_lb = st.sidebar.number_input("Proppant Cost ($/lb)", 0.01, 1.0, 0.1)
water_cost_per_bbl = st.sidebar.number_input("Water Cost ($/bbl)", 0.1, 10.0, 1.5)
additive_cost_per_bbl = st.sidebar.number_input("Additive Cost ($/bbl)", 0.1, 10.0, 2.0)
base_maintenance_cost = st.sidebar.number_input("Maintenance Cost ($/yr)", 10000, 100000, 30000)
base_pump_cost = st.sidebar.number_input("Pump/Energy Cost ($/yr)", 10000, 100000, 20000)
gas_price = st.sidebar.number_input("Gas Price ($/MMcfge)", 1, 100, 5)

# Page Navigation
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
            color_continuous_scale=["green", "yellow", "red"],
            size="Production (MMcfge)",
            size_max=15,
            zoom=5
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

    else:
        n_clusters = st.slider("Select number of clusters", 2, 10, 4)

        df_cluster = df.copy()
        cluster_cols = ['Depth (feet)', 'Thickness (feet)', 'Porosity (decimal)', 'Production (MMcfge)']

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_cluster['Cluster'] = kmeans.fit_predict(df_cluster[cluster_cols])

        fig = px.scatter(
            df_cluster, x='Depth (feet)', y='Production (MMcfge)',
            color='Cluster', hover_data=['ID', 'Porosity (decimal)']
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# Page 2: Economic Analysis
# ------------------------------------------------
elif page == "Economic Analysis":
    st.header("Economic Analysis: CAPEX, OPEX, Revenue & Profit")

    df_econ = df.copy()

    df_econ['CAPEX'] = (
        base_drilling_cost * df_econ['Depth (feet)'] +
        base_completion_cost * df_econ['Gross Perforated Interval (ft)'] +
        proppant_cost_per_lb * df_econ['Proppant per foot (lbs)'] * df_econ['Gross Perforated Interval (ft)'] +
        water_cost_per_bbl * df_econ['Water per foot (bbls)'] * df_econ['Gross Perforated Interval (ft)'] +
        additive_cost_per_bbl * df_econ['Additive per foot (bbls)'] * df_econ['Gross Perforated Interval (ft)']
    )

    df_econ['OPEX'] = (
        base_maintenance_cost +
        base_pump_cost +
        (proppant_cost_per_lb * df_econ['Proppant per foot (lbs)'] *
         df_econ['Gross Perforated Interval (ft)']) +
        (water_cost_per_bbl * df_econ['Water per foot (bbls)'] *
         df_econ['Gross Perforated Interval (ft)']) +
        (additive_cost_per_bbl * df_econ['Additive per foot (bbls)'] *
         df_econ['Gross Perforated Interval (ft)'])
    )

    df_econ['Revenue'] = df_econ['Production (MMcfge)'] * gas_price
    df_econ['Profit'] = df_econ['Revenue'] - df_econ['CAPEX'] - df_econ['OPEX']

    st.subheader("Production vs Economic Metrics")
    st.plotly_chart(px.scatter(df_econ, x='Production (MMcfge)', y='Revenue',
                               title='Production vs Revenue'), use_container_width=True)
    st.plotly_chart(px.scatter(df_econ, x='Production (MMcfge)', y='Profit',
                               title='Production vs Profit'), use_container_width=True)

# ------------------------------------------------
# Page 3: Reservoir Engineering Dashboard
# ------------------------------------------------
elif page == "Reservoir Engineering Dashboard":
    st.header("Reservoir Engineering Insights")

    st.plotly_chart(px.scatter(df, x='Depth (feet)', y='Production (MMcfge)',
                               trendline="ols", title="Production vs Depth"),
                    use_container_width=True)

    st.plotly_chart(px.scatter(df, x='Porosity (decimal)', y='Production (MMcfge)',
                               trendline="ols", title="Production vs Porosity"),
                    use_container_width=True)

    st.plotly_chart(px.scatter(
        df, x='Proppant per foot (lbs)', y='Production (MMcfge)',
        color='Water per foot (bbls)',
        size='Additive per foot (bbls)',
        title="Stimulation Effectiveness"
    ), use_container_width=True)

# ------------------------------------------------
# Page 4: Reservoir Prediction
# ------------------------------------------------
elif page == "Reservoir Prediction":
    st.header("Predict Production & Economic Outcomes for a New Well")

    input_dict = {}
    feature_cols = list(X.columns)

    for col in feature_cols:
        input_dict[col] = st.number_input(
            col,
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    pred_production = gbr.predict(input_scaled)[0]

    st.success(f"Predicted Production: {pred_production:.2f} MMcfge")

    predicted_capex = (
        base_drilling_cost * input_dict['Depth (feet)'] +
        base_completion_cost * input_dict['Gross Perforated Interval (ft)'] +
        proppant_cost_per_lb * input_dict['Proppant per foot (lbs)'] *
        input_dict['Gross Perforated Interval (ft)'] +
        water_cost_per_bbl * input_dict['Water per foot (bbls)'] *
        input_dict['Gross Perforated Interval (ft)'] +
        additive_cost_per_bbl * input_dict['Additive per foot (bbls)'] *
        input_dict['Gross Perforated Interval (ft)']
    )

    predicted_opex = (
        base_maintenance_cost +
        base_pump_cost +
        (proppant_cost_per_lb * input_dict['Proppant per foot (lbs)'] *
         input_dict['Gross Perforated Interval (ft)']) +
        (water_cost_per_bbl * input_dict['Water per foot (bbls)'] *
         input_dict['Gross Perforated Interval (ft)']) +
        (additive_cost_per_bbl * input_dict['Additive per foot (bbls)'] *
         input_dict['Gross Perforated Interval (ft)'])
    )

    predicted_revenue = pred_production * gas_price
    predicted_profit = predicted_revenue - predicted_capex - predicted_opex

    st.write(f"**Predicted CAPEX:** ${predicted_capex:,.2f}")
    st.write(f"**Predicted OPEX:** ${predicted_opex:,.2f}")
    st.write(f"**Predicted Revenue:** ${predicted_revenue:,.2f}")
    st.write(f"**Predicted Profit:** ${predicted_profit:,.2f}")
