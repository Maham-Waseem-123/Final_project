import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# ---------------------------
# BACKEND: Data & Model Training
# ---------------------------

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Maham-Waseem-123/Final_project/main/Shale_Test.csv"
    combined_data = pd.read_csv(url)
    return combined_data

@st.cache_resource
def train_model(df):
    feature_cols = df.drop(columns=['ID', 'Production (MMcfge)']).columns.tolist()
    X = df[feature_cols]
    y = df['Production (MMcfge)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    gbr = GradientBoostingRegressor(
        loss='absolute_error',
        learning_rate=0.1,
        n_estimators=600,
        max_depth=1,
        random_state=42,
        max_features=5
    )
    gbr.fit(X_train_scaled, y_train)
    
    pred_y = gbr.predict(X_test_scaled)
    
    return gbr, scaler, feature_cols, X_test, y_test, pred_y

# Load data and train model
df = load_data()
model, scaler, feature_cols, X_test, y_test, pred_y = train_model(df)

# ---------------------------
# APPLICATION LAYOUT
# ---------------------------

st.set_page_config(page_title="Reservoir Engineering App", layout="wide")

st.sidebar.title("Pages")
page = st.sidebar.radio("Select a Page:", [
    "Economic Analysis",
    "Reservoir Engineering Dashboard",
    "Reservoir Prediction"
])

# ---------------------------
# PAGE 1: Economic Analysis
# ---------------------------

if page == "Economic Analysis":
    st.title("Economic Analysis")

    st.subheader("Adjust Cost Parameters")
    base_drilling_cost = st.slider("Base Drilling Cost ($/ft)", 500, 5000, 1000)
    base_completion_cost = st.slider("Base Completion Cost ($/ft)", 200, 2000, 500)
    proppant_cost_per_lb = st.slider("Proppant Cost ($/lb)", 0.01, 1.0, 0.1)
    water_cost_per_bbl = st.slider("Water Cost ($/bbl)", 0.5, 5.0, 1.5)
    additive_cost_per_bbl = st.slider("Additive Cost ($/bbl)", 0.5, 5.0, 2.0)
    base_maintenance_cost = st.slider("Maintenance Cost ($/year)", 10000, 100000, 30000)
    base_pump_cost = st.slider("Pump/Energy Cost ($/year)", 10000, 50000, 20000)
    gas_price = st.slider("Gas Price ($/MMcfge)", 1, 20, 5)
    
    df['CAPEX'] = (
        base_drilling_cost * df['Depth (feet)'] +
        base_completion_cost * df['Gross Perforated Interval (ft)'] +
        proppant_cost_per_lb * df['Proppant per foot (lbs)'] * df['Gross Perforated Interval (ft)'] +
        water_cost_per_bbl * df['Water per foot (bbls)'] * df['Gross Perforated Interval (ft)'] +
        additive_cost_per_bbl * df['Additive per foot (bbls)'] * df['Gross Perforated Interval (ft)']
    )
    
    df['OPEX'] = (
        base_maintenance_cost +
        base_pump_cost +
        proppant_cost_per_lb * df['Proppant per foot (lbs)'] * df['Gross Perforated Interval (ft)'] +
        water_cost_per_bbl * df['Water per foot (bbls)'] * df['Gross Perforated Interval (ft)'] +
        additive_cost_per_bbl * df['Additive per foot (bbls)'] * df['Gross Perforated Interval (ft)']
    )
    
    df['Revenue'] = df['Production (MMcfge)'] * gas_price
    df['Profit'] = df['Revenue'] - df['CAPEX'] - df['OPEX']
    
    st.subheader("Economic Metrics")
    st.dataframe(df[['ID','CAPEX','OPEX','Revenue','Profit']])
    
    st.subheader("Charts")
    fig1 = px.scatter(df, x='Production (MMcfge)', y='Revenue', size='Profit', color='Profit')
    fig2 = px.scatter(df, x='Production (MMcfge)', y='Profit', size='Revenue', color='Revenue')
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# PAGE 2: Reservoir Engineering Dashboard
# ---------------------------

elif page == "Reservoir Engineering Dashboard":
    st.title("Reservoir Engineering Dashboard")
    
    st.subheader("Production vs Depth")
    fig = px.scatter(df, x='Depth (feet)', y='Production (MMcfge)', color='Porosity (decimal)')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Production vs Porosity")
    fig = px.scatter(df, x='Porosity (decimal)', y='Production (MMcfge)', trendline="ols")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Stimulation Effectiveness")
    fig = px.scatter_3d(df,
                        x='Proppant per foot (lbs)',
                        y='Water per foot (bbls)',
                        z='Production (MMcfge)',
                        color='Additive per foot (bbls)',
                        size='Gross Perforated Interval (ft)')
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# PAGE 3: Reservoir Prediction
# ---------------------------

elif page == "Reservoir Prediction":
    st.title("Predict New Well Production")
    
    st.subheader("Input Parameters")
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
    
    input_data = {}
    for col in feature_cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        if min_val == max_val:
            max_val += 1.0
        input_data[col] = st.slider(
            col,
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=(max_val - min_val) / 1000
        )

    if st.button("Predict Production"):
        input_df = pd.DataFrame([input_data], columns=feature_cols)
        input_scaled = scaler.transform(input_df)
        pred_production = model.predict(input_scaled)[0]
        st.success(f"Predicted Production (MMcfge): {pred_production:.2f}")
