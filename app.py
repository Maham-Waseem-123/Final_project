import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# ---------------------------
# BACKEND: Data & Model Training
# ---------------------------

@st.cache_data
def load_data():
    # Load dataset directly from GitHub
    url = "https://raw.githubusercontent.com/Maham-Waseem-123/Final_project/main/Shale_Test.csv"
    combined_data = pd.read_csv(url)
    return combined_data

@st.cache_resource
def train_model(df):
    X = df.drop(columns=['ID', 'Production (MMcfge)'])
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
    
    return gbr, scaler, X_train.columns, X_test, y_test, pred_y

df = load_data()
model, scaler, feature_cols, X_test, y_test, pred_y = train_model(df)

# ---------------------------
# APPLICATION LAYOUT
# ---------------------------
st.set_page_config(page_title="Reservoir Engineering App", layout="wide")

st.sidebar.title("Pages")
page = st.sidebar.radio("Select a Page:", [
    "Spatial Visualization",
    "Economic Analysis",
    "Reservoir Engineering Dashboard",
    "Reservoir Prediction"
])

# ---------------------------
# PAGE 1: Spatial Visualization
# ---------------------------
# ---------------------------
# PAGE 1: Spatial Visualization
# ---------------------------
if page == "Spatial Visualization":
    st.title("Spatial Visualization")
    st.subheader("Productive Wells Scatter Plot")

    # Define production zones
    def categorize_production(production):
        if production > 1500:   # high production
            return "Productive"
        elif production >= 1300: # moderate production
            return "Moderate"
        else:                  # low production
            return "Unproductive"

    df['Production Zone'] = df['Production (MMcfge)'].apply(categorize_production)

    # Filter only productive wells
    productive_df = df[df['Production Zone'] == "Productive"]

    # Scatter plot: Depth vs Production
    fig = px.scatter(
        productive_df,
        x="Depth (feet)",
        y="Production (MMcfge)",
        size="Gross Perforated Interval (ft)",  # optional: size based on interval
        color="Porosity (decimal)",            # optional: color by porosity
        hover_data=['ID', 'Proppant per foot (lbs)', 'Water per foot (bbls)'],
        title="Productive Wells Scatter Plot"
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# PAGE 2: Economic Analysis
# ---------------------------
elif page == "Economic Analysis":
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
    
    # Calculate CAPEX
    df['CAPEX'] = (
        base_drilling_cost * df['Depth (feet)'] +
        base_completion_cost * df['Gross Perforated Interval (ft)'] +
        proppant_cost_per_lb * df['Proppant per foot (lbs)'] * df['Gross Perforated Interval (ft)'] +
        water_cost_per_bbl * df['Water per foot (bbls)'] * df['Gross Perforated Interval (ft)'] +
        additive_cost_per_bbl * df['Additive per foot (bbls)'] * df['Gross Perforated Interval (ft)']
    )
    
    # Calculate OPEX
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
    
    st.subheader("Economic Metrics")
    st.dataframe(df[['ID','CAPEX','OPEX','Revenue','Profit']])
    
    st.subheader("Charts")
    fig1 = px.scatter(df, x='Production (MMcfge)', y='Revenue', size='Profit', color='Profit')
    fig2 = px.scatter(df, x='Production (MMcfge)', y='Profit', size='Revenue', color='Revenue')
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# PAGE 3: Reservoir Engineering Dashboard
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
# PAGE 4: Reservoir Prediction
# ---------------------------
elif page == "Reservoir Prediction":
    st.title("Predict New Well Production & Economics")
    
    st.subheader("Input Parameters")
    input_data = {}
    for col in feature_cols:
        input_data[col] = st.number_input(col, value=float(df[col].mean()))
    
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    pred_production = model.predict(input_scaled)[0]
    st.success(f"Predicted Production (MMcfge): {pred_production:.2f}")
    
    # Estimate CAPEX & OPEX
    capex = (
        base_drilling_cost * input_df['Depth (feet)'].iloc[0] +
        base_completion_cost * input_df['Gross Perforated Interval (ft)'].iloc[0] +
        proppant_cost_per_lb * input_df['Proppant per foot (lbs)'].iloc[0] * input_df['Gross Perforated Interval (ft)'].iloc[0] +
        water_cost_per_bbl * input_df['Water per foot (bbls)'].iloc[0] * input_df['Gross Perforated Interval (ft)'].iloc[0] +
        additive_cost_per_bbl * input_df['Additive per foot (bbls)'].iloc[0] * input_df['Gross Perforated Interval (ft)'].iloc[0]
    )
    
    opex = (
        base_maintenance_cost + base_pump_cost +
        proppant_cost_per_lb * input_df['Proppant per foot (lbs)'].iloc[0] * input_df['Gross Perforated Interval (ft)'].iloc[0] +
        water_cost_per_bbl * input_df['Water per foot (bbls)'].iloc[0] * input_df['Gross Perforated Interval (ft)'].iloc[0] +
        additive_cost_per_bbl * input_df['Additive per foot (bbls)'].iloc[0] * input_df['Gross Perforated Interval (ft)'].iloc[0]
    )
    
    revenue = pred_production * gas_price
    profit = revenue - capex - opex
    
    st.subheader("Economic Estimate")
    st.write(f"CAPEX: ${capex:,.2f}")
    st.write(f"OPEX: ${opex:,.2f}")
    st.write(f"Revenue: ${revenue:,.2f}")
    st.write(f"Profit: ${profit:,.2f}")




