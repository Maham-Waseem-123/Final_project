import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
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
    
    # Existing wells economic calculation
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
    
    st.subheader("Economic Metrics of Existing Wells")
    st.dataframe(df[['ID','CAPEX','OPEX','Revenue','Profit']])
    
    # New well economics (from prediction)
    if 'predicted_production' in st.session_state:
        st.subheader("Economic Metrics for Predicted Well")
        new_prod = st.session_state.predicted_production
        new_capex = base_drilling_cost * df['Depth (feet)'].mean() + \
                    base_completion_cost * df['Gross Perforated Interval (ft)'].mean()
        new_opex = base_maintenance_cost + base_pump_cost
        new_revenue = new_prod * gas_price
        new_profit = new_revenue - new_capex - new_opex
        st.write(f"Predicted Production: {new_prod:.2f} MMcfge")
        st.write(f"CAPEX: ${new_capex:,.2f}")
        st.write(f"OPEX: ${new_opex:,.2f}")
        st.write(f"Revenue: ${new_revenue:,.2f}")
        st.write(f"Profit: ${new_profit:,.2f}")

elif page == "Reservoir Engineering Dashboard":
    st.title("Reservoir Engineering Dashboard")
    
    # -------------------------------
    # Clean column names
    # -------------------------------
    df.columns = df.columns.str.strip()           # remove leading/trailing spaces
    df.columns = df.columns.str.replace('\n','') # remove line breaks
    df.columns = df.columns.str.replace('\xa0','') # remove non-breaking spaces
    
    # -------------------------------
    # 1. Production vs Depth (highlight top 10% productive wells)
    # -------------------------------
    st.subheader("Production vs Depth")
    
    # Determine top 10% productive wells
    threshold = df['Production (MMcfge)'].quantile(0.90)
    df['Top_Production'] = np.where(df['Production (MMcfge)'] >= threshold, 'Top 10%', 'Others')
    
    fig_depth = px.scatter(df, 
                           x='Depth (feet)', 
                           y='Production (MMcfge)', 
                           color='Top_Production',
                           title="Production vs Depth (Top 10% Highlighted)",
                           labels={'Depth (feet)': 'Depth (ft)', 'Production (MMcfge)': 'Production (MMcfge)'},
                           hover_data=['ID', 'Porosity (decimal)'])
    
    # Add line connecting points for easier trend detection
    fig_depth.update_traces(mode='markers+lines', marker=dict(size=8))
    
    st.plotly_chart(fig_depth, use_container_width=True)
    
    # -------------------------------
    # 2. Production vs Porosity (scatter + regression)
    # -------------------------------
    st.subheader("Production vs Porosity")
    
    fig_porosity = px.scatter(df, 
                              x='Porosity (decimal)', 
                              y='Production (MMcfge)', 
                              trendline="ols",
                              title="Production vs Porosity",
                              labels={'Porosity (decimal)': 'Porosity', 'Production (MMcfge)': 'Production (MMcfge)'},
                              hover_data=['ID', 'Depth (feet)'])
    
    st.plotly_chart(fig_porosity, use_container_width=True)
    
    # -------------------------------
    # 3. Stimulation Effectiveness (3D)
    # -------------------------------
    st.subheader("Stimulation Effectiveness")
    
    fig_stim = px.scatter_3d(df,
                             x='Proppant per foot (lbs)',
                             y='Water per foot (bbls)',
                             z='Production (MMcfge)',
                             color='Additive per foot (bbls)',
                             size='Gross Perforated Interval (ft)',
                             title="Effect of Stimulation Parameters on Production",
                             labels={
                                 'Proppant per foot (lbs)': 'Proppant (lbs/ft)',
                                 'Water per foot (bbls)': 'Water (bbls/ft)',
                                 'Production (MMcfge)': 'Production (MMcfge)',
                                 'Additive per foot (bbls)': 'Additive (bbls/ft)',
                                 'Gross Perforated Interval (ft)': 'Perforated Interval (ft)'
                             },
                             hover_data=['ID', 'Depth (feet)', 'Porosity (decimal)'])
    
    st.plotly_chart(fig_stim, use_container_width=True)

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
        st.session_state.predicted_production = pred_production  # Save for economic analysis



