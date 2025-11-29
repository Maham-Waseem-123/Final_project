# ============================================
# STREAMLIT RESERVOIR ENGINEERING APP (NUMERIC DATA)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px

# ============================================
# 1. LOAD DATA
# ============================================

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Maham-Waseem-123/Final_project/main/Shale_Test.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip().str.replace("\n", "").str.replace("\xa0", "")
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(df.mean(), inplace=True)
    return df

df = load_data()

# ============================================
# 2. TRAIN MODEL
# ============================================

@st.cache_resource
def train_model(df):
    target = "Production (MMcfge)"
    feature_cols = df.drop(columns=["ID", target]).columns.tolist()
    numeric_cols = feature_cols
    X = df[feature_cols].copy()
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    gbr = GradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.1,
        n_estimators=600,
        max_depth=1,
        random_state=42,
        max_features=5
    )
    gbr.fit(X_train, y_train)
    
    return gbr, scaler, feature_cols, numeric_cols

model, scaler, feature_cols, numeric_cols = train_model(df)

# ============================================
# 3. STREAMLIT APP LAYOUT
# ============================================

st.set_page_config(page_title="Reservoir Engineering App", layout="wide")

st.sidebar.title("Pages")
page = st.sidebar.radio("Select a Page:", [
    "Reservoir Engineering Dashboard",
    "Reservoir Prediction",
    "Economic Analysis"
])

# ============================================
# PAGE 1: RESERVOIR ENGINEERING DASHBOARD
# ============================================

if page == "Reservoir Engineering Dashboard":
    
    st.title("Reservoir Engineering Dashboard")
    hover_cols = ["ID"]

    features_to_plot = [
        "Porosity", 
        "Additive per foot (bbls)",
        "Water per foot (bbls)", 
        "Proppant per foot (lbs)"
    ]

    def make_binned_lineplot(xcol, bins=10):
        df['bin'] = pd.cut(df[xcol], bins=bins)
        binned_df = df.groupby('bin', as_index=False)['Production (MMcfge)'].mean()
        binned_df['bin_center'] = binned_df['bin'].apply(lambda x: x.mid)
        binned_df = binned_df.dropna(subset=['Production (MMcfge)'])
        binned_df = binned_df.sort_values("bin_center")

        fig = px.line(
            binned_df,
            x='bin_center',
            y='Production (MMcfge)',
            labels={'bin_center': xcol, 'Production (MMcfge)': 'Production (MMcfge)'},
            markers=True
        )

        fig.update_xaxes(
            range=[binned_df['bin_center'].min(), binned_df['bin_center'].max()],
            dtick=(binned_df['bin_center'].max() - binned_df['bin_center'].min())/bins
        )
        fig.update_yaxes(title_text="Production (MMcfge)")

        st.subheader(f"Production vs {xcol}")
        st.plotly_chart(fig, use_container_width=True)

    for col in features_to_plot:
        make_binned_lineplot(col)

    st.subheader("Depth (feet) vs Production")
    df['Depth_bin'] = pd.cut(df["Depth (feet)"], bins=10)
    binned_depth_df = df.groupby('Depth_bin', as_index=False)['Production (MMcfge)'].mean()
    binned_depth_df['bin_center'] = binned_depth_df['Depth_bin'].apply(lambda x: x.mid)
    binned_depth_df = binned_depth_df.dropna(subset=['Production (MMcfge)'])
    binned_depth_df = binned_depth_df.sort_values("bin_center")

    fig = px.line(
        binned_depth_df,
        x='bin_center',
        y='Production (MMcfge)',
        labels={'bin_center': 'Depth (feet)', 'Production (MMcfge)': 'Production (MMcfge)'},
        markers=True
    )
    fig.update_xaxes(
        range=[binned_depth_df['bin_center'].min(), binned_depth_df['bin_center'].max()],
        dtick=(binned_depth_df['bin_center'].max() - binned_depth_df['bin_center'].min())/10
    )
    fig.update_yaxes(title_text="Production (MMcfge)")
    st.plotly_chart(fig, use_container_width=True)


# ============================================
# PAGE 2: RESERVOIR PREDICTION
# ============================================

elif page == "Reservoir Prediction":
    
    st.title("Predict New Well Production")
    st.subheader("Enter Well Parameters")

    input_data = {}
    for col in feature_cols:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        mean_val = float(df[col].mean())
        if min_val == max_val:
            max_val += 1.0
        input_data[col] = st.slider(col, min_val, max_val, mean_val)

    if st.button("Predict Production"):
        input_df = pd.DataFrame([input_data])
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        pred = model.predict(input_df)[0]
        st.success(f"Predicted Production (MMcfge): {pred:.2f}")
        st.session_state.predicted_production = pred

# ============================================
# PAGE 3: ECONOMIC ANALYSIS
# ============================================

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

    df["CAPEX"] = (
        base_drilling_cost * df["Depth (feet)"] +
        base_completion_cost * df["Gross Perforated Interval (ft)"] +
        proppant_cost_per_lb * df["Proppant per foot (lbs)"] * df["Gross Perforated Interval (ft)"] +
        water_cost_per_bbl * df["Water per foot (bbls)"] * df["Gross Perforated Interval (ft)"] +
        additive_cost_per_bbl * df["Additive per foot (bbls)"] * df["Gross Perforated Interval (ft)"]
    )
    
    df["OPEX"] = (
        base_maintenance_cost +
        base_pump_cost +
        proppant_cost_per_lb * df["Proppant per foot (lbs)"] * df["Gross Perforated Interval (ft)"] +
        water_cost_per_bbl * df["Water per foot (bbls)"] * df["Gross Perforated Interval (ft)"] +
        additive_cost_per_bbl * df["Additive per foot (bbls)"] * df["Gross Perforated Interval (ft)"]
    )
    
    df["Revenue"] = df["Production (MMcfge)"] * gas_price
    df["Profit"] = df["Revenue"] - df["CAPEX"] - df["OPEX"]
    
    st.subheader("Economic Metrics of Existing Wells")
    st.dataframe(df[['ID', 'CAPEX', 'OPEX', 'Revenue', 'Profit']])
    
    if "predicted_production" in st.session_state:
        st.subheader("Economic Metrics for Predicted Well")
        P = st.session_state.predicted_production
        new_capex = (
            base_drilling_cost * df["Depth (feet)"].mean() +
            base_completion_cost * df["Gross Perforated Interval (ft)"].mean() +
            proppant_cost_per_lb * df["Proppant per foot (lbs)"].mean() * df["Gross Perforated Interval (ft)"].mean() +
            water_cost_per_bbl * df["Water per foot (bbls)"].mean() * df["Gross Perforated Interval (ft)"].mean() +
            additive_cost_per_bbl * df["Additive per foot (bbls)"].mean() * df["Gross Perforated Interval (ft)"].mean()
        )
        new_opex = base_maintenance_cost + base_pump_cost
        new_revenue = P * gas_price
        new_profit = new_revenue - new_capex - new_opex
        
        st.write(f"Predicted Production: {P:.2f} MMcfge")
        st.write(f"CAPEX: ${new_capex:,.2f}")
        st.write(f"OPEX: ${new_opex:,.2f}")
        st.write(f"Revenue: ${new_revenue:,.2f}")
        st.write(f"Profit: ${new_profit:,.2f}")



