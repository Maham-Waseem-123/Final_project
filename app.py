# ============================================
# STREAMLIT RESERVOIR ENGINEERING APP
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px

# ============================================
# 1. LOAD DATA WITH INTERVAL CONVERSION
# ============================================

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Maham-Waseem-123/Final_project/main/finaldata.csv"
    df = pd.read_csv(url)

    # Clean column names
    df.columns = df.columns.str.strip().str.replace("\n", "").str.replace("\xa0", "")

    # Convert interval columns to midpoints
    def interval_to_midpoint(interval_str):
        if pd.isna(interval_str):
            return None
        match = re.match(r"[\(\[]([\d\.]+),\s*([\d\.]+)[\)\]]", str(interval_str))
        if match:
            low, high = map(float, match.groups())
            return (low + high) / 2
        else:
            try:
                return float(interval_str)
            except:
                return None

    interval_cols = [
        "Depth (feet)", "Thickness (feet)", "Normalized Gamma Ray (API)",
        "Density (g/cm3)", "Porosity (decimal)", "Resistivity (Ohm-m)"
    ]

    for col in interval_cols:
        df[col] = df[col].apply(interval_to_midpoint)

    # Ensure numeric columns for calculations
    numeric_cols_for_calc = [
        "Depth (feet)", "Gross Perforated Interval (ft)",
        "Proppant per foot (lbs)", "Water per foot (bbls)",
        "Additive per foot (bbls)"
    ]

    for col in numeric_cols_for_calc:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())

    return df

df = load_data()

# ============================================
# 2. TRAIN MODEL — LABEL ENCODING
# ============================================

@st.cache_resource
def train_model(df):

    target = "Production (MMcfge)"
    feature_cols = df.drop(columns=["ID", target]).columns.tolist()

    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    # Fill missing
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0]).astype(str)

    # Label Encode categorical columns
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[feature_cols].copy()
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale numeric columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Train GBRT
    gbr = GradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.1,
        n_estimators=600,
        max_depth=1,
        random_state=42,
        max_features=5
    )
    gbr.fit(X_train, y_train)

    return gbr, scaler, feature_cols, categorical_cols, numeric_cols, encoders

model, scaler, feature_cols, categorical_cols, numeric_cols, encoders = train_model(df)

# ============================================
# 3. STREAMLIT APP LAYOUT
# ============================================

st.set_page_config(page_title="Reservoir Engineering App", layout="wide")

st.sidebar.title("Pages")
page = st.sidebar.radio("Select a Page:", [
    "Economic Analysis",
    "Reservoir Engineering Dashboard",
    "Reservoir Prediction"
])

# ============================================
# PAGE 1: ECONOMIC ANALYSIS
# ============================================

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

    # CAPEX
    df["CAPEX"] = (
        base_drilling_cost * df["Depth (feet)"] +
        base_completion_cost * df["Gross Perforated Interval (ft)"] +
        proppant_cost_per_lb * df["Proppant per foot (lbs)"] * df["Gross Perforated Interval (ft)"] +
        water_cost_per_bbl * df["Water per foot (bbls)"] * df["Gross Perforated Interval (ft)"] +
        additive_cost_per_bbl * df["Additive per foot (bbls)"] * df["Gross Perforated Interval (ft)"]
    )

    # OPEX
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
        new_capex = base_drilling_cost * df["Depth (feet)"].mean() + \
                     base_completion_cost * df["Gross Perforated Interval (ft)"].mean()
        new_opex = base_maintenance_cost + base_pump_cost
        new_revenue = P * gas_price
        new_profit = new_revenue - new_capex - new_opex

        st.write(f"Predicted Production: {P:.2f} MMcfge")
        st.write(f"CAPEX: ${new_capex:,.2f}")
        st.write(f"OPEX: ${new_opex:,.2f}")
        st.write(f"Revenue: ${new_revenue:,.2f}")
        st.write(f"Profit: ${new_profit:,.2f}")

# ============================================
# PAGE 2: RESERVOIR ENGINEERING DASHBOARD
# ============================================

elif page == "Reservoir Engineering Dashboard":

    st.title("Reservoir Engineering Dashboard")
    df["Log_Production"] = np.log1p(df["Production (MMcfge)"])
    hover_cols = ["ID"]

    def make_lineplot(xcol, title):
        fig = px.line(
            df.sort_values(xcol),
            x=xcol,
            y="Log_Production",
            hover_data=hover_cols + [xcol, "Production (MMcfge)"],
            labels={"Log_Production": "Log(EUR + 1)"}
        )
        st.subheader(title)
        st.plotly_chart(fig, use_container_width=True)

    make_lineplot("Porosity (decimal)", "EUR vs Porosity")
    make_lineplot("Resistivity (Ohm-m)", "EUR vs Resistivity")
    make_lineplot("Additive per foot (bbls)", "EUR vs Additive per foot")
    make_lineplot("Water per foot (bbls)", "EUR vs Water per foot")
    make_lineplot("Proppant per foot (lbs)", "EUR vs Proppant per foot")
    make_lineplot("Gross Perforated Interval (ft)", "EUR vs Gross Perforated Interval")

    st.subheader("Depth (feet) vs Production (MMcfge)")
    fig = px.scatter(
        df,
        x="Depth (feet)",
        y="Log_Production",
        hover_data=hover_cols + ["Depth (feet)", "Production (MMcfge)"],
        labels={"Log_Production": "Log(EUR + 1)"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 3: RESERVOIR PREDICTION
# ============================================

elif page == "Reservoir Prediction":

    st.title("Predict New Well Production")
    st.subheader("Enter Well Parameters")

    input_data = {}
    for col in feature_cols:
        if col in numeric_cols:
            min_val, max_val = float(df[col].min()), float(df[col].max())
            mean_val = float(df[col].mean())
            if min_val == max_val:
                max_val += 1.0
            input_data[col] = st.slider(col, min_val, max_val, mean_val)
        else:
            input_data[col] = st.selectbox(col, sorted(df[col].astype(str).unique()))

    if st.button("Predict Production"):

        input_df = pd.DataFrame([input_data])

        # Encode categorical features
        for col in categorical_cols:
            le = encoders[col]

            # If user selects a value not seen in training
            if input_df[col].iloc[0] not in le.classes_:
                le.classes_ = np.append(le.classes_, input_df[col].iloc[0])

            input_df[col] = le.transform(input_df[col].astype(str))

        # Scale numeric
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Predict
        pred = model.predict(input_df)[0]
        st.success(f"Predicted Production (MMcfge): {pred:.2f}")
        st.session_state.predicted_production = pred
