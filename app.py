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
    url = "https://raw.githubusercontent.com/Maham-Waseem-123/Final_project/main/finaldata.csv"
    df = pd.read_csv(url)

    # Clean column names
    df.columns = df.columns.str.strip().str.replace("\n", "").str.replace("\xa0", "")

    return df


# ============================================
# 2. TRAIN MODEL — ONLY NUMERIC FEATURES
# ============================================

@st.cache_resource
def train_model(df):

    target = "Production (MMcfge)"
    feature_cols = df.drop(columns=["ID", target]).columns.tolist()

    # Auto-detect numeric & categorical columns
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    # Fill missing values
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].astype(str)   # keep as string

    # --------------------------
    # Training dataset
    # --------------------------
    X = df[feature_cols]
    y = df[target]

    # Convert categorical columns using one-hot encoding
    X = pd.get_dummies(X, columns=categorical_cols)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale numeric columns only
    scaler = StandardScaler()
    X_train[X_train.select_dtypes(include=[np.number]).columns] = \
        scaler.fit_transform(X_train.select_dtypes(include=[np.number]))

    X_test[X_test.select_dtypes(include=[np.number]).columns] = \
        scaler.transform(X_test.select_dtypes(include=[np.number]))

    # Train model
    gbr = GradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.1,
        n_estimators=600,
        max_depth=1,
        random_state=42,
        max_features=5
    )

    gbr.fit(X_train, y_train)

    return gbr, scaler, feature_cols, categorical_cols, numeric_cols


# ============================================
# 3. LOAD + TRAIN
# ============================================

df = load_data()
model, scaler, feature_cols, categorical_cols, numeric_cols = train_model(df)


# ============================================
# 4. STREAMLIT APP LAYOUT
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

    # Scatter plot for depth
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
            # Numeric slider
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())

            if min_val == max_val:
                max_val += 1.0

            input_data[col] = st.slider(
                col,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )

        else:
            # Categorical dropdown
            input_data[col] = st.selectbox(
                col,
                sorted(df[col].astype(str).unique())
            )

    # --------------- PREDICT ------------------
    if st.button("Predict Production"):

        input_df = pd.DataFrame([input_data])

        # One-hot encode categorical columns like training
        input_df = pd.get_dummies(input_df, columns=categorical_cols)

        # Add missing dummy columns (model expects them)
        train_dummy_cols = model.feature_names_in_

        for col in train_dummy_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[train_dummy_cols]

        # Scale numeric features like training
        input_df[input_df.select_dtypes(include=[np.number]).columns] = \
            scaler.transform(input_df.select_dtypes(include=[np.number]))

        # Predict
        pred = model.predict(input_df)[0]

        st.success(f"Predicted Production (MMcfge): {pred:.2f}")

        st.session_state.predicted_production = pred
