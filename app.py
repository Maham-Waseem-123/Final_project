# streamlit_reservoir_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pykrige.ok import OrdinaryKriging

# ------------------------------------------
# 1. App Header
# ------------------------------------------
st.set_page_config(page_title="2D Reservoir Property Mapping", layout="wide")
st.title("2D Reservoir Property Mapping Application 🚀")
st.markdown("Upload your well dataset, explore reservoir properties interactively, and test 'what-if' scenarios.")

# ------------------------------------------
# 2. Upload Data (once)
# ------------------------------------------
if 'df' not in st.session_state:
    uploaded_file = st.file_uploader("Upload CSV with well data", type=["csv"], key="upload_csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        # GBRT Production Prediction (once)
        feature_cols = [
            'Depth (feet)', 'Thickness (feet)', 'Normalized Gamma Ray (API)',
            'Density (g/cm3)', 'Porosity (decimal)', 'Resistivity (Ohm-m)',
            'Gamma Ray Stdev', 'Porosity Stdev', 'Resistivity Stdev', 'Density Stdev',
            'Gross Perforated Interval (ft)', 'Proppant per foot (lbs)',
            'Water per foot (bbls)', 'Additive per foot (bbls)',
            'Azimuth (degrees)', 'Acre Spacing (acres)'
        ]
        X = df[feature_cols]
        y = df['Production (MMcfge)']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        gbr = GradientBoostingRegressor(
            loss='absolute_error',
            learning_rate=0.1,
            n_estimators=600,
            max_depth=1,
            random_state=42,
            max_features=5
        )
        gbr.fit(X_scaled, y)
        df['Predicted Production'] = gbr.predict(X_scaled)

        st.session_state.df = df
        st.session_state.scaler = scaler
        st.session_state.gbr = gbr

        rmse = mean_squared_error(y, df['Predicted Production']) ** 0.5
        r2 = r2_score(y, df['Predicted Production'])
        st.success("Data Loaded and GBRT Model Trained Successfully!")
        st.markdown(f"**GBRT Model Performance:** RMSE = {rmse:.2f}, R² = {r2:.2f}")

# ------------------------------------------
# 3. Interactive Mapping & What-If Scenarios
# ------------------------------------------
if 'df' in st.session_state:
    df = st.session_state.df
    st.dataframe(df.head(), key="df_head")

    # Property selection
    property_list = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
    prop = st.selectbox("Select Property to Map:", property_list, key="select_property")

    # Interpolation method (only Kriging)
    st.info("Using Ordinary Kriging for Interpolation")
    
    # Add new well form (What-If)
    st.subheader("Add New Well (What-If Scenario)")
    with st.form(key="add_well_form"):
        new_well = {}
        for feature in ['Surface Latitude', 'Surface Longitude'] + property_list:
            new_well[feature] = st.number_input(
                f"{feature}", 
                value=float(df[feature].mean()), 
                key=f"form_{feature}"
            )
        submitted = st.form_submit_button("Add Well")
        if submitted:
            df_new = df.append(new_well, ignore_index=True)
            st.session_state.df = df_new
            st.success("New well added! Interpolation will include this well now.")
            df = df_new

    # Create a grid for interpolation
    x_min, x_max = df['Surface Longitude'].min(), df['Surface Longitude'].max()
    y_min, y_max = df['Surface Latitude'].min(), df['Surface Latitude'].max()
    grid_x = np.linspace(x_min, x_max, 100)
    grid_y = np.linspace(y_min, y_max, 100)

    # Ordinary Kriging
    OK = OrdinaryKriging(
        df['Surface Longitude'], df['Surface Latitude'], df[prop],
        variogram_model='spherical', verbose=False, enable_plotting=False
    )
    interpolated_values, _ = OK.execute('grid', grid_x, grid_y)

    # Plot property map
    fig = px.imshow(
        interpolated_values,
        origin='lower',
        x=grid_x,
        y=grid_y,
        labels={'x': 'Longitude', 'y': 'Latitude', 'color': prop},
        title=f"{prop} Distribution Map"
    )
    st.plotly_chart(fig, use_container_width=True, key="property_map")

    # Show Production Table
    st.subheader("Production & Prediction")
    st.dataframe(df[['ID', 'Production (MMcfge)', 'Predicted Production']], key="production_table")
