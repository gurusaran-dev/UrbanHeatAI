import joblib
import numpy as np

model = joblib.load("UrbanHeat_XGBoost.pkl")

import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Urban Heat Mitigation AI",
    page_icon="🌍",
    layout="wide"
)

# Title
st.title("🌍 Urban Heat Mitigation AI System")
st.sidebar.header("Input Parameters")
st.success("✅ XGBoost Model Loaded Successfully")

st.info("📍 Study Area: Chennai, India")

st.caption("Hackathon Project: Optimizing Urban Heat Mitigation using AI & ML")

st.markdown("""
### AI-powered Decision Support System for Urban Cooling Strategies

This dashboard predicts Urban Heat Island intensity and recommends mitigation strategies using Machine Learning.
""")
st.divider()

st.header("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", "1.06 °C")
col2.metric("RMSE", "1.41 °C")
col3.metric("R² Score", "0.645")
import pandas as pd

df = pd.read_csv("Urban_Heat_Predictions.csv")
st.divider()

st.header("📄 Prediction Results")

st.dataframe(df)
st.divider()

st.header("🔥 Heat Risk Distribution")

risk_counts = df["Heat_Risk"].value_counts()

st.bar_chart(risk_counts)
st.divider()

st.header("📈 Dataset Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Total Locations", len(df))

col2.metric(
    "Average Predicted LST",
    f"{df['Predicted_LST'].mean():.2f} °C"
)

col3.metric(
    "Maximum Predicted LST",
    f"{df['Predicted_LST'].max():.2f} °C"
)
st.divider()

st.header("🚨 High Risk Locations")

high_risk = df[df["Heat_Risk"] == "High"]

st.dataframe(high_risk)
st.divider()

st.header("🌳 AI Cooling Recommendations")

st.dataframe(
    df[
        [
            "Predicted_LST",
            "Heat_Risk",
            "Recommendation"
        ]
    ]
)
ndvi = st.sidebar.slider("NDVI", -1.0, 1.0, 0.30)

ndbi = st.sidebar.slider("NDBI", -1.0, 1.0, 0.20)

elevation = st.sidebar.number_input(
    "Elevation (m)",
    value=20
)
landcover_dict = {
    "Tree Cover":10,
    "Shrubland":20,
    "Grassland":30,
    "Cropland":40,
    "Built-up":50,
    "Bare Soil":60,
    "Snow":70,
    "Water":80,
    "Wetland":90,
    "Mangroves":95,
    "Moss":100
}

landcover_name = st.sidebar.selectbox(
    "Land Cover",
    list(landcover_dict.keys())
)

landcover = landcover_dict[landcover_name]
sr_b2 = st.sidebar.number_input("SR_B2", value=0.10)

sr_b3 = st.sidebar.number_input("SR_B3", value=0.10)

sr_b4 = st.sidebar.number_input("SR_B4", value=0.10)

sr_b5 = st.sidebar.number_input("SR_B5", value=0.20)

sr_b6 = st.sidebar.number_input("SR_B6", value=0.30)

sr_b7 = st.sidebar.number_input("SR_B7", value=0.20)
if st.sidebar.button("🔍 Predict"):

    sample = np.array([[
        ndvi,
        ndbi,
        sr_b2,
        sr_b3,
        sr_b4,
        sr_b5,
        sr_b6,
        sr_b7,
        elevation,
        landcover
    ]])

    prediction = model.predict(sample)[0]

    st.header("Prediction")
    

    st.metric(
        "Predicted LST",
        f"{prediction:.2f} °C"
    )
    if prediction >= 38:
        risk = "🔴 HIGH"

    elif prediction >= 34:
        risk = "🟡 MEDIUM"

    else:
        risk = "🟢 LOW"

    st.subheader(f"Heat Risk : {risk}")
    st.subheader("🌳 Recommended Cooling Strategies")

    if prediction >= 38:

        st.success("✔ Install cool roofs")

        st.success("✔ Plant roadside trees")

        st.success("✔ Reflective pavements")

        st.success("✔ Increase green spaces")

    elif prediction >= 34:

        st.info("✔ Plant native trees")

        st.info("✔ Expand parks")

        st.info("✔ Increase vegetation")

    else:

        st.success("✔ Maintain existing green cover")
st.header("🧠 AI Urban Heat Insights")

avg_lst = df["Predicted_LST"].mean()

high_ratio = len(df[df["Heat_Risk"] == "High"]) / len(df) * 100

if avg_lst > 37:
    st.error("🔥 Chennai shows HIGH overall urban heat stress")

elif avg_lst > 34:
    st.warning("🌡 Moderate urban heat conditions detected")

else:
    st.success("🌿 Urban heat levels are relatively stable")
import numpy as np

st.header("🗺 Urban Heat Map (Simulated Layout)")

df["lat"] = np.random.uniform(12.90, 13.20, len(df))   # Chennai approx
df["lon"] = np.random.uniform(80.10, 80.30, len(df))
import pydeck as pdk

def get_color(risk):
    if risk == "High":
        return [255, 0, 0]
    elif risk == "Medium":
        return [255, 165, 0]
    else:
        return [0, 255, 0]

df["color"] = df["Heat_Risk"].apply(get_color)
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[lon, lat]',
    get_color='color',
    get_radius=80,
    pickable=True
)

view_state = pdk.ViewState(
    latitude=13.08,
    longitude=80.27,
    zoom=10
)

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))