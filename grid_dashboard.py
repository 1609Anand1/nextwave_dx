# Streamlit Grid L# Streamlit Grid Load Forecast Demo (Enhanced with More Features)
# Run with: streamlit run this_file.py

import streamlit as st
import pandas as pd
import numpy as np  # Streamlit Grid Load Forecast Demo (Enhanced with More Features)
# Run with: streamlit run this_file.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import random
from datetime import timedelta
import plotly.express as px

st.set_page_config(layout="wide", page_title="Grid Load Forecast Demo")

# --- Sidebar Controls ---
st.sidebar.title("⚙️ Generation Settings")
coal_capacity = st.sidebar.slider("Coal (MW)", 1000, 4000, 2000)
gas_capacity = st.sidebar.slider("Gas (MW)", 0, 2000, 800)
oil_capacity = st.sidebar.slider("Oil (MW)", 0, 2000, 0)
solar_peak = st.sidebar.slider("Solar Peak (MW)", 0, 1500, 800)
wind_min = st.sidebar.slider("Wind Min (MW)", 0, 500, 200)
wind_max = st.sidebar.slider("Wind Max (MW)", 500, 1500, 600)
bess_capacity = st.sidebar.slider("BESS (MW)", 0, 1000, 400)

st.title("⚡ Grid Load Forecast Dashboard")
st.markdown("""
This prototype forecasts grid load based on weather, time, and calendar features and compares it against available supply from various generation sources. 
Use the controls on the left to simulate your grid's generation profile.
""")

# --- Generate synthetic data ---
dates = pd.date_range(start="2024-12-01", periods=96, freq="15min")  # 1 day, 15-min interval
np.random.seed(42)
temperature = 15 + 10 * np.sin(np.linspace(0, 10 * np.pi, len(dates))) + np.random.normal(0, 1, len(dates))

# Add realistic features
day_of_week = [dt.weekday() for dt in dates]
hour_of_day = [dt.hour + dt.minute / 60 for dt in dates]
is_weekend = [1 if dt.weekday() >= 5 else 0 for dt in dates]
solar_radiation = [max(0, np.sin(np.pi * (h - 6) / 12)) * 1000 if 6 <= h <= 18 else 0 for h in hour_of_day]
wind_speed = 5 + 2 * np.sin(np.linspace(0, 5 * np.pi, len(dates))) + np.random.normal(0, 0.5, len(dates))

# Simulated load
df = pd.DataFrame({
    'timestamp': dates,
    'temperature': temperature,
    'day_of_week': day_of_week,
    'hour_of_day': hour_of_day,
    'is_weekend': is_weekend,
    'solar_radiation': solar_radiation,
    'wind_speed': wind_speed
})
df['load'] = 3000 + 200 * np.cos(np.linspace(0, 3 * np.pi, len(df))) - 8 * df['temperature'] \
              + 0.005 * df['solar_radiation'] - 15 * df['wind_speed'] + np.random.normal(0, 40, len(df))

# --- Train model ---
df.set_index('timestamp', inplace=True)
train_df = df.iloc[:-24]
test_df = df.iloc[-24:]

features = ['temperature', 'hour_of_day', 'day_of_week', 'is_weekend', 'solar_radiation', 'wind_speed']
X_train = train_df[features]
y_train = train_df['load']
X_test = test_df[features]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
load_pred = model.predict(X_test)

# --- Supply simulation ---
def simulate_supply(ts):
    hour = ts.hour + ts.minute / 60
    solar = max(0, solar_peak * np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
    wind = random.randint(wind_min, wind_max)
    bess = bess_capacity if random.random() > 0.5 else 0
    return coal_capacity + gas_capacity + oil_capacity + solar + wind + bess

supply = [simulate_supply(ts) for ts in test_df.index]

# --- Visualization ---
st.subheader("📈 Load Forecast vs Available Supply")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(test_df.index, test_df['load'], label='Actual Load', linestyle='--', marker='o')
ax.plot(test_df.index, load_pred, label='Predicted Load', linestyle='-', marker='x')
ax.plot(test_df.index, supply, label='Available Supply', linestyle='-', marker='s')
ax.fill_between(test_df.index, load_pred, supply, where=(np.array(supply) < load_pred), color='red', alpha=0.3, label='Deficit Risk')
ax.set_title('Forecasted Load vs Simulated Supply')
ax.set_xlabel('Time')
ax.set_ylabel('MW')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Generation mix pie chart ---
st.subheader("📊 Generation Mix")
gen_mix = pd.Series({
    'Coal': coal_capacity,
    'Gas': gas_capacity,
    'Oil': oil_capacity,
    'Solar (Peak)': solar_peak,
    'Wind (Avg)': (wind_min + wind_max) // 2,
    'BESS': bess_capacity
})
fig_pie = px.pie(values=gen_mix.values, names=gen_mix.index, title='Installed Generation Capacity', hole=0.4)
st.plotly_chart(fig_pie, use_container_width=True)

# --- Summary ---
st.subheader("📌 Summary")
deficit_hours = sum([s < l for s, l in zip(supply, load_pred)])
st.markdown(f"- 🔺 **Max Forecast Load:** {max(load_pred):.1f} MW")
st.markdown(f"- ✅ **Max Available Supply:** {max(supply):.1f} MW")
st.markdown(f"- ⚠️ **Hours with Deficit Risk:** {deficit_hours}")

# --- Detailed Table ---
st.markdown("---")
st.subheader("🔍 Hourly Forecast Table")
detail_df = pd.DataFrame({
    'Timestamp': test_df.index,
    'Forecast Load (MW)': load_pred,
    'Simulated Supply (MW)': supply,
    'Status': ['DEFICIT' if s < l else 'OK' for s, l in zip(supply, load_pred)]
})
st.dataframe(detail_df.set_index('Timestamp'), use_container_width=True)


